"""Input handling: real RAW (DNG/CR2/NEF/ARW... via rawpy), .npy Bayer, or sRGB photos to degrade."""
from __future__ import annotations

import os
import numpy as np
import cv2

from .raw_sim import BLACK_LEVEL, WHITE_LEVEL, even_crop, pack_rggb

RAW_EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAX_PIXELS = 1_200_000          # keeps the hosted demo comfortably under ~10 s


class InputError(ValueError):
    pass


def _srgb2xyz():
    return np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], np.float32)


def unpack_rggb(p: np.ndarray) -> np.ndarray:
    h, w, _ = p.shape
    raw = np.empty((2 * h, 2 * w), p.dtype)
    raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2] = (p[..., i] for i in range(4))
    return raw


def bin_bayer(raw: np.ndarray, max_pixels=MAX_PIXELS) -> np.ndarray:
    """Downscale an RGGB mosaic by averaging 2x2 same-colour samples (keeps the Bayer layout)."""
    while raw.shape[0] * raw.shape[1] > max_pixels:
        p = pack_rggb(even_crop(raw, 4)).astype(np.float32)
        h, w, _ = p.shape
        p = p.reshape(h // 2, 2, w // 2, 2, 4).mean((1, 3))
        raw = unpack_rggb(p)
    return raw


def load_camera_raw(path: str) -> dict:
    """Decode a camera RAW with rawpy, preserving per-CFA black levels and as-shot WB.

    The sensor mosaic is converted to the project's normalised RGGB convention.
    No rendered RGB demosaic is used here.
    """
    try:
        import rawpy
    except ImportError as e:
        raise InputError("DNG/RAW support requires rawpy. Install requirements.txt and restart.") from e

    # LibRaw can decode many camera RAW formats whose sensor layouts are not
    # conventional 2x2 Bayer (for example Fuji X-Trans). Those files cannot be
    # fed to our Bayer-trained ONNX models, but they can still be accepted by
    # asking LibRaw to perform its own camera-aware demosaic/render first.
    # Keep the true Bayer path unchanged for supported 2x2 RGB mosaics.
    try:
        with rawpy.imread(path) as r:
            is_standard_bayer = (
                r.raw_type == rawpy.RawType.Flat
                and r.num_colors == 3
                and getattr(r, "raw_pattern", None) is not None
                and np.asarray(r.raw_pattern).shape == (2, 2)
            )
            if not is_standard_bayer:
                try:
                    rendered = r.postprocess(
                        output_bps=8,
                        gamma=(1, 1),
                        no_auto_bright=False,
                        use_camera_wb=True,
                        output_color=rawpy.ColorSpace.sRGB,
                    )
                    rgb = np.asarray(rendered, dtype=np.float32) / 255.0
                    if rgb.ndim != 3 or rgb.shape[2] < 3 or min(rgb.shape[:2]) < 32:
                        raise ValueError(f"LibRaw returned invalid rendered dimensions: {rgb.shape}")
                    rgb = rgb[..., :3]
                    h, w = rgb.shape[:2]
                    scale = min(1.0, np.sqrt(MAX_PIXELS / (h * w)))
                    if scale < 1.0:
                        rgb = cv2.resize(rgb, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA)
                    rgb = even_crop(rgb, 8).astype(np.float32)
                    return dict(
                        srgb=rgb,
                        source=f"camera RAW ({os.path.splitext(path)[1][1:].upper()}) decoded by LibRaw demosaic/render; non-Bayer fallback",
                        raw_fallback=True,
                        fallback_reason="Sensor layout is not a standard 2x2 Bayer mosaic; camera-aware demosaic was used before RGB enhancement.",
                    )
                except Exception as render_error:
                    raise InputError(f"LibRaw recognised this as a non-standard RAW, but could not render it: {type(render_error).__name__}: {render_error}") from render_error

            raw = np.asarray(r.raw_image_visible, dtype=np.float32).copy()
            color_idx = np.asarray(r.raw_colors_visible, dtype=np.uint8).copy()
            pattern = np.asarray(r.raw_pattern[:2, :2], dtype=np.uint8)
            desc = r.color_desc.decode(errors="ignore")
            black_levels = np.asarray(r.black_level_per_channel, dtype=np.float32)
            white = float(r.white_level)
            xyz2cam = np.asarray(r.rgb_xyz_matrix[:3, :3], np.float32).copy()
            camera_wb = np.asarray(r.camera_whitebalance, np.float32).copy()
    except InputError:
        raise
    except Exception as e:
        # Some DNGs are linear/rendered TIFF containers rather than mosaiced
        # sensor RAW. Pillow can decode TIFF-based RGB DNGs; route those to the
        # RGB enhancement path instead of rejecting them as Bayer.
        if os.path.splitext(path)[1].lower() == ".dng":
            try:
                from PIL import Image
                with Image.open(path) as im:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                if rgb.ndim == 3 and rgb.shape[2] == 3 and min(rgb.shape[:2]) >= 32:
                    h, w = rgb.shape[:2]
                    scale = min(1.0, np.sqrt(MAX_PIXELS / (h * w)))
                    if scale < 1.0:
                        rgb = cv2.resize(rgb, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA)
                    return dict(
                        srgb=even_crop(rgb, 8).astype(np.float32),
                        source="RGB/linear DNG container decoded as rendered RGB",
                        raw_fallback=True,
                        fallback_reason="This DNG contains rendered/linear RGB rather than a Bayer sensor mosaic; RGB enhancement was used.",
                    )
            except Exception:
                pass
        raise InputError(f"RAW decoder could not read this file: {type(e).__name__}: {e}") from e

    if raw.ndim != 2 or color_idx.shape != raw.shape or min(raw.shape) < 32:
        raise InputError(f"RAW decoder returned an invalid visible sensor array: {raw.shape}.")
    if not np.isfinite(white) or white <= 0 or black_levels.size == 0:
        raise InputError("DNG has invalid black/white level metadata.")

    # Align using the *visible* CFA map. raw_pattern describes the sensor origin,
    # while raw_image_visible may begin after an odd-sized optical-black margin.
    # Using raw_pattern directly here can shift the Bayer phase and swap colour planes.
    red_code = next((i for i, c in enumerate(desc) if c == "R"), None)
    blue_code = next((i for i, c in enumerate(desc) if c == "B"), None)
    if red_code is None or blue_code is None:
        raise InputError("DNG does not expose a recognisable RGB Bayer colour description.")

    phase = color_idx[:2, :2]
    offset = None
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        if phase[dy, dx] == red_code and phase[1 - dy, 1 - dx] == blue_code:
            offset = (dy, dx)
            break
    if offset is None:
        raise InputError(
            f"Unsupported visible Bayer phase: {phase.tolist()} (colour description {desc!r})."
        )
    dy, dx = offset
    raw, color_idx = raw[dy:, dx:], color_idx[dy:, dx:]
    h, w = raw.shape[0] - raw.shape[0] % 2, raw.shape[1] - raw.shape[1] % 2
    raw, color_idx = raw[:h, :w], color_idx[:h, :w]

    # Apply the black level corresponding to each photosite's CFA colour.
    # This avoids the colour casts caused by averaging the four black levels.
    if black_levels.size <= int(color_idx.max()):
        raise InputError("DNG black-level metadata does not cover all CFA channels.")
    black_map = black_levels[color_idx]
    denom = white - black_map
    if np.any(denom <= 0):
        raise InputError("DNG black/white levels are inconsistent.")
    norm_unclipped = (raw - black_map) / denom
    # Catch a bad CFA/black-level interpretation before it becomes a black or
    # monochrome-looking image several stages downstream.
    if not np.isfinite(norm_unclipped).all():
        raise InputError("DNG produced non-finite sensor values after black-level correction.")
    norm = np.clip(norm_unclipped, 0.0, 1.0).astype(np.float32)
    near_black = float(np.mean(norm < 0.003))
    mean_signal = float(np.mean(norm))
    if near_black > 0.995 or mean_signal < 0.001:
        raise InputError(
            "DNG black-level normalization produced an almost entirely black Bayer frame "
            f"(mean={mean_signal:.5f}, near-black={near_black:.3f}). "
            "Check the camera's CFA/black-level metadata; refusing to render corrupted colour."
        )
    norm = bin_bayer(norm)
    dn = np.round(norm * (WHITE_LEVEL - BLACK_LEVEL) + BLACK_LEVEL).astype(np.uint16)

    from .sidd import ccm_from_colormatrix, fe_settings, generic_calibration
    ccm = None
    if xyz2cam.shape == (3, 3) and np.isfinite(xyz2cam).all() and np.abs(xyz2cam).sum() > 1e-6:
        ccm = (generic_calibration() @ ccm_from_colormatrix(xyz2cam)).astype(np.float32)

    # rawpy WB values are indexed by the same colour IDs used by raw_pattern.
    wb = None
    if camera_wb.size > int(color_idx.max()):
        r_codes = [i for i, c in enumerate(desc) if c == "R"]
        g_codes = [i for i, c in enumerate(desc) if c == "G"]
        b_codes = [i for i, c in enumerate(desc) if c == "B"]
        if r_codes and g_codes and b_codes:
            rg = float(np.mean(camera_wb[r_codes]))
            gg = float(np.mean(camera_wb[g_codes]))
            bg = float(np.mean(camera_wb[b_codes]))
            if np.isfinite([rg, gg, bg]).all() and min(rg, gg, bg) > 0:
                wb = (rg / gg, bg / gg)

    ext = os.path.splitext(path)[1][1:].upper()
    return dict(
        raw=dn, gt=None, fe=fe_settings(ccm, wb), cfa="rggb",
        source=f"camera RAW ({ext}), " + ("as-shot WB" if wb else "AWB estimated")
    )


def load_npz_frame(path: str) -> dict:
    """Packaged real-RAW example (e.g. a SIDD benchmark block): normalised RGGB raw,
    aligned ground-truth sRGB, camera id and original CFA (to restore orientation)."""
    from .sidd import norm_to_dn, fe_settings, camera_ccm, CFA
    z = np.load(path, allow_pickle=False)
    if "raw" not in z:
        raise InputError(".npz must contain a 'raw' array (normalised RGGB Bayer).")
    raw = z["raw"].astype(np.float32)
    if raw.ndim != 2 or min(raw.shape) < 32:
        raise InputError(f"'raw' must be a 2-D Bayer mosaic of at least 32x32, got {raw.shape}")
    raw = even_crop(raw, 2)
    cam = str(z["cam"]) if "cam" in z else ""
    gt = z["gt"].astype(np.float32) / 255 if "gt" in z else None
    if gt is not None:
        gt = even_crop(gt, 2)[: raw.shape[0], : raw.shape[1]]
    label = {"GP": "Google Pixel", "IP": "iPhone 7", "N6": "Nexus 6", "G4": "LG G4", "S6": "Galaxy S6"}.get(cam, "camera")
    wb = tuple(float(v) for v in z["wb"]) if "wb" in z else None
    return dict(raw=norm_to_dn(raw), gt=gt, fe=fe_settings(camera_ccm(cam) if cam else None, wb),
                cfa=str(z["cfa"]) if "cfa" in z else "rggb",
                source=f"real {label} RAW, " + (str(z["desc"]) if "desc" in z else "SIDD") +
                       (", as-shot WB" if wb else ", AWB estimated"))


def load_npy_bayer(path: str) -> dict:
    a = np.load(path)
    if a.ndim != 2:
        raise InputError(f".npy must be a 2-D RGGB Bayer mosaic, got shape {a.shape}")
    a = even_crop(a.astype(np.float32), 2)
    if a.max() <= 1.0:
        a = a * (WHITE_LEVEL - BLACK_LEVEL) + BLACK_LEVEL
    elif a.max() > WHITE_LEVEL:
        a = (a / a.max()) * (WHITE_LEVEL - BLACK_LEVEL) + BLACK_LEVEL
    a = bin_bayer((a - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL))
    from .sidd import fe_settings
    return dict(raw=np.round(a * (WHITE_LEVEL - BLACK_LEVEL) + BLACK_LEVEL).astype(np.uint16),
                gt=None, fe=fe_settings(None, None), cfa="rggb", source="Bayer .npy (RGGB, 10-bit assumed)")


def load_srgb(path_or_array) -> np.ndarray:
    if isinstance(path_or_array, np.ndarray):
        im = path_or_array
        if im.dtype != np.float32:
            im = im.astype(np.float32) / (255.0 if im.max() > 1 else 1.0)
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, 2)
        im = im[..., :3]
    else:
        im = cv2.imread(path_or_array, cv2.IMREAD_COLOR)
        if im is None:
            raise InputError("Could not decode the image file.")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = im.shape[:2]
    if min(h, w) < 32:
        raise InputError(f"Image too small ({w}x{h}); need at least 32x32.")
    s = min(1.0, np.sqrt(MAX_PIXELS / (h * w)))
    if s < 1:
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return even_crop(im, 8).astype(np.float32)


def load_any(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXT:
        return load_camera_raw(path)
    if ext == ".npy":
        return load_npy_bayer(path)
    if ext == ".npz":
        return load_npz_frame(path)
    if ext in IMG_EXT:
        return dict(srgb=load_srgb(path), source="sRGB photo -> simulated sensor RAW")
    # Extension lists differ across camera manufacturers. For other file
    # suffixes, let LibRaw identify the file by its contents.
    try:
        return load_camera_raw(path)
    except InputError as exc:
        raise InputError(f"Unsupported or undecodable image '{ext}'. LibRaw supports many camera RAW formats, but no decoder could read this file. Details: {exc}") from exc
