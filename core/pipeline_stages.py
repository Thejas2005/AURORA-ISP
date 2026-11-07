"""Inspectable, modular Bayer ISP stages; learned model contract is unchanged."""
from __future__ import annotations
import numpy as np
from core.preprocessing.raw_prep import normalise
from core.demosaicing.bilinear import demosaic_bilinear
from core.colour.colour import gray_world_gains, apply_wb, apply_ccm
from core.photofinishing.finish import finish


def run_sensor_frontend(mosaic, metadata):
    raw = np.asarray(mosaic)
    if raw.ndim != 2 or min(raw.shape) < 4 or any(v % 2 for v in raw.shape):
        raise ValueError("Expected even-sized 2-D Bayer mosaic")
    if not np.isfinite(raw).all():
        raise ValueError("RAW mosaic contains nonfinite values")
    normal = normalise(raw, metadata.get("black_level", 0), metadata.get("white_level", 1))
    return {"normalised_raw": normal,
            "demosaiced_rgb": demosaic_bilinear(normal, metadata.get("pattern", "RGGB"))}


def run_colour_stage(rgb, metadata, wb_strength=1.0):
    x = np.asarray(rgb, dtype=np.float32)
    gains = metadata.get("wb_gains")
    gains = gray_world_gains(x) if gains is None else np.asarray(gains, np.float32)
    if gains.shape != (3,) or not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("wb_gains must be three finite positive values")
    blend = 1.0 + float(np.clip(wb_strength, 0, 1)) * (gains - 1.0)
    balanced = apply_wb(x, blend)
    ccm = np.asarray(metadata.get("ccm", np.eye(3)), dtype=np.float32)
    if ccm.shape != (3,3) or not np.isfinite(ccm).all():
        raise ValueError("ccm must be a finite 3x3 matrix")
    return apply_ccm(balanced, ccm)


def run_photofinishing(rgb_linear, tone=0.5, sharpen=0.25):
    return finish(rgb_linear, tone=float(tone), sharp=float(sharpen))


def run_classical_stages(mosaic, metadata, denoise_fn=None):
    """Return named intermediate buffers for debugging/demo visualisation."""
    from core.denoising.classical import denoise
    stages = run_sensor_frontend(mosaic, metadata)
    stages["colour_corrected_rgb"] = run_colour_stage(stages["demosaiced_rgb"], metadata)
    fn = denoise_fn or (lambda image: denoise(image, 0.3))
    stages["denoised_rgb"] = fn(stages["colour_corrected_rgb"])
    stages["final_srgb"] = run_photofinishing(stages["denoised_rgb"])
    return stages
