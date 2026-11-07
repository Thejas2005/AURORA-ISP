"""RGB restoration algorithms. All inputs/outputs are float32 RGB in [0,1]."""
from __future__ import annotations
import numpy as np
import cv2


def _validate(rgb):
    x = np.asarray(rgb, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite HxWx3 RGB image")
    if x.min() < 0 or x.max() > 1:
        raise ValueError("RGB values must be in [0,1]")
    return x


def wavelet_bayes_shrink(rgb, wavelet="db8", level=3):
    """Per-channel BayesShrink. Keeps float range; no uint8 scaling inside."""
    x = _validate(rgb)
    try:
        import pywt
    except ImportError as exc:
        raise RuntimeError("Install PyWavelets: python -m pip install PyWavelets") from exc

    out = np.empty_like(x)
    for c in range(3):
        coeffs = pywt.wavedec2(x[..., c], wavelet=wavelet, level=level, mode="symmetric")
        # Robust noise estimate from finest-scale diagonal detail.
        cH, cV, cD = coeffs[-1]
        sigma = np.median(np.abs(cD)) / 0.67448975
        shrunk = [coeffs[0]]
        for details in coeffs[1:]:
            new_details = []
            for band in details:
                var = max(float(np.mean(band * band) - sigma * sigma), 1e-12)
                threshold = sigma * sigma / np.sqrt(var)
                new_details.append(pywt.threshold(band, threshold, mode="soft"))
            shrunk.append(tuple(new_details))
        rec = pywt.waverec2(shrunk, wavelet=wavelet, mode="symmetric")
        out[..., c] = rec[:x.shape[0], :x.shape[1]]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def bm3d_rgb(rgb, sigma_psd, color_space="ycbcr"):
    """BM3D with explicit normalized sigma. Requires optional bm3d package."""
    x = _validate(rgb)
    try:
        from bm3d import bm3d, BM3DStages
    except ImportError as exc:
        raise RuntimeError("Install BM3D: python -m pip install bm3d") from exc
    sigma = float(sigma_psd)
    if not 0 < sigma <= 0.5:
        raise ValueError("sigma_psd must be in (0,0.5]")
    if color_space.lower() == "rgb":
        channels = [bm3d(x[..., c], sigma_psd=sigma, stage_arg=BM3DStages.ALL_STAGES) for c in range(3)]
        y = np.stack(channels, axis=-1)
    elif color_space.lower() == "ycbcr":
        # BT.601 transform in normalized range. Use lower chroma noise setting.
        ycc = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
        out = np.empty_like(ycc)
        out[..., 0] = bm3d(ycc[..., 0], sigma_psd=sigma, stage_arg=BM3DStages.ALL_STAGES)
        chroma_sigma = max(0.005, sigma * 0.65)
        out[..., 1] = bm3d(ycc[..., 1], sigma_psd=chroma_sigma, stage_arg=BM3DStages.ALL_STAGES)
        out[..., 2] = bm3d(ycc[..., 2], sigma_psd=chroma_sigma, stage_arg=BM3DStages.ALL_STAGES)
        y = cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_YCrCb2RGB)
    else:
        raise ValueError("color_space must be 'rgb' or 'ycbcr'")
    return np.clip(y, 0, 1).astype(np.float32)


def rgb_noise_diagnostics(rgb):
    """Robust high-frequency residual diagnostics; indicative, not ground-truth noise."""
    x = _validate(rgb)
    estimates = []
    for c in range(3):
        ch = x[..., c]
        lap = cv2.Laplacian(ch, cv2.CV_32F, ksize=1)
        # Median absolute deviation; edges/textures can inflate this estimate.
        estimates.append(float(np.median(np.abs(lap - np.median(lap))) / (0.67448975 * np.sqrt(20))))
    return {"noise_sigma_rgb_mad": estimates,
            "mean_rgb": [float(v) for v in x.reshape(-1,3).mean(0)],
            "clip_fraction": float(np.mean((x <= 1/255) | (x >= 254/255)))}
