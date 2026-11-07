"""Transparent heuristic controller. Thresholds are provisional, not trained."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Conditions:
    noise_sigma: float
    mean_luma: float
    clip_frac: float
    black_frac: float
    edge_density: float
    cast: float


@dataclass
class Params:
    denoise: float
    sharpen: float
    tone: float
    wb_strength: float
    skip_heavy: bool
    learned_weight: float = 0.0
    fusion_reason: str = "classical_only"


def estimate_noise_sigma(gray):
    h, w = gray.shape
    if min(h, w) < 4:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    response = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)[1:-1, 1:-1]
    return float(np.abs(response).sum() * np.sqrt(0.5 * np.pi) / (6 * response.size))


def analyse(rgb_lin):
    x = np.asarray(rgb_lin, dtype=np.float32)
    if (x.ndim != 3 or x.shape[-1] != 3 or min(x.shape[:2]) < 4
            or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1):
        raise ValueError("Expected finite HxWx3 linear RGB in [0,1], minimum 4x4")
    gray = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    means = x.reshape(-1, 3).mean(0) + 1e-6
    return Conditions(
        noise_sigma=estimate_noise_sigma(gray),
        mean_luma=float(gray.mean()),
        clip_frac=float((x.max(-1) > 0.98).mean()),
        black_frac=float((gray < 0.01).mean()),
        edge_density=float((np.hypot(gx, gy) > 0.1).mean()),
        cast=float(np.abs(means / means[1] - 1).max()),
    )


def decide(c):
    # Avoid skipping processing when substantial clipping or black crush is present.
    skip = (c.noise_sigma < 0.004 and c.mean_luma > 0.15
            and c.clip_frac < 0.02 and c.black_frac < 0.5)
    denoise_strength = 0.0 if skip else float(np.clip((c.noise_sigma - 0.004) / 0.05, 0.1, 1.0))
    denoise_strength *= 1.0 - 0.5 * min(c.edge_density * 4, 1.0)
    sharpen = 0.2 if c.noise_sigma > 0.03 else 0.5
    tone = 0.2 if c.clip_frac > 0.02 else (0.8 if c.mean_luma < 0.1 else 0.5)
    wb_strength = float(np.clip(c.cast * 2.0, 0.0, 1.0))
    return Params(denoise_strength, sharpen, tone, wb_strength, skip)
