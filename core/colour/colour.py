import numpy as np

def gray_world_gains(rgb):
    means = rgb.reshape(-1, 3).mean(0) + 1e-6
    return (means[1] / means).astype(np.float32)

def apply_wb(rgb, gains):
    return np.clip(rgb * np.asarray(gains, np.float32), 0, 1)

def apply_ccm(rgb, ccm):
    return np.clip(rgb @ np.asarray(ccm, np.float32).T, 0, 1)

def srgb_encode(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)

def srgb_decode(s):
    s = np.clip(s, 0, 1)
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)

def srgb_to_lab(srgb):
    """sRGB [0,1] (D65) -> CIE L*a*b*."""
    lin = srgb_decode(srgb)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)
