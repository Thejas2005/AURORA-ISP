import cv2, numpy as np
from core.colour.colour import srgb_encode

def tone_map(lin, strength=0.5):
    k = 0.6 * strength
    return np.clip(lin * (1 + k) / (1 + k * lin), 0, 1)

def sharpen(rgb, amount):
    if amount <= 0.01:
        return rgb
    return np.clip(rgb + amount * (rgb - cv2.GaussianBlur(rgb, (0, 0), 1.2)), 0, 1)

def finish(lin, tone=0.5, sharp=0.4):
    return sharpen(srgb_encode(tone_map(lin, tone)).astype(np.float32), sharp)
