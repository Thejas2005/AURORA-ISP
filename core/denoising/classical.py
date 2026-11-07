import cv2, numpy as np

def denoise(rgb, strength):
    """Adaptive-strength edge-preserving NLM denoising for RGB float [0,1].

    `strength` is a user-facing [0,1] control. The previous mapping produced
    h<=18 even at maximum strength, which was too weak for severe chroma noise.
    This mapping reaches h=35 while retaining the original image dimensions.
    """
    if strength <= 0.01:
        return np.asarray(rgb, dtype=np.float32)
    x = np.asarray(rgb, dtype=np.float32)
    u8 = np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)
    strength = float(np.clip(strength, 0.0, 1.0))
    h_luma = 5.0 + 30.0 * strength
    h_chroma = 6.0 + 32.0 * strength
    # Larger search window is more effective on spatially distributed sensor
    # grain; template size 7 retains more small structures than template 9.
    out = cv2.fastNlMeansDenoisingColored(
        u8, None, h_luma, h_chroma, templateWindowSize=7, searchWindowSize=21
    )
    return out.astype(np.float32) / 255.0
