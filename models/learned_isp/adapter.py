"""Bayer-mosaic -> trained AuroraUNet -> camera-rendered sRGB adapter."""
import numpy as np
import torch
from core.raw.bayer import pack_bayer
from core.metadata import metadata_vector

class LearnedISP:
    def __init__(self, model, device=None):
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, mosaic, meta):
        mosaic = np.asarray(mosaic, dtype=np.float32)
        if mosaic.ndim != 2 or min(mosaic.shape) < 8 or any(v % 2 for v in mosaic.shape):
            raise ValueError("Learned ISP requires an even-sized Bayer mosaic of at least 8x8")
        if not np.isfinite(mosaic).all():
            raise ValueError("RAW mosaic contains nonfinite values")
        # SIDD-CC values are already black-level-subtracted and normalised; preserve negatives.
        packed = pack_bayer(mosaic, meta.get("pattern", "RGGB"))
        tensor = torch.from_numpy(np.ascontiguousarray(packed.transpose(2,0,1)[None])).to(self.device)
        metadata = torch.from_numpy(metadata_vector(meta)[None]).to(self.device)
        output = self.model(tensor, metadata)[0].permute(1,2,0).float().cpu().numpy()
        if output.shape != (*mosaic.shape,3) or not np.isfinite(output).all():
            raise RuntimeError("Learned adapter returned invalid sRGB")
        return np.clip(output,0,1).astype(np.float32)
