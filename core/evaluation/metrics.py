"""Image metrics for sRGB float images in [0,1]; validates contracts."""
import numpy as np
from skimage.metrics import structural_similarity
from core.colour.colour import srgb_to_lab

def _pair(a,b):
    a=np.asarray(a,dtype=np.float32); b=np.asarray(b,dtype=np.float32)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[-1]!=3: raise ValueError("Expected matching HxWx3 images")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or a.min()<0 or b.min()<0 or a.max()>1 or b.max()>1: raise ValueError("Metrics expect finite sRGB [0,1]")
    return a,b
def psnr(a,b):
    a,b=_pair(a,b); mse=float(np.mean((a-b)**2)); return float("inf") if mse==0 else float(10*np.log10(1/mse))
def ssim(a,b):
    a,b=_pair(a,b)
    if min(a.shape[:2])<7: raise ValueError("SSIM requires spatial dimensions >= 7")
    return float(structural_similarity(a,b,channel_axis=-1,data_range=1.0))
def delta_e2000(a,b):
    a,b=_pair(a,b)
    from skimage.color import deltaE_ciede2000
    return float(np.mean(deltaE_ciede2000(srgb_to_lab(a),srgb_to_lab(b))))
def lpips_score(a,b):
    a,b=_pair(a,b)
    try: import torch,lpips
    except ImportError: return None
    if not hasattr(lpips_score,"model"): lpips_score.model=lpips.LPIPS(net="alex").eval()
    t=lambda x: torch.from_numpy(x.transpose(2,0,1)[None])*2-1
    with torch.inference_mode(): return float(lpips_score.model(t(a),t(b)).item())
def all_metrics(pred,ref):
    return {"PSNR":psnr(pred,ref),"SSIM":ssim(pred,ref),"DeltaE2000":delta_e2000(pred,ref),"LPIPS":lpips_score(pred,ref)}
