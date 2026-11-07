"""Bilinear Bayer demosaicing with normalised interpolation masks."""
import numpy as np
from scipy.ndimage import convolve
from core.raw.bayer import PATTERNS

def demosaic_bilinear(mosaic,pattern="RGGB"):
    m=np.asarray(mosaic,dtype=np.float32)
    if pattern not in PATTERNS: raise ValueError(pattern)
    if m.ndim!=2 or min(m.shape)<4 or m.shape[0]%2 or m.shape[1]%2: raise ValueError("Expected even HxW Bayer mosaic")
    positions={"RGGB":((0,0),(0,1),(1,0),(1,1)),"BGGR":((1,1),(1,0),(0,1),(0,0)),"GRBG":((0,1),(0,0),(1,1),(1,0)),"GBRG":((1,0),(1,1),(0,0),(0,1))}
    r,gr,gb,b=positions[pattern]
    masks=[np.zeros_like(m) for _ in range(3)]
    masks[0][r[0]::2,r[1]::2]=1
    masks[1][gr[0]::2,gr[1]::2]=1; masks[1][gb[0]::2,gb[1]::2]=1
    masks[2][b[0]::2,b[1]::2]=1
    krb=np.array([[1,2,1],[2,4,2],[1,2,1]],np.float32)
    kg=np.array([[0,1,0],[1,4,1],[0,1,0]],np.float32)
    result=[]
    for mask,k in zip(masks,(krb,kg,krb)):
        numerator=convolve(m*mask,k,mode="reflect")
        denominator=convolve(mask,k,mode="reflect")
        result.append(numerator/np.maximum(denominator,1e-8))
    return np.clip(np.stack(result,-1),0,1)

def demosaic_bilinear_rggb(m): return demosaic_bilinear(m,"RGGB")
