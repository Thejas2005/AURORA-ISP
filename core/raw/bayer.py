"""Bayer helpers. No cropping or implicit sensor-layout changes."""
import numpy as np
PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")

def pack_bayer(mosaic, pattern="RGGB"):
    """HxW mosaic -> H/2 x W/2 x 4, ordered R, G-on-R-row, G-on-B-row, B."""
    if pattern not in PATTERNS: raise ValueError(f"Unknown Bayer pattern: {pattern}")
    m=np.asarray(mosaic)
    if m.ndim!=2 or m.shape[0]%2 or m.shape[1]%2: raise ValueError("Expected even HxW mosaic")
    a,b,c,d=(m[0::2,0::2],m[0::2,1::2],m[1::2,0::2],m[1::2,1::2])
    planes={"RGGB":(a,b,c,d),"BGGR":(d,c,b,a),"GRBG":(b,a,d,c),"GBRG":(c,d,a,b)}[pattern]
    return np.stack(planes,-1)

def unpack_bayer(packed, pattern="RGGB"):
    if pattern not in PATTERNS: raise ValueError(f"Unknown Bayer pattern: {pattern}")
    x=np.asarray(packed)
    if x.ndim!=3 or x.shape[-1]!=4: raise ValueError("Expected HxWx4")
    r,gr,gb,blue=[x[...,i] for i in range(4)]
    a,b,c,d={"RGGB":(r,gr,gb,blue),"BGGR":(blue,gb,gr,r),"GRBG":(gr,r,blue,gb),"GBRG":(gb,blue,r,gr)}[pattern]
    h,w=x.shape[:2]; out=np.empty((2*h,2*w),dtype=x.dtype)
    out[0::2,0::2]=a;out[0::2,1::2]=b;out[1::2,0::2]=c;out[1::2,1::2]=d
    return out

def pack_rggb(mosaic): return pack_bayer(mosaic,"RGGB")
def unpack_rggb(packed): return unpack_bayer(packed,"RGGB")

def to_rggb(mosaic, pattern):
    """Deprecated: pattern conversion by cropping changes geometry. Use pattern-aware demosaicing."""
    if pattern!="RGGB": raise ValueError("Do not crop to convert Bayer patterns; use pattern-aware processing")
    return np.asarray(mosaic)

def rgb_to_bayer(rgb,pattern="RGGB"):
    x=np.asarray(rgb)
    if x.ndim!=3 or x.shape[-1]!=3 or x.shape[0]%2 or x.shape[1]%2: raise ValueError("Expected even HxWx3")
    out=np.empty(x.shape[:2],dtype=x.dtype)
    tiles={"RGGB":((0,1),(1,2)),"BGGR":((2,1),(1,0)),"GRBG":((1,0),(2,1)),"GBRG":((1,2),(0,1))}
    if pattern not in tiles: raise ValueError(pattern)
    t=tiles[pattern]
    for y in (0,1):
        for xx in (0,1): out[y::2,xx::2]=x[y::2,xx::2,t[y][xx]]
    return out
