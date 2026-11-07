import numpy as np

def normalise(mosaic,black_level,white_level):
    """Normalise sensor samples. Scalars or HxW broadcastable level maps accepted."""
    x=np.asarray(mosaic,dtype=np.float32)
    black=np.asarray(black_level,dtype=np.float32)
    white=np.asarray(white_level,dtype=np.float32)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(black)) or not np.all(np.isfinite(white)):
        raise ValueError("Nonfinite RAW values or levels")
    if np.any(white<=black): raise ValueError("white_level must exceed black_level")
    return np.clip((x-black)/(white-black),0,1)
