"""Three-mode same-input benchmark. Synthetic smoke mode is not quality evaluation."""
import argparse,time,json
import numpy as np
from core.pipeline import traditional_isp,learned_isp,hybrid_isp

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--size",type=int,default=256); ap.add_argument("--checkpoint")
    a=ap.parse_args()
    raw=np.random.default_rng(7).random((a.size,a.size),dtype=np.float32)
    meta={"pattern":"RGGB","black_level":0,"white_level":1}
    learned=None
    if a.checkpoint:
        import torch
        from models.learned_isp.unet import AuroraUNet
        from models.learned_isp.adapter import LearnedISP
        model=AuroraUNet(); state=torch.load(a.checkpoint,map_location="cpu",weights_only=True); model.load_state_dict(state); learned=LearnedISP(model)
    results={}
    for name,fn in [("traditional",lambda:traditional_isp(raw,meta)),("learned",lambda:learned_isp(raw,meta,learned)),("hybrid",lambda:hybrid_isp(raw,meta,learned)[0])]:
        if name=="learned" and learned is None:
            results[name]={"status":"not_run","reason":"trained checkpoint not supplied"}; continue
        t=time.perf_counter(); out=fn(); elapsed=(time.perf_counter()-t)*1000
        results[name]={"status":"smoke_only","latency_ms":elapsed,"shape":list(out.shape),"dtype":str(out.dtype)}
    print(json.dumps({"note":"Synthetic random input; not a quality benchmark.","results":results},indent=2))
if __name__=="__main__": main()
