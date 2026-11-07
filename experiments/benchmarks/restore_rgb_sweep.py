"""Reproducible RGB denoising sweep. Parameter selection uses reference metrics;
report this as a reference-assisted benchmark, not blind deployment behaviour."""
from __future__ import annotations
import argparse, csv, time, sys
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.denoising.restoration import bm3d_rgb, wavelet_bayes_shrink, rgb_noise_diagnostics


def load(path):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p.resolve())
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def metrics(a,b):
    af=a.astype(np.float32)/255.; bf=b.astype(np.float32)/255.
    return {"PSNR":float(peak_signal_noise_ratio(bf,af,data_range=1.)),
            "SSIM":float(structural_similarity(bf,af,channel_axis=-1,data_range=1.)),
            "DeltaE2000":float(np.mean(deltaE_ciede2000(rgb2lab(af),rgb2lab(bf))))}


def save_float(x,path):
    x=np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all(): raise ValueError(f"Nonfinite output: {path}")
    # Exactly one conversion from [0,1] float to 8-bit.
    u8=np.round(np.clip(x,0,1)*255).astype(np.uint8)
    Image.fromarray(u8,"RGB").save(path)
    return u8


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--noisy",default="test_images/noisy_parrots.png")
    ap.add_argument("--reference",default="test_images/clean_parrots.jpeg")
    ap.add_argument("--output",default="outputs/aurora_modular_rgb")
    ap.add_argument("--sigmas",default="0.04,0.06,0.08,0.10,0.12,0.15,0.18,0.22")
    ap.add_argument("--skip-bm3d",action="store_true")
    args=ap.parse_args()
    noisy=load(args.noisy); ref=load(args.reference)
    if noisy.shape != ref.shape: raise ValueError(f"Shape mismatch: {noisy.shape} vs {ref.shape}; align explicitly, do not silently resize.")
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    save_float(noisy.astype(np.float32)/255.,out/"noisy_input.png")
    rows=[]
    base=metrics(noisy,ref); rows.append({"method":"noisy_input","sigma_psd":"","color_space":"","latency_ms":0,**base})
    diagnostics=rgb_noise_diagnostics(noisy.astype(np.float32)/255.)
    (out/"noise_diagnostics.json").write_text(__import__("json").dumps(diagnostics,indent=2))
    # Corrected wavelet candidate.
    t=time.perf_counter(); wave=wavelet_bayes_shrink(noisy.astype(np.float32)/255.); latency=(time.perf_counter()-t)*1000
    wu=save_float(wave,out/"wavelet_bayesshrink.png")
    rows.append({"method":"wavelet_bayesshrink","sigma_psd":"","color_space":"RGB","latency_ms":latency,**metrics(wu,ref)})
    if not args.skip_bm3d:
        for cs in ("rgb","ycbcr"):
            for sigma in [float(v) for v in args.sigmas.split(",")]:
                t=time.perf_counter()
                pred=bm3d_rgb(noisy.astype(np.float32)/255.,sigma,cs)
                latency=(time.perf_counter()-t)*1000
                name=f"bm3d_{cs}_sigma_{sigma:.3f}"
                pu=save_float(pred,out/f"{name}.png")
                rows.append({"method":name,"sigma_psd":sigma,"color_space":cs,"latency_ms":latency,**metrics(pu,ref)})
                print(f"{name}: PSNR={rows[-1]['PSNR']:.3f} SSIM={rows[-1]['SSIM']:.4f} DeltaE={rows[-1]['DeltaE2000']:.3f} latency={latency/1000:.2f}s")
    fields=["method","sigma_psd","color_space","PSNR","SSIM","DeltaE2000","latency_ms"]
    with (out/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    # A contact sheet for visual comparison.
    from PIL import ImageDraw
    files=[out/"noisy_input.png",out/"wavelet_bayesshrink.png"]+[p for p in sorted(out.glob("bm3d_*.png"))]
    thumbs=[]
    for p in files:
        im=Image.open(p).convert("RGB"); im.thumbnail((384,256))
        tile=Image.new("RGB",(400,292),"white"); tile.paste(im,((400-im.width)//2,8))
        ImageDraw.Draw(tile).text((8,270),p.stem,fill="black"); thumbs.append(tile)
    cols=2; rows_n=(len(thumbs)+cols-1)//cols
    sheet=Image.new("RGB",(cols*400,rows_n*292),(230,230,230))
    for i,tile in enumerate(thumbs): sheet.paste(tile,((i%cols)*400,(i//cols)*292))
    sheet.save(out/"contact_sheet.jpg",quality=92)
    print("Outputs:",out.resolve()); print("Metrics:",(out/"metrics.csv").resolve())


if __name__=="__main__": main()
