"""AURORA Processing Studio local FastAPI backend."""
from __future__ import annotations
import base64, io, os, tempfile, time
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
import psutil
from fastapi.staticfiles import StaticFiles

from core.adaptive_controller.controller import analyse
from core.evaluation.metrics import all_metrics
from core.denoising.classical import denoise as rgb_nlm_denoise
from core.photofinishing.finish import sharpen

# Unified RAW engine (classical + learned + hybrid ONNX) from the Hybrid ISP.
from isp.io import load_any as load_hybrid_input, InputError as HybridInputError
from isp.engine import ISPEngine

ROOT = Path(__file__).resolve().parents[1]
_HYBRID_ENGINE = None

def get_hybrid_engine():
    global _HYBRID_ENGINE
    if _HYBRID_ENGINE is None:
        _HYBRID_ENGINE = ISPEngine(
            precision=os.environ.get("AURORA_RAW_PRECISION", "int8"),
            model_dir=str(ROOT / "hybrid_models")
        )
    return _HYBRID_ENGINE

app = FastAPI(title="AURORA Processing Studio", version="0.6.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent/"static"), name="static")


def load_rgb_file(filename: str, content: bytes):
    """Load ordinary display RGB/RGBA images. This is an RGB restoration path,
    not a Bayer RAW reconstruction path."""
    try:
        with Image.open(io.BytesIO(content)) as im:
            im = im.convert("RGB")
            if im.width < 16 or im.height < 16:
                raise ValueError("RGB image must be at least 16x16 pixels.")
            rgb = np.asarray(im, dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"Could not decode PNG/JPEG image: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Invalid RGB image.")
    return rgb.astype(np.float32)


def process_rgb_image(rgb: np.ndarray, mode: str):
    """Conservative RGB restoration for already-rendered PNG/JPEG inputs.

    Never sends display RGB through the Bayer RAW U-Net or RAW demosaicer.
    """
    if mode == "rgb_denoise":
        # Moderate NLM for a general-purpose denoising preview.
        # Strong default: the previous 0.55 mapped to h≈11 and left visible
        # chroma grain. 0.85 maps to h≈30 in the revised NLM implementation.
        return np.clip(rgb_nlm_denoise(rgb, 0.85), 0, 1).astype(np.float32)
    if mode == "rgb_denoise_detail":
        # Lower denoising strength with restrained sharpening to preserve texture.
        cleaned = rgb_nlm_denoise(rgb, 0.68)
        # Very restrained sharpening: avoid re-amplifying residual grain.
        return np.clip(sharpen(cleaned, 0.035), 0, 1).astype(np.float32)
    if mode == "rgb_full_enhance":
        # Full rendered-RGB enhancement chain: conservative gray-world WB,
        # adaptive exposure lift, local contrast on luminance, chroma-aware
        # denoising, restrained saturation and detail sharpening.
        # Stage 1: explicitly consume the RGB Denoise output. Full Enhance is
        # sequential, not an independent enhancement of the noisy source.
        x = process_rgb_image(rgb, "rgb_denoise")
        x = np.clip(x.astype(np.float32), 0.0, 1.0)
        # Conservative gray-world correction; cap gains to avoid colour casts
        # being over-corrected in strongly coloured scenes.
        means = np.maximum(x.mean(axis=(0, 1)), 1e-4)
        target = float(np.mean(means))
        gains = np.clip(target / means, 0.88, 1.14)
        x = np.clip(x * gains.reshape(1, 1, 3), 0.0, 1.0)

        # Adaptive exposure normalisation based on robust luminance percentiles.
        y = 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]
        p_low, p_mid, p_high = np.percentile(y, [1.0, 50.0, 99.5])
        if p_high - p_low > 1e-4:
            # Lift dark images toward a useful display midtone without forcing
            # already well-exposed images to the same brightness.
            target_mid = 0.42
            gamma = float(np.clip(np.log(max(target_mid, 1e-4)) /
                                  np.log(np.clip(p_mid, 1e-3, 0.999)), 0.62, 1.35))
            x = np.power(np.clip(x, 0.0, 1.0), gamma).astype(np.float32)

        # CLAHE on LAB luminance improves local contrast while preserving hue.
        lab = cv2.cvtColor(np.round(x*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        lab[...,0] = clahe.apply(lab[...,0])
        x = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

        # Finishing: restrained saturation and sharpening. Avoid a second
        # denoising pass, which can create waxy textures after the first NLM.
        hsv = cv2.cvtColor(np.round(np.clip(x,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[...,1] = np.clip(hsv[...,1].astype(np.float32)*1.06, 0, 255).astype(np.uint8)
        x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)/255.0
        return np.clip(sharpen(x, 0.07), 0.0, 1.0).astype(np.float32)
    if mode == "rgb_original":
        return rgb.copy()
    raise ValueError(f"Unsupported RGB mode: {mode}")


def png_data_url(rgb):
    x=np.asarray(rgb,dtype=np.float32)
    buf=io.BytesIO()
    Image.fromarray(np.round(np.clip(x,0,1)*255).astype(np.uint8),"RGB").save(buf,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode("ascii")


def reference_array(filename: str, content: bytes):
    """Decode an optional reference image to display-referred RGB for metrics.

    Unlike the primary input selector, this path is deliberately independent:
    it never enters the enhancement pipeline. Pillow handles rendered formats;
    LibRaw handles camera RAW reference files and renders them to sRGB.
    """
    suffix = Path(filename or "reference").suffix.lower()
    try:
        with Image.open(io.BytesIO(content)) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            return rgb
    except Exception:
        pass

    # RAW reference: use LibRaw via rawpy, including Bayer and non-Bayer CFA
    # formats. Metrics compare the rendered RGB target to the ISP output.
    try:
        import rawpy
        with tempfile.NamedTemporaryFile(suffix=suffix or ".raw", delete=False) as tmp:
            tmp.write(content)
            temp_path = tmp.name
        try:
            with rawpy.imread(temp_path) as raw:
                rendered = raw.postprocess(output_bps=8, gamma=(2.222, 4.5),
                                           no_auto_bright=True, use_camera_wb=True)
            return np.asarray(rendered, dtype=np.float32) / 255.0
        finally:
            try: os.unlink(temp_path)
            except OSError: pass
    except Exception as exc:
        raise ValueError(f"Could not decode reference image '{filename}' as RGB or camera RAW: {exc}") from exc


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent/"static"/"index.html")


@app.get("/api/health")
def health():
    raw_models = {name: (ROOT/"hybrid_models"/f"{name}_{os.environ.get('AURORA_RAW_PRECISION','int8')}.onnx").is_file()
                  for name in ("learned", "hybrid")}
    return {"status":"ok", "device":"cpu", "gpu_available":False,
            "precision":os.environ.get("AURORA_RAW_PRECISION", "int8").upper(),
            "checkpoint_exists":any(raw_models.values()),
            "checkpoint":str(ROOT/"hybrid_models"), "model_loaded":_HYBRID_ENGINE is not None,
            "raw_models":raw_models, "rgb_backend":"enabled"}


def performance_snapshot(start_time: float, image_shape, precision=None):
    """Host-side request and process resource measurements; not phone estimates."""
    proc = psutil.Process(os.getpid())
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    h, w = int(image_shape[0]), int(image_shape[1])
    mpix_s = ((h * w) / 1_000_000) / max(elapsed_ms / 1000.0, 1e-6)
    return {
        "end_to_end_ms": round(elapsed_ms, 2),
        "throughput_mpix_s": round(mpix_s, 2),
        "process_ram_mb": round(rss_mb, 1),
        "system_cpu_percent": round(psutil.cpu_percent(interval=None), 1),
        "cpu_utilization_percent": round(proc.cpu_percent(interval=None), 1),
        "gpu": "Not enabled (CPUExecutionProvider)",
        "vram_mb": None,
        "model_precision": precision or "N/A",
    }


@app.post("/api/process")
async def process(file: UploadFile=File(...), reference: Optional[UploadFile]=File(None),
                  pattern: str=Form("AUTO"), modes: str=Form("traditional,learned,hybrid")):
    request_start = time.perf_counter()
    # Prime the per-process CPU counter so the response reports CPU used during this request.
    psutil.Process(os.getpid()).cpu_percent(interval=None)
    try:
        filename = file.filename or "input"
        suffix = Path(filename).suffix.lower()
        content = await file.read()
        selected = [m.strip() for m in modes.split(",") if m.strip()]
        results = {}

        # Ordinary PNG/JPEG images are already rendered RGB. They use a separate
        # RGB restoration path; they are never misrepresented as Bayer RAW input.
        if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"):
            rgb = load_rgb_file(filename, content)
            ref = reference_array(reference.filename or "reference", await reference.read()) if reference else None
            if ref is not None and ref.shape != rgb.shape:
                raise ValueError(f"Reference dimensions {ref.shape[:2]} must exactly match input RGB dimensions {rgb.shape[:2]}. No resizing/alignment is performed.")
            allowed = {"rgb_original", "rgb_denoise", "rgb_denoise_detail", "rgb_full_enhance"}
            if not selected or any(m not in allowed for m in selected):
                raise ValueError("For PNG/JPEG inputs choose RGB Original, RGB Denoise or RGB Detail-Preserving Denoise. Bayer RAW modes cannot process display RGB.")
            # Reuse controller analysis on the RGB image, which is already sRGB.
            conditions = analyse(rgb)
            for mode in selected:
                pred = process_rgb_image(rgb, mode)
                item = {"image": png_data_url(pred), "shape": list(pred.shape)}
                if ref is not None:
                    item["metrics"] = all_metrics(pred, ref)
                results[mode] = item
            return {
                "filename": filename, "input_type": "display_rgb",
                "input_shape": list(rgb.shape[:2]), "metadata": {"encoding": "sRGB"},
                "conditions": {k: float(getattr(conditions, k)) for k in
                    ("noise_sigma", "mean_luma", "clip_frac", "black_frac", "edge_density", "cast")},
                "device": "cpu", "checkpoint": None, "results": results,
                "performance": performance_snapshot(request_start, rgb.shape[:2]),
                "warning": "PNG/JPEG is already-rendered RGB, not sensor RAW. The Bayer U-Net, demosaicing, sensor white balance and RAW colour matrix are not applied. Metrics require a separate matched clean RGB reference."
            }

        # Bayer RAW formats use the Hybrid ISP engine. This is the same FastAPI
        # backend as the rendered-RGB restoration route above.
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            temp_path = tmp.name
        try:
            loaded = load_hybrid_input(temp_path)
        finally:
            try: os.unlink(temp_path)
            except OSError: pass
        # Non-Bayer camera RAW (e.g. X-Trans) and RGB/linear DNG containers
        # are still accepted. LibRaw/Pillow renders them to RGB, after which
        # the rendered-RGB restoration path is used. Never mislabel this as
        # Bayer RAW or send it through Bayer-trained ONNX models.
        if "srgb" in loaded:
            rgb = loaded["srgb"]
            ref = reference_array(reference.filename or "reference", await reference.read()) if reference else None
            if ref is not None and ref.shape != rgb.shape:
                raise ValueError(f"Reference dimensions {ref.shape[:2]} must exactly match decoded RGB dimensions {rgb.shape[:2]}.")
            mode_map = {"traditional": "rgb_original", "learned": "rgb_denoise", "hybrid": "rgb_full_enhance"}
            if not selected or any(m not in mode_map for m in selected):
                raise ValueError("Select Traditional, Learned and/or Hybrid to run the compatible RGB fallback.")
            conditions = analyse(rgb)
            results = {}
            for mode in selected:
                rgb_mode = mode_map[mode]
                pred = process_rgb_image(rgb, rgb_mode)
                item = {"image": png_data_url(pred), "shape": list(pred.shape),
                        "pipeline": f"RGB fallback: {rgb_mode}"}
                if ref is not None:
                    item["metrics"] = all_metrics(pred, ref)
                results[mode] = item
            return {
                "filename": filename, "input_type": "raw_rendered_rgb_fallback",
                "input_shape": list(rgb.shape[:2]),
                "metadata": {"source": loaded.get("source", "camera RAW rendered to RGB"), "cfa": "not applicable"},
                "conditions": {k: float(getattr(conditions, k)) for k in
                    ("noise_sigma", "mean_luma", "clip_frac", "black_frac", "edge_density", "cast")},
                "device": "cpu", "checkpoint": None, "results": results,
                "performance": performance_snapshot(request_start, rgb.shape[:2]),
                "warning": loaded.get("fallback_reason", "This RAW sensor layout is not standard Bayer. LibRaw rendered it to RGB; Bayer demosaicing and Bayer-trained neural models were not used."),
            }
        if "raw" not in loaded:
            raise ValueError("Input could not be decoded as Bayer RAW or rendered RGB.")
        raw = loaded["raw"]
        fe = loaded.get("fe", {})
        ref = reference_array(reference.filename or "reference", await reference.read()) if reference else None
        allowed = {"traditional", "learned", "hybrid"}
        if not selected or any(m not in allowed for m in selected):
            raise ValueError("For Bayer RAW inputs choose Traditional, Learned and/or Hybrid ISP.")
        eng = get_hybrid_engine()
        results = {}
        # Run classical first: it supplies measured sensor/controller information
        # shared by the learned and hybrid paths.
        classical_out, info, classical_times = eng.classical(raw, **fe)
        conditions = analyse(np.clip(classical_out, 0, 1))
        for mode in selected:
            if mode == "traditional":
                pred, mode_info, times = classical_out, info, classical_times
            elif mode == "learned":
                if not eng.available("learned"):
                    raise ValueError("Learned ONNX model is missing from hybrid_models.")
                pred, mode_info, times = eng.learned(raw, info, **fe)
            else:
                if not eng.available("hybrid"):
                    raise ValueError("Hybrid ONNX model is missing from hybrid_models.")
                pred, mode_info, times = eng.hybrid(raw, **fe)
            item = {"image": png_data_url(pred), "shape": list(pred.shape),
                    "latency_ms": round(sum(times.values()), 2)}
            if ref is not None:
                if ref.shape != pred.shape:
                    raise ValueError(f"Reference RGB dimensions {ref.shape[:2]} must match rendered output dimensions {pred.shape[:2]} for {mode}. No resizing/alignment is performed.")
                item["metrics"] = all_metrics(pred, ref)
            if mode == "hybrid" and mode_info:
                item["fusion"] = {"reason":"Hybrid ONNX restoration with classical RAW front-end"}
            results[mode] = item
        return {"filename":filename, "input_type":"bayer_raw", "input_shape":list(raw.shape),
                "metadata":{"source":loaded.get("source","camera RAW"), "cfa":"RGGB canonical"},
                "conditions":{k:float(getattr(conditions,k)) for k in
                    ("noise_sigma","mean_luma","clip_frac","black_frac","edge_density","cast")},
                "device":"cpu", "checkpoint":{"precision":eng.precision,
                    "models":{n:eng.available(n) for n in ("learned","hybrid")}},
                "performance": performance_snapshot(request_start, classical_out.shape[:2], eng.precision),
                "results":results,
                "warning":"DNG/NPY/NPZ use the Hybrid ISP Bayer pipeline. PNG/JPEG use the separate rendered-RGB restoration pipeline."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
