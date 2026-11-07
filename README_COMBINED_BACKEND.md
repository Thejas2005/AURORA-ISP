# AURORA — Combined Backend

A single FastAPI backend and browser interface for both input families.

## One server

```powershell
.
un_aurora.bat
```

Open http://127.0.0.1:8000

Or install manually:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Routing

- DNG / NPY / NPZ → Hybrid ISP RAW engine: sensor decode, CFA normalization, adaptive classical ISP, learned ONNX and hybrid ONNX.
- PNG / JPG / JPEG → rendered-RGB restoration: original, NLM denoise, detail-preserving denoise, and sequential full enhancement.
- Both routes use the same `/api/process` endpoint, health endpoint, frontend, and local server.
- RGB is never passed into the Bayer network.
- RAW model precision defaults to INT8. Set `AURORA_RAW_PRECISION=fp16` or `fp32` before launch to select another exported ONNX model.

## Model runtime

The current ONNX engine uses ONNX Runtime CPU execution. This package does not claim CUDA execution. It does not start a separate Gradio server or a second FastAPI server.

## Folder map

- `app/` unified FastAPI backend + UI
- `isp/` Hybrid ISP RAW decoder, adaptive classical pipeline, and analysis
- `hybrid_models/` exported learned/hybrid ONNX models
- `core/` retained RGB restoration and AURORA research components
- `models/checkpoints/` retained original AURORA checkpoint for reference

## Notes

DNG compatibility depends on camera RAW format support in LibRaw/rawpy. The backend rejects unsupported/non-Bayer DNGs with an explicit error. Metrics require an exactly aligned clean RGB reference.

## Expanded camera RAW compatibility

The RAW loader now keeps two distinct routes:

- Standard 2x2 RGB Bayer mosaics decoded by LibRaw continue through the native Bayer RAW pipeline (Traditional, Learned and Hybrid ONNX).
- Non-Bayer camera RAW layouts that LibRaw can render (for example X-Trans) are demosaiced by LibRaw to RGB and then processed through AURORA's RGB restoration/enhancement path. RGB/linear TIFF-based DNG containers are also attempted through the RGB decoder when LibRaw cannot open them.

The fallback is explicitly reported as `raw_rendered_rgb_fallback`; it is not represented as native Bayer reconstruction. Support is limited by the installed LibRaw build and the camera/file format. Some proprietary, encrypted, damaged, or otherwise unsupported files may still fail to decode.


## Performance reporting and mobile status

- The web interface measures end-to-end request time on the host computer (browser-to-local-backend round trip).
- The current inference backend reports CPU execution. CUDA/GPU acceleration is not enabled; the UI therefore reports GPU as not active.
- Mobile inference is not implemented or measured in this package. The device table is a benchmark plan, not a prediction. Do not present phone runtimes as measured results until the model is integrated with Android NNAPI or iOS Core ML and timed on physical devices.
- PSNR, SSIM, Delta E 2000 and LPIPS are shown only when a dimension-matched clean reference is supplied.
