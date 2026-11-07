# AURORA Processing Studio v6 — implementation status

This release follows `AURORA_ISP_Source_of_Truth_v1.md` and keeps the project focused on P2 smartphone ISP requirements.

## Decided and preserved
- Project remains a hybrid Bayer RAW-to-RGB smartphone ISP, not a generic photo editor.
- Traditional, Learned and Hybrid are independently selectable.
- Existing AuroraUNet architecture, five-value metadata contract and v4 checkpoint files are unchanged.
- Local processing is preferred; no cloud inference endpoint is included.
- RGB denoising experiments remain separate from Bayer RAW ISP evidence.

## Implemented in this release
- Local FastAPI backend with a browser Processing Studio.
- DNG Bayer input via rawpy, and NPY/NPZ Bayer mosaics.
- Bayer pattern override (RGGB/BGGR/GRBG/GBRG).
- Existing traditional, learned and hybrid paths connected to the UI.
- Per-input condition analysis display.
- Optional reference image and PSNR/SSIM/DeltaE2000/LPIPS metrics.
- Output preview and PNG download.
- Heuristic adaptive fusion: learned-dominant in normal/noisy/highlight conditions; more classical contribution for detected underexposure.
- Hybrid policy unit tests and app smoke test using a synthetic NPY Bayer input.

## Verification performed in the build environment
- 29 automated tests passed.
- FastAPI application import succeeded.
- `/api/health` and `/` returned HTTP 200.
- `/api/process` completed an end-to-end synthetic NPY Bayer request for Traditional, Learned and Hybrid and returned outputs plus reference metrics.
- The uploaded v4 checkpoint is present in the package and was not modified.

These tests verify software integration and contracts. The synthetic test image is not a quality benchmark.

## Proposed, not yet proven
- Adaptive fusion weights are heuristic and informed by a two-capture smoke test. The full 24-capture robustness split must be rerun before claiming improvement.
- Exposure recovery is limited by the information in the input. Severe clipping cannot be reconstructed reliably from a single RAW frame.
- The condition analyser estimates noise from image statistics; its estimate is not calibrated for every sensor.
- DNG handling uses a simplified scalar black-level subtraction and the embedded CFA pattern. Per-channel black-level maps, camera-specific colour matrices, lens shading, dual-gain RAW, and all vendor metadata are not yet implemented.
- White-balance and CCM calibration need colour-chart or suitable reference data.
- No FP16/INT8 quality-speed comparison has been completed in this release.
- Android, WebGPU, CUDA deployment profiles and public hosting have not been validated by this package.
- Browser UI is a local front end to the Python backend; it is not browser-local WebGPU inference.

## Run
Install:
```powershell
python -m pip install -r requirements-app.txt
```
Start:
```powershell
.\run_studio.ps1
```
Open:
`http://127.0.0.1:8000`

See `docs/PROCESSING_STUDIO.md` for input formats, reference-image rules and limitations.

## PNG/JPEG support added
- `.png`, `.jpg`, `.jpeg` uploads are now accepted.
- UI switches to RGB Original, RGB Denoise and RGB Detail-Preserving Denoise.
- RGB images use OpenCV NLM and optional restrained sharpening only.
- They are not sent through the Bayer U-Net, Bayer demosaicer, or RAW white-balance/CCM stages.
- PNG and JPEG upload endpoint tests pass.

## Full RGB enhancement mode added
- RGB Full Enhancement is now available for PNG/JPG/JPEG.
- It applies conservative white balance, adaptive exposure lift, LAB luminance CLAHE, moderate NLM denoising, restrained saturation and sharpening.
- It remains a separate rendered-RGB pipeline and does not call the Bayer RAW U-Net.
- Automated suite remains green (32 tests); visual quality must be evaluated on representative photos and clean references.
