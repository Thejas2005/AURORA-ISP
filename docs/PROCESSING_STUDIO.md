# AURORA Processing Studio

## Start locally on Windows

1. Activate the existing project environment.
2. Install app dependencies if not already installed:
   `python -m pip install -r requirements-app.txt`
3. Run `.\run_studio.ps1` or `run_studio.bat`.
4. Open `http://127.0.0.1:8000`.

The UI calls the local FastAPI backend. Image inference stays on the development PC; the browser does not upload images to a remote service.

## Supported input in this milestone

- DNG Bayer mosaics decoded with rawpy, using the embedded CFA pattern, ISO and shutter metadata where present.
- NPY 2-D Bayer arrays, assumed normalised [0,1] and RGGB unless overridden.
- PNG, JPG and JPEG display-RGB images through a separate RGB restoration path (NLM denoising and detail-preserving NLM).
- NPZ with a 2-D `raw` array; optional `pattern`, `iso`, `exposure_ms`, `black_level`, `white_level` scalar arrays.

PNG/JPG/JPEG are not passed to the Bayer U-Net. They are already-rendered RGB and use RGB denoising only; RAW demosaicing, sensor white balance and sensor CCM are not applied. Not supported in this milestone: proprietary RAW formats not decoded by rawpy, multi-frame HDR, dual-pixel RAW, non-Bayer CFA layouts, and all vendor-specific metadata. RGB denoising is a separate experiment, not mislabeled as Bayer RAW ISP.

## Reference metrics

Optional clean reference must be full-resolution RGB with exactly the same HxW as the Bayer mosaic. The backend does not resize or register images. PSNR/SSIM/DeltaE2000/LPIPS are computed only when supplied. LPIPS is `null` if the optional dependency/model is unavailable.

## Modes

- Traditional: classical Bayer front end, WB/CCM, NLM and photofinishing.
- Learned: existing four-channel packed Bayer U-Net -> camera-rendered sRGB. No second photofinishing pass.
- Hybrid: both paths plus condition-dependent output fusion. Learned output is the dominant contribution in normal/noisy/highlight conditions; a larger classical contribution is used for detected underexposure.

Fusion weights are heuristic proposals informed by the two-capture smoke test. They are not claimed as optimal and must be validated on the complete held-out split before being described as a measured improvement.

## Model and privacy

The app loads the existing checkpoint lazily. It does not train, modify or overwrite model weights. Input data is held in memory for the request and processed by the local Python process.


## PNG/JPG/JPEG RGB workflow

When a `.png`, `.jpg` or `.jpeg` file is selected, the UI switches to RGB modes:
- RGB Original: pass-through baseline.
- RGB Denoise: stronger OpenCV coloured non-local means denoising (h≈30 at default strength), intended for visibly noisy rendered images.
- RGB Detail-Preserving Denoise: medium-strength NLM plus very restrained unsharp masking to avoid re-amplifying residual noise.

This branch does not invoke the Bayer RAW model. It is intended for ordinary photos and noisy rendered images, not as evidence of RAW-to-RGB reconstruction. Supply a separate clean reference with exactly matching dimensions to calculate image metrics.


## Full rendered-RGB enhancement

The RGB workflow includes `RGB Full Enhancement`:
1. Conservative gray-world white-balance adjustment with capped channel gains.
2. Adaptive exposure lift using robust luminance percentiles.
3. LAB-luminance CLAHE local contrast enhancement.
4. Moderate chroma-aware non-local-means denoising.
5. Restrained saturation adjustment and sharpening.

This path is for already-rendered RGB photographs. It does not reconstruct sensor RAW, and it does not invoke the Bayer U-Net. The processing stages are classical image enhancement and should be validated against matched references and visual inspection for each camera/image domain.


### Sequential RGB enhancement order
`RGB Full Enhancement` now explicitly calls the `RGB Denoise` stage first, then performs white-balance adjustment, adaptive exposure, LAB/CLAHE local contrast, restrained saturation and sharpening. It does not run a second NLM pass after contrast enhancement.
