# AURORA Modular ISP v5 — implementation notes

## Scope
This update adds inspectable ISP stages and a reproducible RGB restoration sweep inspired by modular ISP design patterns. It does not import Samsung Modular Neural ISP source code or weights.

## Preserved model contract
- Existing `AuroraUNet` architecture and checkpoint files are unchanged.
- Input remains canonical packed Bayer RAW with four channels and the existing five-dimensional metadata contract.
- Output remains camera-rendered sRGB.
- No second gamma/tone-mapping pass is applied to learned output.
- No retraining or claim of improved learned-model quality is made by this refactor.

## Added modules
- `core/pipeline_stages.py`: explicit sensor front-end, colour, denoising and photofinishing stages with intermediate outputs.
- `core/pipeline.py::traditional_isp_with_intermediates`: opt-in diagnostic interface; existing public ISP functions remain available.
- `core/denoising/restoration.py`: optional BM3D RGB (RGB-channel and YCrCb modes), corrected float-domain BayesShrink, and indicative noise diagnostics.
- `experiments/benchmarks/restore_rgb_sweep.py`: BM3D sigma sweep, wavelet baseline, CSV, noise diagnostics and contact sheet.

## Run
From project root:
```powershell
python -m pip install -r requirements-restoration.txt
python -m experiments.benchmarks.restore_rgb_sweep
```
Optional:
```powershell
python -m experiments.benchmarks.restore_rgb_sweep --sigmas 0.04,0.06,0.08,0.10,0.12,0.15,0.18,0.22
```

Outputs:
`outputs/aurora_modular_rgb/`

## Important evaluation limitation
The sweep computes reference metrics for each sigma. Choosing the best sigma using the same reference is reference-assisted parameter selection and must not be presented as blind inference performance. For an unbiased claim, tune on a separate calibration set and report on held-out images.

The input's noise model is unknown and the clean reference is JPEG. The target 30 dB is aspirational, not guaranteed. Clipping, compression, misalignment or information loss can make 30 dB unattainable. Do not fabricate scores or silently align/resize images.

## Samsung-inspired architectural ideas
Adopted as design patterns: modular stages, inspectable intermediate outputs, configurable processing, and separate evaluation of modules. No Samsung source code or weights are included. Review upstream licence terms before any direct reuse:
https://github.com/SamsungLabs/modular_neural_isp
