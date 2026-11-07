# AURORA ISP v4 — training release

## Implemented in this release

- SIDD-CC paired Bayer RAW to camera-rendered sRGB patch loader.
- Exact capture-index pairing for noisy RAW, clean RAW, metadata and clean sRGB.
- Deterministic instance-level train/validation partition before patch extraction.
- Four-pattern Bayer canonical packing; no crop-based CFA conversion.
- Metadata-conditioned U-Net predicts sRGB directly.
- AMP option, gradient clipping, cosine learning-rate schedule, best/latest atomic checkpoints and optional checkpoint resume.
- Validation patch crops are deterministic. Training crops are sampled dynamically.

## Not yet established

- No trained checkpoint is included.
- No image-quality improvement is claimed.
- No official test-set metrics, ablations, quantisation results or deployment latency are claimed.
- The model is a baseline candidate; architecture changes must be driven by measured results.

## Why sRGB is the target

SIDD-CC provides paired clean sRGB renderings corresponding to its Bayer RAW captures. The model therefore learns the camera-rendered RAW-to-sRGB mapping. Its output must not be treated as linear RGB or passed through a second gamma/tone-mapping stage.


## v4.0.1 hotfix — patch-boundary correction

The first training attempt exposed a crop-boundary bug in the SIDD-CC loader.
The crop origin was bounded using half the requested full-resolution patch size,
which allowed a crop near an image edge to be truncated. The loader now bounds
origins using `image_dimension - patch_size` and samples only even coordinates,
so packed Bayer input and full-resolution sRGB target remain spatially aligned.
A synthetic regression test checks exact tensor shapes. This is a loader correctness
fix; it is not evidence of model quality or training performance.
