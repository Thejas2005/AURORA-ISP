# AURORA ISP v4 — SIDD-CC data contract

## Verified dataset representation

- Root contains `Raw/train/Data/<instance>/` and `SRGB/train/Data/<instance>/`.
- RAW MAT files contain `data`: single-channel, full-resolution Bayer mosaic, float32.
- RAW values are already black-level-subtracted/normalised by SIDD-CC. Preserve negative values; do not subtract black level a second time or clip before the network.
- Metadata MAT contains `cfa_str`; the loader refuses unknown patterns. CFA is canonicalised by `pack_bayer` into `[R, Gr, Gb, B]` planes.
- Matching clean sRGB PNG is the supervision target. It is camera-rendered sRGB, not linear RGB or an independent ideal-colour reference.
- Pairing uses the exact capture prefix and capture index. Missing pairs or dimension mismatch raise errors.
- The training/validation split is deterministic and by instance directory before patch sampling. The held-out official test benchmark is not used for training or validation.

## Model tensor contract

- Input: `float32 [4, patch_size/2, patch_size/2]` canonical packed Bayer.
- Metadata: `float32 [5]` from `core.metadata.metadata_vector`: ISO, exposure_ms, analog_gain, noise_sigma, mean_luma transformations.
- Target/output: `float32 [3, patch_size, patch_size]` camera-rendered sRGB in `[0,1]`.
- Patch origins are aligned to even sensor coordinates so the Bayer packing phase is preserved.
- Noise sigma is estimated from paired noisy-clean RAW patch residuals; mean luma is estimated from the aligned target patch. Exposure and analog gain remain neutral defaults until their units/values are reliably decoded.

## Training/evaluation caveats

- The clean sRGB target reproduces the dataset camera rendering. It should not be described as a scene-linear or universally colourimetric ground truth.
- Validation patch PSNR/L1 during training are monitoring metrics only. Final claims require the fixed official test blocks and the same test protocol for all ISP modes.
- Dataset licensing/terms must be respected. Do not redistribute SIDD-CC files.
