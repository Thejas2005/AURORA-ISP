# AURORA ISP v3.1 — Pipeline Correctness Update

## Implemented in this update
- Separated learned linear-RGB reconstruction from public sRGB photofinishing. Hybrid learned restoration now receives one shared finishing pass; it no longer tone-maps/sharpens, decodes sRGB, and finishes a second time.
- Added an explicit minimum Bayer mosaic size of 8x8 for the learned path (packed dimensions >=4x4), matching the two pooling stages in the U-Net.
- Added finite/positive metadata validation and a shared helper that fills missing noise/luminance conditioning from image analysis while preserving supplied values and recording provenance.
- Connected image-condition estimates to learned-mode and hybrid-mode model conditioning.
- Made hybrid white-balance strength operational and incorporated black/clipping fractions into the conservative skip-heavy decision.
- Tightened controller input range validation.
- Added regression tests for the hybrid finishing path, minimum dimensions, metadata estimation and invalid metadata.

## Still unverified
- No trained checkpoint or real SIDD training/evaluation is included.
- Controller thresholds and white-balance strength remain provisional heuristics and require validation on real RAW scenes.
- Learned mode assumes the model's target is linear RGB. The dataset's ground-truth transfer function and camera colour pipeline must be verified before training.
- The learned branch intentionally predicts colour from RAW itself; classical WB/CCM is not applied to that branch. Its paired training targets must therefore define the desired colour rendering.
- This update does not claim image-quality gains, real-device latency, quantisation performance, WebGPU or Android deployment.
