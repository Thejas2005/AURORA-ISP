# AURORA ISP v3 — correctness and integration release

## Fixed in this release
- One public input contract: even-sized 2-D Bayer mosaic + metadata; all public modes return sRGB float [0,1].
- Added a real Bayer packing/model adapter. The four packed channels preserve R, Gr, Gb, B semantics for all four Bayer patterns.
- Added a single documented five-value metadata transform shared by inference and the future dataset loader.
- Added explicit learned mode and corrected hybrid integration: a raw-domain adapter is called with Bayer mosaic + metadata; a bare U-Net is never passed as an RGB callback.
- Hardened controller input validation and small-image noise estimation.
- Hardened metric shape/range/finite validation; removed mutable-default LPIPS cache.
- Removed the misleading incomplete training loop. Training now fails with an actionable dataset audit gate instead of importing a nonexistent loader and pretending to train.
- Split core and training dependency guidance; PyTorch install is selected for the user's CUDA/CPU environment.
- Added a same-input three-mode smoke benchmark; learned mode is clearly marked not run without trained weights.

## Not claimed / still pending
- No trained checkpoint, dataset loader, measured quality improvement, real-data benchmark, ablation, or failure-analysis results are included.
- Controller thresholds are provisional heuristics.
- The dataset's MAT tensor semantics and target colour space must be verified before a loader is written.
- Browser WebGPU/WASM deployment and Android/native deployment remain unimplemented; the existing frontend is a local prototype, not proof of those targets.
- INT8/FP16 optimisation is not claimed as complete. Establish a trained FP32 baseline first, then measure quality, latency and memory after optimisation.

## Required next integration sequence
1. Audit downloaded SIDD files and licence; document pair IDs, scene IDs, Bayer status, value ranges and colour space.
2. Implement scene-safe paired dataset loader and train/validation loop; patch extraction only after split.
3. Train checkpoint; validate output and colour-space contracts.
4. Run three modes on identical real test pairs and references; report PSNR, SSIM, Delta E 2000, LPIPS (if available), params, model size, latency and peak memory.
5. Run ablations and failure analysis; then test FP16/INT8 and browser/native deployment separately.
