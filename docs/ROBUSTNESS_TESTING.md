# AURORA RAW robustness benchmark

This benchmark evaluates the existing checkpoint without retraining or modifying weights.

## Conditions
- Baseline
- Additional RAW-domain Gaussian noise at 1.25x, 1.75x and 2.5x estimated residual standard deviation
- Underexposure at 1/2x and 1/4x
- Overexposure at 1.5x and 2x, with clipping fraction recorded
- Warm/cool CFA channel-gain perturbations
- Sparse hot pixels

The clean RAW and camera-rendered sRGB references are held fixed. These are controlled stress tests, not exact physical camera simulations. White-balance casts are deliberately not given a correction gain to the learned model because its current five-value metadata contract has no WB-gain field.

## Run a smoke test first
```powershell
python -m experiments.benchmarks.evaluate_robustness `
  --data-root "C:\PATH\TO\SIDD-CC" `
  --checkpoint models/checkpoints/aurora_v4_best.pt `
  --crop 1024 --max-captures 2 --warmup 1 --runs 2 `
  --save-images --out outputs/raw_robustness_smoke
```

## Run the complete validation split
```powershell
python -m experiments.benchmarks.evaluate_robustness `
  --data-root "C:\PATH\TO\SIDD-CC" `
  --checkpoint models/checkpoints/aurora_v4_best.pt `
  --crop 1024 --max-captures 0 --warmup 1 --runs 3 `
  --save-images --out outputs/raw_robustness_full
```
Use `--max-captures 0` to evaluate the entire validation split.

## Outputs
- `robustness_summary.json`: scenario/mode aggregate metrics, scenario parameters, checkpoint/device metadata and CUDA peak allocated memory
- `per_capture_scenario.csv`: capture-level PSNR, SSIM, DeltaE2000, LPIPS and synchronized median latency
- `images/<scenario>/<capture>/<mode>.png`: optional RGB output images

## Important limitations
- SIDD-CC data loader parses the CFA pattern from metadata. The test uses the stored capture pattern.
- The current trained model does not accept explicit WB gains, so colour-cast stress tests assess its generalisation rather than an explicitly conditioned correction.
- Exposure tests change the noisy RAW input and exposure metadata but retain the clean sRGB target. Severe clipping is inherently information-destructive.
- Synthetic perturbations are controlled stress tests and cannot establish robustness to every real sensor, ISP, or illumination condition.
- Do not use these validation scores as official test-set results.
