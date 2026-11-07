# AURORA ISP v4.0.2

## Why this update exists
The first v4 training implementation decoded the same full-resolution capture files once per patch. With eight patches per capture, that repeated large MAT and PNG reads and made the first epoch unnecessarily slow. It also printed no batch-level progress, making a long epoch look stalled.

## Changes
- Cache the current capture's noisy RAW, clean RAW, sRGB target and metadata in the dataset object.
- Use a capture-aware batch sampler: capture order is shuffled each epoch, but all patches for a capture are processed together so the one-capture cache is reused.
- Keep memory bounded to one decoded capture per dataset/worker rather than caching the entire dataset.
- Add batch progress, loss, rolling step-time and approximate remaining-epoch time output.
- Add validation progress output.
- Enable cuDNN benchmarking and high float32 matmul precision on CUDA.
- Preserve the existing split, patch count, architecture, target and loss.

## Validation
`python -m pytest -q`: 25 passed. Python compilation passed. This validates code/regression tests, not training speed or image quality on the user's machine.

## Expected operational effect
Each capture is decoded once per epoch instead of once per patch (for the current eight patches per capture). The actual speedup depends on SSD/HDD speed, CPU, RAM and GPU throughput; no fixed speedup or ETA is claimed.
