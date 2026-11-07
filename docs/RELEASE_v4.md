# AURORA ISP v4 release notes

**Release focus:** trainable Bayer RAW-to-camera-rendered-sRGB baseline.

### Implemented
- SIDD-CC loader with capture-stem pairing and strict dimension checks.
- Bayer CFA parsing and canonical 4-plane packing.
- Instance-directory train/validation split before patch sampling.
- Patch-based metadata-conditioned U-Net training.
- Validation crop determinism, AMP option, gradient clipping, cosine LR schedule, atomic best/latest checkpoint saves and resume support.
- Learned output contract is camera-rendered sRGB; no second gamma/tone-map pass.
- Dataset contract and training instructions.

### Evidence status
- Inherited regression suite: 24 tests pass in the build environment.
- Training has not been run against the user's local 32.6 GB dataset from this build environment; the dataset is not included in the release archive.
- No trained checkpoint is included. No PSNR/SSIM/Delta E/LPIPS improvement or runtime advantage is claimed.
- Final reporting must use the official held-out SIDD-CC test blocks and same-input comparisons across traditional, learned and hybrid modes.

### Hardware note
The development GPU is an RTX 4060 Laptop GPU with 8 GB VRAM. Start with patch size 128 and batch size 1–2, then adjust based on actual peak memory. This is a conservative starting configuration, not a measured optimum.
