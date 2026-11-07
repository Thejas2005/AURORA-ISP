# Known limitations / unknowns
- Python Traditional + Hybrid paths run; Hybrid currently falls back to classical denoising (no trained model).
- `models/learned_isp`, training and quantisation scripts are written but **never executed**.
- `datasets/loader.py` does not exist yet; real-data evaluation is not possible yet.
- `to_rggb` only handles RGGB correctly; GRBG/GBRG are crop-shifted, BGGR needs a channel swap. Verify per dataset.
- Browser build takes ordinary images (not Bayer/DNG), runs JS on CPU; WebGPU not used; no learned path.
- Synthetic smoke-test scores are NOT results (reference is an unprocessed scene; ISPs apply WB/tone).
- Dataset licence and Samsung repo (non-commercial terms) must be reviewed before reuse.


## Processing Studio v6 additions
- The local Studio currently accepts Bayer DNG (rawpy), NPY and NPZ mosaics only. It does not accept arbitrary RGB JPEG/PNG as learned Bayer input.
- DNG black-level handling is currently scalar median subtraction; per-channel black-level maps and camera-specific colour calibration remain future work.
- Hybrid fusion weights are heuristic and must be revalidated on the complete held-out robustness split.
- The Studio connects to a local FastAPI Python backend; it is not a browser-local WebGPU model and no public hosted URL is provided in this build.
