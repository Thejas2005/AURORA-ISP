# AURORA ISP — v4.0.2

Adaptive Unified RAW Optimization and Reconstruction Architecture. Computer Vision Hackathon 2026, P2: Intelligent and Computationally Efficient Smartphone ISP.

## Current status
v3 fixes core interface and model-adapter correctness. It is **not yet a trained or benchmarked ISP**. There are no real-data quality claims. See `docs/IMPROVEMENTS_v3.md` and `docs/DATASET_CONTRACT.md`.

## Environment
Use a project virtual environment. Install core dependencies with `python -m pip install -r requirements-core.txt`. Install PyTorch separately using the official selector for the machine's CUDA/CPU configuration. Do not blindly install a CPU or mismatched CUDA wheel.

## Public API
```python
from core.pipeline import traditional_isp, learned_isp, hybrid_isp
from models.learned_isp.unet import AuroraUNet
from models.learned_isp.adapter import LearnedISP

meta = {"pattern":"RGGB", "black_level":0, "white_level":1,
        "iso":800, "exposure_ms":8.0, "analog_gain":2.0}
traditional = traditional_isp(bayer_mosaic, meta)  # sRGB HxWx3 float32 [0,1]
model = AuroraUNet(base=24, meta_dim=5)
# Load a trained checkpoint before using the learned path.
adapter = LearnedISP(model)
learned = learned_isp(bayer_mosaic, meta, adapter)
hybrid, conditions, decisions = hybrid_isp(bayer_mosaic, meta, adapter)
```

Input must be a finite, even-sized, single-channel Bayer mosaic (minimum 4x4), with an explicit Bayer pattern and correct black/white levels. Learned mode requires trained weights. The v4 training U-Net predicts camera-rendered sRGB directly. Do not pass this prediction through linear-RGB tone mapping or gamma encoding again.

## Tests and synthetic smoke benchmark
```powershell
python -m pytest -q
python -m experiments.benchmarks.compare_modes --size 256
```
The benchmark uses synthetic random input and reports runtime/shape only. It is not an image-quality evaluation. Supply `--checkpoint` only after training a valid checkpoint.

## Training
Training is intentionally gated until the downloaded dataset's actual layout, Bayer representation, pairing, licence and colour-space semantics are verified. Follow `docs/DATASET_CONTRACT.md`; do not assume SIDD RAW-RGB is a Bayer mosaic.

## v3.1 correctness update
See `docs/IMPROVEMENTS_v3_1.md` for the pipeline corrections, metadata-conditioning policy, validation changes and remaining limitations. Training remains intentionally gated pending SIDD-CC format and pairing verification.

## AURORA ISP v4 — train the SIDD-CC RAW-to-sRGB model

**v4.0.2 performance update:** decoded RAW/GT/metadata/sRGB data is cached one capture at a time. Capture-aware batching keeps patches from the same capture together, while shuffling capture order each epoch. This avoids repeatedly decoding large MAT/PNG files for every patch without loading the complete dataset into RAM. Training now prints batch progress and an approximate within-epoch ETA. The model, patch count, split policy and loss objective are unchanged.

The v4 training path uses paired SIDD-CC Bayer mosaics and camera-rendered sRGB targets. From the repository root, activate the project virtual environment and run:

```powershell
python -m experiments.training.train --data-root "C:\Users\theja\Desktop\CV_Hackaton\AURORA-ISP\datasets\SIDD-CC" --epochs 30 --patch-size 128 --batch-size 2 --patches-per-capture 8 --workers 0 --amp
```

Use the actual path where your existing SIDD-CC dataset resides. Do not copy or download the dataset again just to train v4. The best checkpoint is written to `models/checkpoints/aurora_v4_best.pt`; the latest checkpoint is `models/checkpoints/aurora_v4_best_latest.pt`. To resume, add `--resume "models/checkpoints/aurora_v4_best_latest.pt"`.

For a lower-memory start, use `--patch-size 128 --batch-size 1`. If GPU memory permits, test larger patches/batches experimentally. Training uses the GPU automatically when CUDA is available; `--amp` enables CUDA mixed precision. CPU fallback is supported but will be slower.

Validation metrics printed during training are patch-level monitoring values. They are not the final official test benchmark and must not be presented as final hackathon results. No trained weights or performance claims are bundled with this source release.


## Modular ISP v5 and RGB restoration experiment
See `docs/MODULAR_ISP_V5.md`. Optional dependencies: `requirements-restoration.txt`. Run `python -m experiments.benchmarks.restore_rgb_sweep` from the project root. This RGB benchmark is separate from the Bayer RAW learned ISP.


## AURORA Processing Studio
Install `requirements-app.txt`, then run `run_studio.ps1` on Windows and open http://127.0.0.1:8000. See `docs/PROCESSING_STUDIO.md` for supported Bayer RAW formats and metric rules.
