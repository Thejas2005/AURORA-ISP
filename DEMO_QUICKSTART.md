# AURORA demo quickstart

## Start
Open PowerShell in this folder, activate your Python environment, then run:

```powershell
python -m pip install -r requirements-app.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Keep the terminal open while using the app.

Alternatively, activate the environment and double-click `run_aurora.bat`.

If PowerShell blocks `.ps1` files, use the direct `python -m uvicorn` command above. No execution-policy change is needed.

## Demonstration sequence
1. Upload an ordinary `.jpg` or `.png`; choose **Full RGB Enhancement**.
2. Upload a noisy photo; compare **RGB Original**, **RGB Denoise**, and **Full RGB Enhancement**.
3. Upload a Bayer DNG or 2-D Bayer NPY/NPZ; select Traditional, Learned, and Hybrid.
4. Supply a clean, exactly aligned RGB reference only when computing PSNR/SSIM/Delta E/LPIPS.
5. Download the result and check its dimensions.

## Honest model routing
- Bayer DNG/NPY/NPZ: Traditional / Learned AURORA U-Net / Hybrid Bayer RAW pipeline.
- PNG/JPG/JPEG: separate rendered-RGB enhancement path (classical stages). The Bayer checkpoint is not applied to rendered RGB.
- Do not claim RGB neural inference, Android inference, mobile latency or 30 dB unless separately implemented and measured.

## Known limits
- RGB full enhancement is heuristic and should be visually inspected for colour casts, halos and over-enhancement.
- Severe highlight clipping cannot be reconstructed from a single rendered image.
- DNG metadata and camera calibration support are not universal.
