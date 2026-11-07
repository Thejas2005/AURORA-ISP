# AURORA ISP architecture (diagram for submission)
```mermaid
flowchart LR
  IN[Input + metadata] --> PRE[Black level / normalise]
  PRE --> DEM[Demosaic] --> WB[WB + CCM]
  WB --> AN[Condition analysis<br/>noise, luma, clipping, texture, cast]
  AN --> CTRL{Adaptive controller}
  CTRL -- clean --> FIN
  CTRL -- noisy --> RES[Classical denoise OR Learned U-Net]
  RES --> FIN[Tone map, gamma, sharpen, sRGB]
  FIN --> OUT[RGB export]
  OUT --> EVAL[PSNR / SSIM / DeltaE / LPIPS / latency / memory]
```
Modes: **Traditional** = fixed classical path (`core/pipeline.py:traditional_isp`); **Learned** = `models/learned_isp`;
**Hybrid** = classical front end + controller + learned restore (`hybrid_isp`). Status of each: see README.
