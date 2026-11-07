# Milestones (all items **Proposed** until you approve)
| # | Goal | Exit criterion |
|---|------|----------------|
| M0 | Repo, classical ISP, metrics, tests, browser studio | **Done in this scaffold** (synthetic smoke test only) |
| M1 | Resolve open decisions: Android device, input scope, team size | Recorded in SOURCE_OF_TRUTH |
| M2 | Inspect SIDD (Small) files, metadata, licence; decide Bayer availability; write `datasets/loader.py` with scene-level splits | Loader yields (packed RAW, meta, target) |
| M3 | Train U-Net baseline (PyTorch, RTX 4060); validate on held-out scenes | Checkpoint + val metrics |
| M4 | Real-data comparison Traditional vs Learned vs Hybrid; fill `eval_P2_TXX.md` | Measured table |
| M5 | Optimise (FP16/INT8), re-validate quality, benchmark per runtime | Size/latency/memory table |
| M6 | Port learned model to browser (onnxruntime-web, WASM/WebGPU) + host public URL | Live URL works on phone |
| M7 | Ablations + failure cases + design doc | Docs complete |
| M8 | Android CPU path; GPU/NPU only if validated on the actual phone | Device-labelled benchmarks |
Deadline dates are **Unknown** -- add them once you share the hackathon schedule.
