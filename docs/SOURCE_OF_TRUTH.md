# AURORA ISP --- Project Source of Truth

**Version:** 1.0\
**Last updated:** 6 November 2026\
**Purpose:** This document records decisions already made for the
Computer Vision Hackathon 2026, Problem Statement 2. Use it as the
continuity reference if a future assistant response contradicts the
agreed scope.

------------------------------------------------------------------------

## 1. Project identity

-   **Project name:** AURORA ISP
-   **Expansion:** Adaptive Unified RAW Optimization and Reconstruction
    Architecture
-   **Hackathon:** Computer Vision Hackathon 2026
-   **Problem statement:** **P2 --- Intelligent and Computationally
    Efficient Smartphone ISP**
-   **Primary objective:** Build a learned, computationally efficient
    smartphone image-signal-processing pipeline that converts degraded
    sensor/RAW input into high-quality RGB output while balancing colour
    accuracy, noise reduction, detail preservation, robustness and
    device-resource efficiency.
-   This is an **ISP project**, not a generic photo-enhancement app and
    not a low-light-only restoration project.

## 2. Hackathon requirements to preserve

The P2 solution must cover the following areas.

### Classical computer-vision / ISP stages

-   Bayer RAW preprocessing and demosaicing when Bayer RAW is supplied.
-   White-balance correction.
-   Colour-correction matrix (CCM).
-   Denoising and edge-preserving filtering.
-   Gamma correction and tone mapping.
-   Sharpening and colour-space transformations.

### Deep-learning stages

-   Implement or adapt a learned ISP, such as a suitable U-Net-based
    RAW-to-RGB model or another justified architecture.
-   Adapt/fine-tune using an appropriate dataset.
-   Learn image restoration and colour transformation.
-   Handle noise and illumination variation robustly.

### Hybrid ISP

-   Integrate classical ISP operations with learned components.
-   Use a reasoned processing sequence.
-   Avoid colour distortion and fine-detail loss.
-   Compare **traditional ISP**, **learned ISP**, and **hybrid ISP** on
    the same test inputs.

### Advanced requirements

-   Adapt processing to sensor noise and illumination.
-   Demonstrate RAW-to-RGB reconstruction wherever Bayer RAW is
    available.
-   Apply at least one optimisation method: quantisation, pruning,
    knowledge distillation or reduced precision.
-   Measure model size, memory use and inference latency.
-   Show behaviour under challenging conditions, including degradation
    and recovery.
-   Compare against a conventional ISP baseline.

### Required evaluation measures

-   PSNR
-   SSIM
-   Colour accuracy (Delta E)
-   LPIPS or a justified perceptual metric
-   Model parameter count and model size
-   Inference latency
-   Memory consumption

**Important:** Do not claim a metric improved until it has actually been
measured on a defined test set.

## 3. Agreed architecture direction

AURORA is a multi-stage, adaptive **hybrid ISP**.

1.  **Input and sensor analysis**
    -   Accept DNG/Bayer RAW where supported, and supported RAW-RGB or
        degraded sensor-image inputs.
    -   Read available metadata such as ISO, exposure, white balance and
        Bayer pattern.
    -   Preserve original sensor values and relevant metadata.
2.  **Classical RAW preprocessing**
    -   Black-level subtraction and sensor-range normalisation.
    -   Bayer-pattern identification and demosaicing for Bayer mosaics.
    -   White balance and colour correction.
    -   Appropriate initial denoising.
3.  **Image-condition analysis**
    -   Estimate noise level/variance.
    -   Analyse luminance distribution and underexposure.
    -   Detect saturation and near-black clipping.
    -   Measure local texture/variance and edge density.
    -   Estimate colour cast / white-balance deviation.
4.  **Learned ISP**
    -   Initial candidate: metadata-conditioned U-Net-based RAW-to-RGB
        model.
    -   For Bayer mosaics, investigate a four-channel packed Bayer
        representation.
    -   Use multi-scale features and residual connections where
        justified.
    -   NAFNet may be evaluated as a denoising component or baseline; it
        is **not by itself the complete ISP**.
    -   Final architecture choice depends on the actual training pairs,
        RAW representation and supervision available.
5.  **Adaptive hybrid controller**
    -   Begin with a transparent, measurable rule-based controller.
    -   Example policy: stronger denoising for high-noise inputs;
        preserve edges in high-texture regions; protect highlights when
        clipping is detected; apply colour correction according to
        estimated cast; skip unnecessary heavy operations on
        already-clean inputs.
    -   A learned controller is optional and should only be added if
        experiments justify it.
6.  **Refinement and output**
    -   Colour correction, gamma, tone mapping and controlled
        sharpening.
    -   Convert to the declared output colour space and export RGB.
7.  **Evaluation**
    -   Compare traditional, learned and hybrid paths.
    -   Show intermediate outputs and benchmark results.

## 4. Required comparison design

Implement three independently identifiable modes:

-   **Traditional ISP:** classical processing only.
-   **Learned ISP:** neural RAW-to-RGB path.
-   **Hybrid ISP:** the proposed combination of classical and learned
    stages with adaptive control.

Run all three on the same eligible inputs and evaluate with the same
protocol. Do not make unfair comparisons by changing inputs, target
references or output settings between modes.

## 5. Dataset and model research already considered

### SIDD

-   Official project site: https://abdokamel.github.io/sidd/
-   SIDD Small was identified as a practical initial dataset to inspect.
-   SIDD includes paired noisy/ground-truth material and metadata;
    available representations include RAW-RGB and sRGB.
-   **RAW-RGB is not the same thing as a Bayer mosaic.** Do not describe
    RAW-RGB data as Bayer RAW.
-   Verify the exact files, data representation, metadata and licence
    before designing the training loader.
-   If Bayer RAW/DNG supervision is required, establish whether usable
    Bayer mosaics and suitable targets are available; do not assume the
    RAW-RGB pairs alone satisfy that requirement.

### Samsung Modular Neural ISP

-   Repository considered:
    https://github.com/SamsungLabs/modular_neural_isp
-   Relevant components include denoising, photofinishing, enhancement,
    AWB/CCM and evaluation.
-   NAFNet variants are relevant to denoising; photofinishing includes
    operations such as tone mapping and gamma correction.
-   SIDD metadata support was identified in parts of the repository, but
    SIDD MAT data may need conversion to the loader's expected format.
-   Review repository and individual-file licences before copying code,
    weights or assets. The repository includes non-commercial licence
    terms; do not assume unrestricted reuse.
-   Treat this repository as a technical reference or candidate
    component, not as proof that AURORA is already implemented.

## 6. Development hardware and target platforms

### Confirmed

-   Development OS: Windows.
-   GPU: NVIDIA RTX 4060, reported as 12 GB VRAM.
-   More than 100 GB free disk space was reported.
-   Target environments: Windows desktop and Android mobile.
-   A publicly accessible hosted URL is required by the hackathon.
-   The user wants processing to use the device's own resources and
    prefers local processing rather than uploading images to a cloud
    inference service.

### Deployment direction

Use a dual-mode strategy:

-   **Hosted public application:** a live URL that judges can open
    immediately on a phone or laptop.
-   **Browser-local mode:** process images locally in the browser where
    supported, using WebGPU/WASM with a CPU fallback.
-   **Native Windows mode:** use native acceleration such as CUDA/ONNX
    Runtime where supported, with CPU fallback.
-   **Native Android mode:** start with a reliable CPU path; add GPU/NPU
    backends only when supported by the actual test device and
    validated.

Keep the same user workflow and project identity across modes, but do
not force one binary or runtime onto every platform.

**Critical distinction:** Browser WebGPU is not equivalent to native
CUDA or a phone's native GPU/NPU runtime. Report benchmarks separately
by runtime and device. Do not claim NPU acceleration unless it is
actually running on an NPU and measured.

The hosted URL must be a functional application, not merely a landing
page. Browser-local processing is the intended route for an immediately
usable hosted demo without requiring image uploads.

## 7. Training and optimisation direction

-   Use PyTorch for model development/training.
-   Start with a manageable U-Net-based RAW-to-RGB baseline, subject to
    data compatibility.
-   Use patch-based training and mixed precision where supported.
-   Use scene/capture-aware train/validation/test splits; do not leak
    neighbouring patches from the same scene across splits.
-   Evaluate generalisation on unseen scenes.
-   Compare the original model against an optimised version, such as
    FP16 and/or INT8 quantisation.
-   Validate image quality after optimisation; lower latency alone is
    not sufficient.
-   Tune batch size, patch size, loss functions and model size
    experimentally rather than treating proposed values as established
    results.

## 8. Evaluation and experiment plan

### Core experiments

A. **Image quality:** traditional vs learned vs hybrid on identical test
inputs.\
B. **Robustness:** different noise levels/ISO, illumination, exposure
and unseen inputs.\
C. **Efficiency:** original vs optimised model, with runtime-specific
latency and memory.

### Reporting rules

-   Report PSNR, SSIM, Delta E and LPIPS where valid reference data and
    colour-space assumptions permit.
-   State the target reference and colour space for every metric.
-   Report model-only latency separately from full-pipeline latency.
-   Distinguish cold-start/model-load time from warm inference time.
-   Report memory use and model size.
-   For Android, note device model and relevant
    thermal/sustained-performance conditions.
-   Do not invent measurements or imply that results from one device
    apply to all devices.

### Ablations to plan

-   Without adaptive controller.
-   Without classical denoising.
-   Without learned restoration.
-   Without metadata conditioning.
-   Without colour correction.
-   Original precision/model vs optimised precision/model.

### Failure analysis

Document real failure cases, their input conditions, visible artefacts,
likely cause and mitigation. Never hide a failure by excluding it
silently from the test set.

## 9. UI direction

Working interface concept: **AURORA Processing Studio**.

It should provide: - RAW/DNG or supported sensor-image input and visible
metadata. - Traditional / Learned / Hybrid processing modes. - Adaptive
image-condition analysis. - Side-by-side or three-way output
comparison. - Actual measured metrics: PSNR, SSIM, Delta E, LPIPS,
latency and memory where available. - Runtime/device indicator. -
Loading, error and unsupported-input states. - Output export.

Do not show fabricated benchmark values, placeholder scores presented as
real results, or unsupported hardware claims.

## 10. Repository structure direction

``` text
AURORA-ISP/
├── frontend/
│   └── src/
│       ├── studio/
│       ├── comparison/
│       ├── benchmarks/
│       └── components/
├── core/
│   ├── raw/
│   ├── preprocessing/
│   ├── demosaicing/
│   ├── colour/
│   ├── denoising/
│   ├── adaptive_controller/
│   ├── photofinishing/
│   └── evaluation/
├── models/
│   ├── classical_baselines/
│   ├── learned_isp/
│   ├── hybrid_isp/
│   ├── checkpoints/
│   ├── onnx/
│   └── quantized/
├── runtimes/
│   ├── windows_cuda/
│   ├── windows_cpu/
│   ├── android_cpu/
│   ├── android_gpu/
│   └── android_npu/
├── datasets/
│   ├── sidd/
│   ├── samsung_isp/
│   └── evaluation/
├── experiments/
│   ├── training/
│   ├── benchmarks/
│   ├── ablations/
│   └── failure_cases/
├── docs/
│   ├── architecture.md
│   ├── design_document.md
│   ├── evaluation.md
│   └── limitations.md
└── README.md
```

This is a proposed structure, not a claim that these files or folders
already exist.

## 11. Hackathon submission checklist

The submission instructions identified these deliverables: - Public live
application URL. - GitHub repository with complete source code. - System
architecture diagram. - Design document. - Comparative evaluation
results in `eval_probID_TeamID.md` (example: `eval_P2_T01`); first line
contains team members' USNs comma-separated, with metrics in the
prescribed order. - Ablation study. - Failure-case analysis.

Rubric values shown in the table: - Evaluation metrics: 25 -
Implementation: 20 - UI/UX and deployment: 15 - Project documentation:
15 - Demo and viva: 25 - Total: 100

The source document's introductory prose appears inconsistent with its
rubric table. Confirm the discrepancy with the organisers rather than
silently changing the table.

## 12. Explicitly out of scope for the initial build

Do not spend early development time on: - User
accounts/authentication. - Cloud GPU inference. - Social sharing or
photo galleries. - Generic filters unrelated to ISP requirements. -
Unrelated AI features. - Supporting every camera RAW format. - Separate
unrelated algorithms for every platform. - Claiming unmeasured NPU
acceleration.

## 13. Decisions still open --- do not invent answers

These were asked but not yet answered:

1.  **Android test hardware:** Snapdragon, MediaTek, Exynos, Google
    Tensor, multiple devices, or unknown?
2.  **Input scope:** Bayer RAW/DNG only; RAW-RGB and degraded sensor
    data; or both as distinct paths?
3.  **Team size:** 2--3, 4--5, or 6+?

Other decisions that require inspection or experiments: - Exact training
dataset and licence-compatible use. - Exact learned ISP architecture. -
Exact RAW formats and Bayer patterns supported in the first milestone. -
Exact Android acceleration backend based on the actual phone. - Model
size/latency targets based on measured baselines.

## 14. Working rules for future planning and implementation

1.  Treat this document and the original P2 statement as the project
    baseline.
2.  Do not silently change the problem statement, scope, architecture
    direction or deployment constraints.
3.  Clearly label every statement as one of:
    -   **Decided:** explicitly agreed above.
    -   **Proposed:** a recommendation not yet approved.
    -   **Unknown:** needs user input, source inspection or measurement.
    -   **Measured:** backed by reproducible experiment results.
4.  Do not invent datasets, files, model results, hardware capabilities,
    team decisions or completed implementation.
5.  When new evidence conflicts with this document, explain the conflict
    and ask before changing a decision.
6.  Keep the project focused on P2 smartphone ISP requirements.
7.  Prioritise a working, measurable end-to-end pipeline over adding
    features for appearance.
8.  If a future response contradicts this document, the user can say:
    **"Check the AURORA ISP Source of Truth and correct your plan."**

------------------------------------------------------------------------

**Status:** This document records the decisions and directions available
as of 6 November 2026. It is a continuity reference, not evidence that
implementation, training or benchmarking has already been completed.
