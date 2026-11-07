# AURORA ISP — comparison and corrective changes (v2)

## Retained from the supplied Claude repository
- Project structure, classical ISP, rule-based hybrid controller, FiLM metadata-conditioned U-Net,
  metrics, synthetic benchmark, frontend, docs and placeholder runtime directories.

## Fixed
- Bayer patterns now use explicit pattern-aware sampling; no cropping/shifted conversion.
- Bilinear demosaicing uses sample-count normalisation at image borders.
- Normalisation rejects invalid black/white levels instead of silently hiding them.
- Classical and hybrid paths preserve spatial dimensions for all four Bayer patterns.
- FiLM model requires correctly shaped finite metadata when conditioning is enabled.
- Hybrid RGB restoration callback explicitly rejects incompatible output contracts.
- New tests cover all four patterns, flat-field reconstruction and model backward pass.

## Remaining critical gaps
- No verified SIDD-CC paired loader or metadata normalisation specification.
- Learned path is not connected to the three-way comparison or hybrid pipeline.
- Hybrid currently falls back to classical denoising if no RGB-domain restorer is supplied.
- No training results or trained checkpoint; no real evaluation results.
- Browser implementation has not been checked for parity with the Python pipeline.
- ONNX export and quantisation have not been validated.
- Current synthetic benchmark is a wiring smoke test, NOT a quality benchmark.

## Next engineering step
Validate SIDD-CC file structure and sensor metadata, implement a dataset contract,
then build a unified inference adapter for traditional / learned / hybrid methods.
