"""Three Bayer-mosaic ISP modes; public outputs are sRGB float images in [0,1]."""
import numpy as np
from core.preprocessing.raw_prep import normalise
from core.demosaicing.bilinear import demosaic_bilinear
from core.colour.colour import gray_world_gains, apply_wb, apply_ccm
from core.denoising.classical import denoise
from core.photofinishing.finish import finish
from core.adaptive_controller.controller import analyse, decide
from core.metadata import metadata_with_conditions

IDENTITY_CCM = np.eye(3, dtype=np.float32)


def _validate(mosaic, learned=False):
    x = np.asarray(mosaic)
    minimum = 8 if learned else 4
    if x.ndim != 2 or min(x.shape) < minimum or any(v % 2 for v in x.shape):
        raise ValueError(f"Input must be an even-sized 2-D Bayer mosaic (minimum {minimum}x{minimum})")
    if not np.isfinite(x).all():
        raise ValueError("RAW mosaic contains nonfinite values")
    return x


def _front_end(mosaic, meta):
    x = normalise(mosaic, meta.get("black_level", 0), meta.get("white_level", 1))
    return demosaic_bilinear(x, meta.get("pattern", "RGGB"))


def _gains(rgb, meta):
    gains = meta.get("wb_gains")
    return gray_world_gains(rgb) if gains is None else np.asarray(gains, dtype=np.float32)


def _apply_colour(rgb, meta, wb_strength=1.0):
    gains = _gains(rgb, meta)
    if gains.shape != (3,) or not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("wb_gains must contain three finite positive values")
    blended_gains = 1.0 + float(np.clip(wb_strength, 0, 1)) * (gains - 1.0)
    balanced = apply_wb(rgb, blended_gains)
    return apply_ccm(balanced, meta.get("ccm", IDENTITY_CCM))


def classical_stage(mosaic, meta):
    raw_rgb = _front_end(_validate(mosaic), meta)
    return _apply_colour(raw_rgb, meta, 1.0)


def _learned_srgb(mosaic, meta, learned_model, conditions=None):
    if learned_model is None:
        raise ValueError("Learned mode requires a trained model adapter/checkpoint")
    model_meta = metadata_with_conditions(meta, conditions) if conditions is not None else dict(meta)
    srgb = np.asarray(learned_model(_validate(mosaic, learned=True), model_meta), dtype=np.float32)
    if srgb.shape != (*np.asarray(mosaic).shape, 3) or not np.isfinite(srgb).all():
        raise ValueError("Learned adapter must return finite HxWx3 sRGB")
    return np.clip(srgb, 0, 1)


def traditional_isp(mosaic, meta):
    rgb = classical_stage(mosaic, meta)
    return finish(denoise(rgb, 0.3), tone=0.5, sharp=0.4)


def learned_isp(mosaic, meta, learned_model):
    """Learned Bayer-to-camera-rendered-sRGB path; no second photofinishing pass."""
    mosaic = _validate(mosaic, learned=True)
    conditions = analyse(_front_end(mosaic, meta))
    return _learned_srgb(mosaic, meta, learned_model, conditions)


def hybrid_isp(mosaic, meta, learned_model=None):
    """Adaptive hybrid. Classical and learned outputs are fused in sRGB space."""
    mosaic = _validate(mosaic, learned=learned_model is not None)
    raw_rgb = _front_end(mosaic, meta)
    conditions = analyse(raw_rgb)
    params = decide(conditions)
    classical = finish(denoise(_apply_colour(raw_rgb, meta, params.wb_strength), params.denoise),
                       tone=params.tone, sharp=params.sharpen)
    if learned_model is None:
        return classical, conditions, params
    learned = _learned_srgb(mosaic, meta, learned_model, conditions)
    # Evidence-informed adaptive fusion policy. The smoke test showed that
    # learned-only wins most conditions, while the classical branch helped
    # slightly at 2x underexposure. Keep the learned path dominant by default.
    # These weights are proposed controller settings, not yet validated on
    # the full 24-capture robustness split.
    if conditions.mean_luma < 0.18 or conditions.black_frac > 0.45:
        learned_weight = 0.65
        fusion_reason = "underexposure: retain more classical contribution"
    elif conditions.noise_sigma > 0.025:
        learned_weight = 0.90
        fusion_reason = "high estimated noise: learned-dominant fusion"
    elif conditions.clip_frac > 0.08:
        learned_weight = 0.90
        fusion_reason = "highlight clipping: avoid aggressive classical tone changes"
    else:
        learned_weight = 0.85
        fusion_reason = "normal conditions: learned-dominant fusion"
    fused = (1.0 - learned_weight) * classical + learned_weight * learned
    params.learned_weight = learned_weight
    params.fusion_reason = fusion_reason
    return np.clip(fused, 0, 1).astype(np.float32), conditions, params



def traditional_isp_with_intermediates(mosaic, meta, denoise_strength=0.3,
                                       tone=0.5, sharpen=0.25):
    """Inspectable classical Bayer path; returns named intermediate RGB/RAW stages."""
    from core.pipeline_stages import run_classical_stages
    from core.denoising.classical import denoise as nlm_denoise
    stages = run_classical_stages(
        mosaic, meta,
        denoise_fn=lambda rgb: nlm_denoise(rgb, float(np.clip(denoise_strength, 0, 1)))
    )
    # Recompute final photofinishing with caller-selected controls.
    from core.pipeline_stages import run_photofinishing
    stages["final_srgb"] = run_photofinishing(
        stages["denoised_rgb"], tone=tone, sharpen=sharpen
    )
    return stages
