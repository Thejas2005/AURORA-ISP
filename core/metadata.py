"""Stable five-value conditioning contract shared by training and inference.

Capture-derived values take precedence. If noise_sigma or mean_luma is absent,
callers may fill it from the image-condition analyser before constructing the
vector. Defaults remain for backward compatibility, but are explicitly neutral
assumptions rather than measurements.
"""
import numpy as np

META_FIELDS = ("iso", "exposure_ms", "analog_gain", "noise_sigma", "mean_luma")
META_DIM = len(META_FIELDS)


def metadata_with_conditions(meta, conditions):
    """Copy metadata and fill missing estimated features from Conditions.

    Existing capture/user values are preserved. The returned mapping records
    provenance for estimated features without changing the numeric model schema.
    """
    result = dict(meta or {})
    provenance = dict(result.get("metadata_provenance", {}))
    for field in ("noise_sigma", "mean_luma"):
        if field not in result or result[field] is None:
            result[field] = float(getattr(conditions, field))
            provenance[field] = "estimated_from_image"
        else:
            provenance.setdefault(field, "provided")
    result["metadata_provenance"] = provenance
    return result


def metadata_vector(meta):
    """Return float32 [5]: log10(ISO/100), log10(exposure_ms), log2(gain), noise, luma.

    The transformations and clipping bounds are part of the model contract.
    Training must use this exact function or an equivalent implementation.
    """
    meta = dict(meta or {})
    raw = [meta.get("iso", 100), meta.get("exposure_ms", 1.0),
           meta.get("analog_gain", 1.0), meta.get("noise_sigma", 0.0),
           meta.get("mean_luma", 0.5)]
    try:
        iso, exposure, gain, noise, luma = [float(v) for v in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("Metadata values must be numeric") from exc
    vals_in = np.asarray([iso, exposure, gain, noise, luma], dtype=np.float64)
    if not np.isfinite(vals_in).all():
        raise ValueError("Capture metadata must be finite")
    if iso <= 0 or exposure <= 0 or gain <= 0:
        raise ValueError("ISO, exposure_ms and analog_gain must be positive")
    if noise < 0:
        raise ValueError("noise_sigma must be nonnegative")
    vals = [np.log10(iso / 100.0), np.log10(exposure), np.log2(gain), noise, luma]
    limits = [(-2, 3), (-4, 4), (-4, 8), (0, 0.25), (0, 1)]
    out = np.array([np.clip(v, *lim) for v, lim in zip(vals, limits)], dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Metadata transformation produced nonfinite values")
    return out
