from core.adaptive_controller.controller import Conditions, Params
from core.pipeline import hybrid_isp
import numpy as np


def test_adaptive_hybrid_keeps_learned_dominant_for_normal_conditions(monkeypatch):
    import core.pipeline as pipeline
    rgb = np.full((16,16,3), 0.35, np.float32)
    cond = Conditions(0.01, 0.35, 0.0, 0.0, 0.1, 0.0)
    monkeypatch.setattr(pipeline, "_front_end", lambda raw, meta: rgb)
    monkeypatch.setattr(pipeline, "_apply_colour", lambda x, meta, strength: x)
    monkeypatch.setattr(pipeline, "analyse", lambda x: cond)
    monkeypatch.setattr(pipeline, "decide", lambda x: Params(0.2,0.2,0.5,0.0,False))
    monkeypatch.setattr(pipeline, "denoise", lambda x, strength: x)
    monkeypatch.setattr(pipeline, "finish", lambda x, tone, sharp: x)
    learned = lambda raw, meta: np.full((16,16,3),0.6,np.float32)
    out, conditions, params = hybrid_isp(np.full((16,16),0.35,np.float32), {}, learned)
    assert params.learned_weight == 0.85
    assert params.fusion_reason.startswith("normal")
    assert out.shape == (16,16,3)


def test_adaptive_hybrid_uses_more_classical_for_underexposure(monkeypatch):
    import core.pipeline as pipeline
    rgb = np.full((16,16,3), 0.05, np.float32)
    cond = Conditions(0.01, 0.05, 0.0, 0.8, 0.1, 0.0)
    monkeypatch.setattr(pipeline, "_front_end", lambda raw, meta: rgb)
    monkeypatch.setattr(pipeline, "_apply_colour", lambda x, meta, strength: x)
    monkeypatch.setattr(pipeline, "analyse", lambda x: cond)
    monkeypatch.setattr(pipeline, "decide", lambda x: Params(0.2,0.2,0.5,0.0,False))
    monkeypatch.setattr(pipeline, "denoise", lambda x, strength: x)
    monkeypatch.setattr(pipeline, "finish", lambda x, tone, sharp: x)
    learned = lambda raw, meta: np.full((16,16,3),0.4,np.float32)
    out, conditions, params = hybrid_isp(np.full((16,16),0.05,np.float32), {}, learned)
    assert params.learned_weight == 0.65
    assert "underexposure" in params.fusion_reason
