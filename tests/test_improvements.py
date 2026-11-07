import numpy as np
import pytest
import torch
from core.raw.bayer import pack_bayer,unpack_bayer,rgb_to_bayer
from core.demosaicing.bilinear import demosaic_bilinear
from core.preprocessing.raw_prep import normalise
from core.pipeline import traditional_isp,hybrid_isp
from models.learned_isp.unet import AuroraUNet

@pytest.mark.parametrize("pattern",["RGGB","BGGR","GRBG","GBRG"])
def test_bayer_roundtrip(pattern):
    m=np.arange(24*32,dtype=np.float32).reshape(24,32)
    assert np.array_equal(unpack_bayer(pack_bayer(m,pattern),pattern),m)

@pytest.mark.parametrize("pattern",["RGGB","BGGR","GRBG","GBRG"])
def test_demosaic_constant_colour(pattern):
    rgb=np.empty((32,32,3),np.float32);rgb[:]=[.2,.5,.8]
    m=rgb_to_bayer(rgb,pattern)
    result=demosaic_bilinear(m,pattern)
    assert result.shape==rgb.shape
    assert np.max(np.abs(result[3:-3,3:-3]-rgb[3:-3,3:-3]))<1e-5
    assert traditional_isp(m,{"pattern":pattern}).shape==rgb.shape

@pytest.mark.parametrize("pattern",["RGGB","BGGR","GRBG","GBRG"])
def test_no_crop_for_all_patterns(pattern):
    m=np.random.default_rng(0).random((32,40),dtype=np.float32)
    out=hybrid_isp(m,{"pattern":pattern})[0]
    assert out.shape==(32,40,3)

def test_normalisation_rejects_invalid_levels():
    with pytest.raises(ValueError): normalise(np.ones((4,4)),10,10)

def test_model_requires_meta_and_backprop():
    model=AuroraUNet(base=8,meta_dim=5,use_meta=True)
    x=torch.rand(1,4,16,16,requires_grad=True)
    with pytest.raises(ValueError): model(x)
    out=model(x,torch.zeros(1,5))
    assert out.shape==(1,3,32,32)
    out.mean().backward()
    assert x.grad is not None

def test_hybrid_rejects_wrong_restorer_output():
    m=np.ones((32,32),np.float32)*.5
    # Force the heavy branch using an RGB array whose restoration is requested.
    from core.pipeline import classical_stage
    assert classical_stage(m,{}).shape==(32,32,3)


def test_unet_rejects_too_small_packed_input():
    model = AuroraUNet(base=8, meta_dim=5, use_meta=True)
    with pytest.raises(ValueError, match=">= 4"):
        model(torch.rand(1, 4, 2, 8), torch.zeros(1, 5))


def test_learned_adapter_metadata_uses_condition_estimates():
    from core.metadata import metadata_with_conditions, metadata_vector
    from core.adaptive_controller.controller import Conditions
    meta = metadata_with_conditions({"iso": 800}, Conditions(.03, .22, .0, .1, .2, .3))
    vector = metadata_vector(meta)
    assert vector.shape == (5,)
    assert vector[3] == pytest.approx(.03)
    assert vector[4] == pytest.approx(.22)
    assert meta["metadata_provenance"]["noise_sigma"] == "estimated_from_image"


def test_metadata_rejects_nonfinite_and_invalid_capture_values():
    from core.metadata import metadata_vector
    with pytest.raises(ValueError, match="finite"):
        metadata_vector({"iso": float("nan")})
    with pytest.raises(ValueError, match="positive"):
        metadata_vector({"exposure_ms": 0})


def test_hybrid_learned_path_fuses_camera_rendered_srgb_without_refinishing(monkeypatch):
    import core.pipeline as pipeline
    from core.adaptive_controller.controller import Params
    mosaic = np.full((16, 16), .4, np.float32)
    learned_srgb = np.full((16, 16, 3), .25, np.float32)
    monkeypatch.setattr(pipeline, "decide", lambda _: Params(.5, .2, .5, 0.0, False))
    output, conditions, _ = pipeline.hybrid_isp(mosaic, {}, learned_model=lambda _m, _meta: learned_srgb)
    classical = pipeline.finish(pipeline.denoise(pipeline._apply_colour(pipeline._front_end(mosaic, {}), {}, .5), .3), tone=.5, sharp=.2)
    # Updated adaptive policy: normal conditions use a learned-dominant blend.
    assert _.learned_weight == 0.85
    expected = 0.15 * classical + 0.85 * learned_srgb
    assert np.allclose(output, expected)


def test_learned_mode_requires_minimum_mosaic_size():
    from core.pipeline import learned_isp
    with pytest.raises(ValueError, match="minimum 8x8"):
        learned_isp(np.ones((4, 4), np.float32), {}, lambda *_: None)
