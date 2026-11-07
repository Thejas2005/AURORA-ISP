import numpy as np
from core.raw.bayer import rgb_to_bayer, pack_rggb, unpack_rggb
from core.demosaicing.bilinear import demosaic_bilinear_rggb
from core.evaluation.metrics import psnr, ssim, delta_e2000
from core.pipeline import traditional_isp, hybrid_isp

def test_pack_roundtrip():
    m = np.random.rand(16, 16).astype(np.float32)
    assert np.array_equal(unpack_rggb(pack_rggb(m)), m)

def test_demosaic_flat_field():
    rgb = np.ones((32, 32, 3), np.float32) * np.array([.2, .5, .8], np.float32)
    out = demosaic_bilinear_rggb(rgb_to_bayer(rgb))
    assert np.abs(out[4:-4, 4:-4] - rgb[4:-4, 4:-4]).max() < 1e-5

def test_metrics_identity():
    a = np.random.rand(32, 32, 3).astype(np.float32)
    assert psnr(a, a) == float("inf") and abs(ssim(a, a) - 1) < 1e-6 and delta_e2000(a, a) < 1e-6

def test_pipelines_shape_range():
    m = np.random.rand(64, 64).astype(np.float32); meta = {"black_level": 0, "white_level": 1}
    for out in (traditional_isp(m, meta), hybrid_isp(m, meta)[0]):
        assert out.shape == (64, 64, 3) and 0 <= out.min() and out.max() <= 1
