import numpy as np
from experiments.benchmarks.evaluate_robustness import make_scenarios


def test_robustness_scenarios_are_deterministic_and_shape_preserving():
    noisy = np.full((32, 40), 0.35, dtype=np.float32)
    clean = noisy.copy()
    meta = {"pattern": "RGGB", "exposure_ms": 1.0}
    a = make_scenarios(noisy, clean, meta, seed=123)
    b = make_scenarios(noisy, clean, meta, seed=123)
    assert [x[0] for x in a] == [x[0] for x in b]
    for left, right in zip(a, b):
        assert left[1].shape == noisy.shape
        assert np.array_equal(left[1], right[1])
        assert np.isfinite(left[1]).all()
    assert a[0][0] == "baseline"
    assert len(a) == 11


def test_robustness_does_not_mutate_source_or_metadata():
    noisy = np.full((32, 40), 0.4, dtype=np.float32)
    clean = noisy.copy()
    meta = {"pattern": "BGGR", "exposure_ms": 2.0}
    original = noisy.copy()
    make_scenarios(noisy, clean, meta, seed=5)
    assert np.array_equal(noisy, original)
    assert meta == {"pattern": "BGGR", "exposure_ms": 2.0}
