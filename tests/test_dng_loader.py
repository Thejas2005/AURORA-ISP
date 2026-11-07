import sys
import types
import numpy as np

from isp.io import load_camera_raw


class FakeRaw:
    raw_type = 0
    num_colors = 3
    raw_pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
    color_desc = b"RGBG"
    black_level_per_channel = [64, 65, 66, 67]
    white_level = 1023
    rgb_xyz_matrix = np.eye(3, dtype=np.float32)
    camera_whitebalance = [2.0, 1.0, 1.5, 1.0]

    def __init__(self):
        self.raw_image_visible = np.full((64, 64), 500, np.uint16)
        tile = np.array([[0, 1], [3, 2]], np.uint8)
        self.raw_colors_visible = np.tile(tile, (32, 32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_dng_loader_uses_per_photosite_black_levels(monkeypatch, tmp_path):
    fake = types.SimpleNamespace(
        RawType=types.SimpleNamespace(Flat=0),
        imread=lambda path: FakeRaw(),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    p = tmp_path / "sample.dng"
    p.write_bytes(b"mock")
    result = load_camera_raw(str(p))

    raw = result["raw"]
    assert raw.shape == (64, 64)
    assert raw.dtype == np.uint16
    assert result["cfa"] == "rggb"
    assert result["fe"]["wb"] is not None
    # Four distinct per-plane black levels must remain distinguishable.
    assert len(np.unique(raw)) >= 3
    assert "as-shot WB" in result["source"]


def test_dng_loader_reports_decoder_errors(monkeypatch, tmp_path):
    def fail(path):
        raise OSError("corrupt raw")
    fake = types.SimpleNamespace(RawType=types.SimpleNamespace(Flat=0), imread=fail)
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    p = tmp_path / "bad.dng"
    p.write_bytes(b"bad")
    import pytest
    from isp.io import InputError
    with pytest.raises(InputError, match="RAW decoder could not read"):
        load_camera_raw(str(p))


def test_dng_loader_aligns_visible_cfa_not_sensor_origin(monkeypatch, tmp_path):
    # Simulate an odd left optical-black margin: sensor pattern starts RGGB,
    # but raw_image_visible starts at the second sensor column (GRBG phase).
    class ShiftedVisibleRaw(FakeRaw):
        def __init__(self):
            super().__init__()
            tile = np.array([[1, 0], [2, 3]], np.uint8)  # visible GRBG
            self.raw_colors_visible = np.tile(tile, (32, 32))
            # Values are keyed by the colour IDs; red and blue remain distinct.
            self.raw_image_visible[0::2, 0::2] = 300  # G
            self.raw_image_visible[0::2, 1::2] = 500  # R
            self.raw_image_visible[1::2, 0::2] = 100  # B
            self.raw_image_visible[1::2, 1::2] = 300  # G

    fake = types.SimpleNamespace(
        RawType=types.SimpleNamespace(Flat=0),
        imread=lambda path: ShiftedVisibleRaw(),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    p = tmp_path / "shifted.dng"
    p.write_bytes(b"mock")
    result = load_camera_raw(str(p))
    raw = result["raw"]
    # Canonical RGGB output must begin with R, G / G, B after phase correction.
    assert raw[0, 0] > raw[0, 1]
    assert raw[0, 0] > raw[1, 0]
    assert raw[1, 1] < raw[0, 1] + 1000


def test_dng_loader_rejects_invalid_black_normalization(monkeypatch, tmp_path):
    import pytest
    from isp.io import InputError

    class BlackRaw(FakeRaw):
        def __init__(self):
            super().__init__()
            self.raw_image_visible[:] = 64

    fake = types.SimpleNamespace(
        RawType=types.SimpleNamespace(Flat=0),
        imread=lambda path: BlackRaw(),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    p = tmp_path / "black.dng"
    p.write_bytes(b"mock")
    with pytest.raises(InputError, match="almost entirely black"):
        load_camera_raw(str(p))
