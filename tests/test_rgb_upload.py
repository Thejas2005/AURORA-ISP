import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def make_image(fmt):
    arr = np.zeros((48, 64, 3), dtype=np.uint8)
    arr[..., 0] = np.arange(64, dtype=np.uint8)[None, :] * 3
    arr[..., 1] = 90
    arr[..., 2] = 160
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format=fmt)
    return buf.getvalue()


def test_png_rgb_input_runs_rgb_restoration_modes():
    response = client.post(
        "/api/process",
        files={"file": ("sample.png", make_image("PNG"), "image/png")},
        data={"modes": "rgb_original,rgb_denoise,rgb_denoise_detail"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["input_type"] == "display_rgb"
    assert set(data["results"]) == {"rgb_original", "rgb_denoise", "rgb_denoise_detail"}
    assert all(v["image"].startswith("data:image/png;base64,") for v in data["results"].values())


def test_jpeg_rgb_input_runs_denoising():
    response = client.post(
        "/api/process",
        files={"file": ("sample.jpeg", make_image("JPEG"), "image/jpeg")},
        data={"modes": "rgb_denoise"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["input_type"] == "display_rgb"


def test_rgb_input_rejects_bayer_unet_modes():
    response = client.post(
        "/api/process",
        files={"file": ("sample.jpg", make_image("JPEG"), "image/jpeg")},
        data={"modes": "learned"},
    )
    assert response.status_code == 400
    assert "display RGB" in response.json()["detail"]


def test_jpeg_rgb_full_enhancement_mode():
    response = client.post(
        "/api/process",
        files={"file": ("sample.jpg", make_image("JPEG"), "image/jpeg")},
        data={"modes": "rgb_full_enhance"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "rgb_full_enhance" in data["results"]
    assert data["input_type"] == "display_rgb"


def test_full_enhancement_starts_from_rgb_denoise(monkeypatch):
    import app.main as main
    calls = []
    original = main.process_rgb_image
    def traced(rgb, mode):
        calls.append(mode)
        if mode == "rgb_denoise":
            return np.clip(rgb * 0.8, 0, 1)
        return original(rgb, mode)
    monkeypatch.setattr(main, "process_rgb_image", traced)
    sample = np.full((48, 64, 3), 0.35, dtype=np.float32)
    result = original(sample, "rgb_full_enhance")
    assert calls == ["rgb_denoise"]
    assert result.shape == sample.shape
    assert np.isfinite(result).all()


def test_dng_loader_does_not_require_nonexistent_rawpy_metadata(monkeypatch):
    import sys
    import types
    import app.main as main

    class FakeRaw:
        raw_image_visible = np.full((32, 32), 512, dtype=np.uint16)
        raw_pattern = np.array([[0, 1], [1, 2]], dtype=np.int16)
        color_desc = b"RGBG"
        black_level_per_channel = [0, 0, 0, 0]
        white_level = 1023
        def __enter__(self): return self
        def __exit__(self, *args): return False

    fake_rawpy = types.SimpleNamespace(imread=lambda path: FakeRaw())
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)
    mosaic, meta = main.load_sensor_file("test.dng", b"0" * 2048)
    assert mosaic.shape == (32, 32)
    assert meta["pattern"] == "RGGB"
    assert meta["iso"] == 100.0
    assert meta["exposure_ms"] == 1.0
