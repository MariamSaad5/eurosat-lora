import io

import pytest

# The training Docker image has no FastAPI, and CI also runs the tests
# inside that image, so these tests skip themselves there.
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from PIL import Image
from transformers import ViTImageProcessor

from src.quantize import quantize_int8, save_quantized
from tests.helpers import tiny_vit

CLASSES = [f"class_{i}" for i in range(10)]


@pytest.fixture
def client(tmp_path, monkeypatch):
    # A tiny INT8 model saved in the same format quantize.py produces
    model_path = tmp_path / "int8.pt"
    processor = ViTImageProcessor(size={"height": 32, "width": 32})
    save_quantized(quantize_int8(tiny_vit().eval()), processor, CLASSES, str(model_path))

    import app.main as api
    monkeypatch.setattr(api, "MODEL_PATH", str(model_path))
    with TestClient(api.app) as c:  # "with" runs the startup code that loads the model
        yield c


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(30, 120, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["classes"] == CLASSES


def test_predict_returns_a_known_class(client):
    response = client.post("/predict", files={"file": ("tile.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_k"]) == 3
    assert body["top_k"][0]["class"] == body["predicted_class"]


def test_predict_rejects_non_images(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert response.status_code == 400