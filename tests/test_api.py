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
    monkeypatch.setattr(api, "PREDICTION_LOG", str(tmp_path / "predictions.csv"))
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

def test_predictions_are_logged_with_features_and_label(client, tmp_path):
    import csv

    client.post("/predict", files={"file": ("a.png", _png_bytes(), "image/png")},
                data={"label": "class_3", "batch": "reference"})
    client.post("/predict", files={"file": ("b.png", _png_bytes(), "image/png")})  # no label, no batch

    with open(tmp_path / "predictions.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["label"] == "class_3" and rows[0]["batch"] == "reference"
    assert rows[1]["label"] == "" and rows[1]["batch"] == ""
    assert rows[0]["prediction"] in CLASSES
    assert float(rows[0]["brightness"]) > 0


def test_unknown_labels_are_not_logged_as_labels(client, tmp_path):
    import csv

    client.post("/predict", files={"file": ("a.png", _png_bytes(), "image/png")}, data={"label": "Moon"})
    with open(tmp_path / "predictions.csv", newline="") as f:
        assert next(csv.DictReader(f))["label"] == ""


def test_metrics_endpoint_exposes_model_and_request_metrics(client):
    client.post("/predict", files={"file": ("a.png", _png_bytes(), "image/png")}, data={"label": "class_0"})
    text = client.get("/metrics").text
    assert "model_predictions_total" in text
    assert "model_prediction_confidence_bucket" in text
    assert "model_labelled_predictions_total" in text
    assert 'handler="/predict"' in text
