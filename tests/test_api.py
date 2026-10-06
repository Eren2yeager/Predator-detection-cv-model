import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main


def _png_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (0, 0, 0)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def client(monkeypatch):
    def fake_load():
        main.detector.ready = True

    def fake_detect(image, threshold, class_ids=None):
        return (
            [
                {
                    "class_id": 13,
                    "label": "person",
                    "confidence": 0.9,
                    "bbox": [0.0, 0.0, 4.0, 4.0],
                }
            ],
            1.0,
        )

    monkeypatch.setattr(main.detector, "load", fake_load)
    monkeypatch.setattr(main.detector, "detect", fake_detect)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["num_classes"] == 20
    assert "person" in body["classes"]


def test_detect_happy_path(client):
    response = client.post("/detect", json={"frame": _png_b64()})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["detections"][0]["label"] == "person"
    assert body["image"] == {"width": 16, "height": 16}


def test_detect_invalid_base64(client):
    response = client.post("/detect", json={"frame": "not-valid-!!!"})
    assert response.status_code == 400


def test_detect_oversize(client, monkeypatch):
    monkeypatch.setattr(main.settings, "max_image_mb", 0.00001)
    response = client.post("/detect", json={"frame": _png_b64()})
    assert response.status_code == 413
