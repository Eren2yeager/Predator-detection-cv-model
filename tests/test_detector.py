import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.config import CLASS_NAMES
from app.detector import Detector, InferenceError, decode_base64_image, serialize_detections


def _png_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 20, 30)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_decode_plain_and_data_url():
    assert decode_base64_image(_png_b64()).size == (32, 24)
    assert decode_base64_image(f"data:image/png;base64,{_png_b64()}").size == (32, 24)


def test_serialize_maps_and_filters():
    result = SimpleNamespace(
        xyxy=np.array([[1.0, 2.0, 10.0, 20.0], [5.0, 5.0, 6.0, 6.0]]),
        confidence=np.array([0.9, 0.4]),
        class_id=np.array([13, 0]),
    )
    output = serialize_detections(result, CLASS_NAMES, {13})
    assert len(output) == 1
    assert output[0]["label"] == "person"
    assert output[0]["class_id"] == 13
    assert output[0]["bbox"] == [1.0, 2.0, 10.0, 20.0]


def test_detect_passes_threshold_and_returns_tuple():
    detector = Detector("unused", warmup=False)
    detector.ready = True
    captured = {}

    def fake_predict(image, threshold):
        captured["threshold"] = threshold
        return SimpleNamespace(
            xyxy=np.array([[0.0, 0.0, 1.0, 1.0]]),
            confidence=np.array([0.8]),
            class_id=np.array([0]),
        )

    detector._predict = fake_predict
    detections, inference_ms = detector.detect(Image.new("RGB", (8, 8)), 0.33)
    assert captured["threshold"] == 0.33
    assert detections[0]["label"] == "bear"
    assert inference_ms >= 0.0


def test_detect_requires_ready():
    detector = Detector("unused", warmup=False)
    with pytest.raises(InferenceError):
        detector.detect(Image.new("RGB", (8, 8)), 0.5)
