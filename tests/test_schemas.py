import pytest
from pydantic import ValidationError

from app.schemas import DetectRequest


def test_missing_frame_rejected():
    with pytest.raises(ValidationError):
        DetectRequest()


def test_empty_frame_rejected():
    with pytest.raises(ValidationError):
        DetectRequest(frame="   ")


def test_threshold_bounds():
    with pytest.raises(ValidationError):
        DetectRequest(frame="aGVsbG8=", threshold=1.5)
    with pytest.raises(ValidationError):
        DetectRequest(frame="aGVsbG8=", threshold=-0.1)


def test_classes_normalized_to_ids():
    request = DetectRequest(frame="aGVsbG8=", classes=["person", 0])
    assert request.classes == [0, 13]


def test_unknown_class_rejected():
    with pytest.raises(ValidationError):
        DetectRequest(frame="aGVsbG8=", classes=["dragon"])
    with pytest.raises(ValidationError):
        DetectRequest(frame="aGVsbG8=", classes=[99])
