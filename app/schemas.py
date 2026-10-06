"""Pydantic request/response models for the detection API."""
from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from app.config import CLASS_NAMES

_NAME_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


class DetectRequest(BaseModel):
    frame: str = Field(..., description="Base64-encoded JPEG/PNG, optionally a data URL.")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    classes: Optional[List[Union[str, int]]] = Field(
        default=None, description="Optional filter by class name or class id."
    )

    @field_validator("frame")
    @classmethod
    def _frame_present(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("frame must be a non-empty base64 string")
        return value

    @field_validator("classes")
    @classmethod
    def _known_classes(cls, value: Optional[List[Union[str, int]]]) -> Optional[List[int]]:
        if value is None:
            return None
        resolved = set()
        for item in value:
            if isinstance(item, bool):
                raise ValueError(f"invalid class selector: {item!r}")
            if isinstance(item, int):
                if not 0 <= item < len(CLASS_NAMES):
                    raise ValueError(f"unknown class id: {item}")
                resolved.add(item)
            elif isinstance(item, str):
                key = item.strip().lower()
                if key not in _NAME_TO_ID:
                    raise ValueError(f"unknown class name: {item!r}")
                resolved.add(_NAME_TO_ID[key])
            else:
                raise ValueError(f"invalid class selector: {item!r}")
        return sorted(resolved)


class ModelInfo(BaseModel):
    name: str
    num_classes: int
    resolution: int
    device: str


class ImageInfo(BaseModel):
    width: int
    height: int


class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float
    bbox: List[float] = Field(..., description="Pixel coordinates [x1, y1, x2, y2].")


class Timing(BaseModel):
    decode: float
    inference: float
    total: float


class DetectResponse(BaseModel):
    request_id: str
    model: ModelInfo
    image: ImageInfo
    count: int
    detections: List[Detection]
    timing_ms: Timing
