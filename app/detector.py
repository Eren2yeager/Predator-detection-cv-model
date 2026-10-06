"""RF-DETR Nano model loading and single-frame inference."""
from __future__ import annotations

import base64
import binascii
import io
import logging
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple

from PIL import Image

from app.config import CLASS_NAMES, MODEL_CONFIG, MODEL_NAME

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised when the model is unavailable or inference fails."""


class InvalidImageError(ValueError):
    """Raised when the base64 payload cannot be decoded into an image."""


class PayloadTooLargeError(ValueError):
    """Raised when the decoded image exceeds the configured size limit."""


def resolve_device(preference: str) -> str:
    """Resolve the requested device to ``cuda`` or ``cpu``."""
    preference = (preference or "auto").strip().lower()
    if preference == "cpu":
        return "cpu"
    if preference not in {"auto", "cuda"}:
        raise ValueError("DEVICE must be one of: auto, cuda, cpu")
    try:
        import torch
    except Exception:
        if preference == "cuda":
            raise InferenceError("CUDA was requested but PyTorch is unavailable.")
        return "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        raise InferenceError("CUDA was requested but is not available.")
    return "cuda" if torch.cuda.is_available() else "cpu"


def decode_base64_image(frame: str, max_bytes: Optional[int] = None) -> Image.Image:
    """Decode a base64 (optionally data-URL) image string into an RGB image."""
    payload = frame.strip()
    if payload.lower().startswith("data:"):
        comma = payload.find(",")
        if comma == -1:
            raise InvalidImageError("Malformed data URL: missing ',' separator.")
        payload = payload[comma + 1 :]
    payload = "".join(payload.split())
    if not payload:
        raise InvalidImageError("Empty image payload.")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError(f"Invalid base64 image data: {exc}") from exc
    return decode_image_bytes(raw, max_bytes)


def decode_image_bytes(raw: bytes, max_bytes: Optional[int] = None) -> Image.Image:
    """Decode raw uploaded image bytes into an RGB image."""
    if max_bytes is not None and len(raw) > max_bytes:
        raise PayloadTooLargeError(
            "Decoded image is "
            f"{len(raw) / (1024 * 1024):.2f} MB which exceeds the "
            f"{max_bytes / (1024 * 1024):.2f} MB limit."
        )
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc
    return image.convert("RGB")


def render_annotated_image(image: Image.Image, detections: Sequence[dict]) -> bytes:
    """Return a PNG copy of ``image`` with detection boxes and labels drawn."""
    from PIL import ImageDraw

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    line_width = max(2, round(min(output.size) / 250))
    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        box = (round(x1), round(y1), round(x2), round(y2))
        color = (0, 230, 118)
        draw.rectangle(box, outline=color, width=line_width)
        label = f'{item["label"]} {item["confidence"]:.2f}'
        text_box = draw.textbbox((0, 0), label)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_top = max(0, box[1] - text_height - 6)
        draw.rectangle(
            (box[0], label_top, box[0] + text_width + 8, label_top + text_height + 6),
            fill=color,
        )
        draw.text((box[0] + 4, label_top + 3), label, fill=(0, 0, 0))

    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    return buffer.getvalue()


def _find_detection_module(wrapper: Any):
    """Find the RF-DETR nn.Module holding ``transformer.*`` weights.

    Current RF-DETR releases expose the module through
    ``RFDETRNano.model.model``: the first ``model`` is a ``ModelContext``, not
    itself an ``nn.Module``. Older releases may expose the module directly.
    """
    from torch import nn

    def holds_detr(module: nn.Module) -> bool:
        return any(name.startswith("transformer.") for name, _ in module.named_parameters())

    attributes = ("model", "module", "detr", "net", "inference_model")
    pending = [wrapper]
    seen = set()
    while pending:
        node = pending.pop(0)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, nn.Module) and holds_detr(node):
            return node
        for attribute in attributes:
            child = getattr(node, attribute, None)
            if child is not None and id(child) not in seen:
                pending.append(child)
    raise InferenceError("Could not locate the RF-DETR module inside the wrapper.")


def _first_result(result: Any) -> Any:
    if isinstance(result, (list, tuple)):
        return result[0] if result else None
    return result


def serialize_detections(result: Any, class_names: Sequence[str], class_ids=None) -> List[dict]:
    """Convert a supervision ``Detections`` object into JSON-friendly dicts."""
    import numpy as np

    detection = _first_result(result)
    if detection is None:
        return []
    boxes = np.asarray(getattr(detection, "xyxy", []))
    scores = np.asarray(getattr(detection, "confidence", []))
    classes = np.asarray(getattr(detection, "class_id", []))
    if boxes.size == 0:
        return []
    output = []
    for box, score, class_id in zip(boxes, scores, classes):
        class_id = int(class_id)
        if class_ids is not None and class_id not in class_ids:
            continue
        x1, y1, x2, y2 = (float(value) for value in box[:4])
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        output.append(
            {
                "class_id": class_id,
                "label": label,
                "confidence": round(float(score), 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )
    return output


class Detector:
    """Loads the fine-tuned checkpoint once and runs single-frame inference."""

    def __init__(self, model_path, device: str = "auto", warmup: bool = True):
        self.model_path = str(model_path)
        self.device = resolve_device(device)
        self.class_names = list(CLASS_NAMES)
        self.model_name = MODEL_NAME
        self.resolution = int(MODEL_CONFIG.get("resolution", 384))
        self.num_classes = len(self.class_names)
        self.ready = False
        self._lock = threading.Lock()
        self._wrapper: Any = None
        self._module: Any = None
        self._predictor: Any = None
        self._warmup_enabled = bool(warmup)

    # ---- loading ---------------------------------------------------------
    def load(self) -> None:
        import torch

        torch.set_grad_enabled(False)
        wrapper = self._build_wrapper()
        module = _find_detection_module(wrapper)
        state = self._read_state_dict()
        # Some RF-DETR builds register this bookkeeping buffer while older
        # checkpoints do not. Preserve the model's initialized value, then
        # require an exact match for every learned parameter and other buffer.
        module_state = module.state_dict()
        kp_buffer = "_kp_active_mask"
        if kp_buffer in module_state and kp_buffer not in state:
            state[kp_buffer] = module_state[kp_buffer]
        try:
            module.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise InferenceError(f"Checkpoint does not match the RF-DETR model: {exc}") from exc
        module.to(self.device)
        module.eval()
        predictor = wrapper if callable(getattr(wrapper, "predict", None)) else module
        self._wrapper = wrapper
        self._module = module
        self._predictor = predictor
        if self._warmup_enabled:
            self._run_warmup()
        self.ready = True
        logger.info(
            "Model ready: %s device=%s classes=%d resolution=%d",
            self.model_name,
            self.device,
            self.num_classes,
            self.resolution,
        )

    def _build_wrapper(self):
        try:
            import rfdetr
        except ImportError as exc:
            raise InferenceError("The 'rfdetr' package is not installed.") from exc
        factory = None
        for name in ("RFDETRNano", "RFDETRN"):
            factory = getattr(rfdetr, name, None)
            if factory is not None:
                break
        if factory is None:
            raise InferenceError("rfdetr does not expose a Nano detector class.")
        config = {key: value for key, value in MODEL_CONFIG.items() if key != "pretrain_weights"}
        config["num_classes"] = self.num_classes
        config["device"] = self.device
        if self.device == "cpu":
            config["amp"] = False
        try:
            return factory(pretrain_weights=None, **config)
        except TypeError as exc:
            logger.warning("Reduced rfdetr constructor arguments (%s).", exc)
            reduced = {
                key: config[key]
                for key in ("num_classes", "resolution", "encoder", "dec_layers", "device")
                if key in config
            }
            return factory(pretrain_weights=None, **reduced)

    def _read_state_dict(self):
        import torch

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state = checkpoint
        else:
            raise InferenceError("Checkpoint does not contain a state_dict.")
        cleaned = {}
        for key, value in state.items():
            cleaned[key[len("model.") :] if key.startswith("model.") else key] = value
        return cleaned

    # ---- inference -------------------------------------------------------
    def _run_warmup(self) -> None:
        try:
            import numpy as np

            dummy = Image.fromarray(
                np.zeros((self.resolution, self.resolution, 3), dtype="uint8")
            )
            self._predict(dummy, 0.99)
            logger.info("Warmup inference completed.")
        except Exception as exc:  # pragma: no cover - warmup is best effort
            logger.warning("Warmup inference failed: %s", exc)

    def _predict(self, image: Image.Image, threshold: float):
        return self._predictor.predict(image, threshold=float(threshold))

    def detect(self, image: Image.Image, threshold: float, class_ids=None) -> Tuple[List[dict], float]:
        """Run inference and return ``(detections, inference_ms)``."""
        if not self.ready:
            raise InferenceError("Model is not ready.")
        started = time.perf_counter()
        with self._lock:
            result = self._predict(image, threshold)
        inference_ms = (time.perf_counter() - started) * 1000.0
        return serialize_detections(result, self.class_names, class_ids), inference_ms
