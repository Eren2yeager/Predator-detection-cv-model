"""Load the real checkpoint and run one inference.

Usage:
    python scripts/smoke_test.py [path/to/image.jpg]

With no image argument a random 640x480 image is used; this validates that the
checkpoint loads with a strict state_dict match and that inference runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.detector import Detector  # noqa: E402


def main() -> int:
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        rng = np.random.default_rng(0)
        image = Image.fromarray((rng.random((480, 640, 3)) * 255).astype("uint8"))

    print(f"Loading model from {settings.model_path} ...")
    detector = Detector(settings.model_path, device=settings.device, warmup=False)
    detector.load()
    print(f"Model ready on {detector.device}. Running inference on {image.size} image ...")
    detections, inference_ms = detector.detect(image, settings.default_threshold)
    print(f"Ran in {inference_ms:.1f} ms; {len(detections)} detections")
    for detection in detections:
        print("  ", detection)
    assert detector.ready
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
