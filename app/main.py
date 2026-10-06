"""FastAPI application exposing the frame detection endpoint."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.detector import (
    Detector,
    InferenceError,
    InvalidImageError,
    PayloadTooLargeError,
    decode_base64_image,
    decode_image_bytes,
    render_annotated_image,
)
from app.schemas import (
    DetectRequest,
    DetectResponse,
    Detection,
    ImageInfo,
    ModelInfo,
    Timing,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app")

detector = Detector(settings.model_path, device=settings.device, warmup=settings.warmup)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.to_thread(detector.load)
    except Exception as exc:  # pragma: no cover - surfaced via /ready
        logger.exception("Model failed to load: %s", exc)
    yield


app = FastAPI(
    title="Wildlife Detection Service",
    description="RF-DETR Nano object detection over video frames.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if not detector.ready:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    return {"status": "ready"}


@app.get("/model/info")
async def model_info():
    return {
        "name": detector.model_name,
        "num_classes": detector.num_classes,
        "resolution": detector.resolution,
        "device": detector.device,
        "classes": detector.class_names,
        "ready": detector.ready,
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    if not detector.ready:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    max_bytes = int(settings.max_image_mb * 1024 * 1024)
    try:
        image = await asyncio.to_thread(decode_base64_image, request.frame, max_bytes)
    except PayloadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    decoded_at = time.perf_counter()
    threshold = (
        request.threshold if request.threshold is not None else settings.default_threshold
    )
    class_ids = set(request.classes) if request.classes else None
    try:
        detections, inference_ms = await asyncio.to_thread(
            detector.detect, image, threshold, class_ids
        )
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finished = time.perf_counter()
    decode_ms = (decoded_at - started) * 1000.0
    total_ms = (finished - started) * 1000.0
    response = DetectResponse(
        request_id=request_id,
        model=ModelInfo(
            name=detector.model_name,
            num_classes=detector.num_classes,
            resolution=detector.resolution,
            device=detector.device,
        ),
        image=ImageInfo(width=image.width, height=image.height),
        count=len(detections),
        detections=[Detection(**item) for item in detections],
        timing_ms=Timing(
            decode=round(decode_ms, 2),
            inference=round(inference_ms, 2),
            total=round(total_ms, 2),
        ),
    )
    logger.info(
        "request_id=%s image=%dx%d count=%d decode=%.2fms inference=%.2fms total=%.2fms",
        request_id,
        image.width,
        image.height,
        len(detections),
        decode_ms,
        inference_ms,
        total_ms,
    )
    return response


@app.post(
    "/detect/annotated",
    response_class=Response,
    responses={
        200: {"description": "PNG image with detection boxes and labels.", "content": {"image/png": {}}},
        400: {"description": "Invalid or undecodable image."},
        413: {"description": "Image exceeds MAX_IMAGE_MB."},
        503: {"description": "Model is not ready."},
    },
    summary="Upload an image and return it with detection boxes",
)
async def detect_annotated(
    image_file: UploadFile = File(..., description="JPEG or PNG image to detect."),
    threshold: float = Form(
        default=settings.default_threshold,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold from 0 to 1.",
    ),
):
    """Run detection on an uploaded file and return an annotated PNG."""
    if not detector.ready:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    max_bytes = int(settings.max_image_mb * 1024 * 1024)
    raw = await image_file.read(max_bytes + 1)
    try:
        image = await asyncio.to_thread(decode_image_bytes, raw, max_bytes)
    except PayloadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        detections, inference_ms = await asyncio.to_thread(
            detector.detect, image, threshold, None
        )
        annotated = await asyncio.to_thread(render_annotated_image, image, detections)
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Annotated inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    total_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "request_id=%s route=detect/annotated image=%dx%d count=%d inference=%.2fms total=%.2fms",
        request_id,
        image.width,
        image.height,
        len(detections),
        inference_ms,
        total_ms,
    )
    return Response(
        content=annotated,
        media_type="image/png",
        headers={
            "X-Request-ID": request_id,
            "X-Detection-Count": str(len(detections)),
            "X-Inference-Ms": f"{inference_ms:.2f}",
        },
    )


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
