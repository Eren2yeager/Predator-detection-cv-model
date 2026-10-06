# Wildlife / Animal Detection Service

A small, stateless HTTP service that runs a fine-tuned **RF-DETR Nano** detector over
video frames. Send a frame as base64 JSON, get raw detections back. The future client is
responsible for capturing frames and acting on the results.

## Model

- `RFDETRNano` (RF-DETR), `rfdetr` version `1.11.1`, Apache-2.0.
- Input: RGB resolved to `384x384` internally.
- 20 classes: bear, cat, cattle, crocodile, deer, dog, elephant, fox, hyena, jackal,
  leopard, lion, monkey, person, rabbit, snake, tiger, weasel, wild_boar, wolf.
- Weights: `model/checkpoint_best_total.pth`.

## Setup

Install Git LFS and run `git lfs install` before cloning so the model weights are
downloaded. For an existing clone, run `git lfs pull`. The service also requires the
Python 3.12 install already on this machine. From PowerShell in the repo root:

```powershell
.\run.ps1
```

`run.ps1` creates `.venv`, installs PyTorch from the CUDA 12.4 wheel index, installs the
rest of `requirements.txt`, then starts the server at `http://127.0.0.1:8000`. Use
`.\run.ps1 -SkipInstall` to skip installation on later runs.

Manual equivalent:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Copy `.env.example` to `.env` to change `MODEL_PATH`, `DEVICE` (`auto`/`cuda`/`cpu`),
`DEFAULT_THRESHOLD`, `MAX_IMAGE_MB`, host/port, or `WARMUP`.

## API

Interactive docs: `http://127.0.0.1:8000/docs`.

### `POST /detect`

```json
{ "frame": "<base64 jpeg/png>", "threshold": 0.5, "classes": ["person", "tiger"] }
```

- `frame` (required): base64 image; a `data:image/...;base64,` prefix is accepted.
- `threshold` (optional): 0-1, defaults to `DEFAULT_THRESHOLD`.
- `classes` (optional): filter by class name or id; defaults to all classes.

Response `200`:

```json
{
  "request_id": "uuid",
  "model": { "name": "RFDETRNano", "num_classes": 20, "resolution": 384, "device": "cuda" },
  "image": { "width": 1920, "height": 1080 },
  "count": 1,
  "detections": [
    { "class_id": 13, "label": "person", "confidence": 0.87, "bbox": [x1, y1, x2, y2] }
  ],
  "timing_ms": { "decode": 1.2, "inference": 14.0, "total": 15.5 }
}
```

`bbox` is `[x1, y1, x2, y2]` in pixels of the exact frame submitted.

### `POST /detect/annotated`

Use this endpoint from Swagger when you want to upload an image file and inspect
the boxes visually. In `/docs`, expand **POST `/detect/annotated`**, click **Try it
out**, choose a JPEG or PNG file, optionally set the confidence threshold, and click
**Execute**. A successful call returns a downloadable PNG with class labels and
confidence scores drawn over the detected objects. The response includes
`X-Detection-Count`, `X-Inference-Ms`, and `X-Request-ID` headers. The image size is
limited by `MAX_IMAGE_MB`.

The endpoint uses `multipart/form-data`; install `python-multipart` from
`requirements.txt` before starting the server.

### Other endpoints

- `GET /health` -> `{"status": "ok"}`
- `GET /ready` -> 200 when the model is loaded, else 503
- `GET /model/info` -> model name, class count, resolution, device, class list

Errors return `{"detail": "..."}` with status `400` (bad image), `413` (too large),
`422` (invalid request), `503` (model not ready), or `500` (inference failure).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\smoke_test.py [optional-image.jpg]
```

`scripts/smoke_test.py` loads the real checkpoint (strict state_dict check) and runs one
inference; run it after installing dependencies to confirm the model works end to end.

## Notes

- The service serializes GPU inference with a lock and runs it off the event loop, so
  concurrent requests queue rather than crash. One uvicorn worker is intended.
- No authentication or CORS is enabled; bind to `127.0.0.1` only.
