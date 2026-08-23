# CarVision — Backend

FastAPI service that powers vehicle detection and identification. Runs YOLOv8 to find every vehicle in an uploaded image, then — once a vehicle is selected — crops it and runs a ResNet50 classifier to identify make, model, and year.

> **Status:** `/detect_car` works. `/analyze_selected_car` is currently broken — see [Known issues](#known-issues) below.

## Stack

- **FastAPI**
- **Ultralytics YOLOv8** (`yolov8n.pt`) for vehicle detection
- **OpenCV** / **NumPy** for image decoding and cropping
- **Pydantic Settings** for config
- Classifier inference is delegated to `infer.py` — which is meant to be pulled in from [`real-time-detection-ml`](https://github.com/crsvalle/real-time-detection-ml) (see Known issues)

## Getting started

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install fastapi uvicorn "python-multipart" ultralytics opencv-python numpy pydantic-settings
uvicorn app.main:app --reload
```

Runs at `http://localhost:8000` by default. There's no `requirements.txt` in the repo yet — worth generating one (`pip freeze > requirements.txt`) once the environment is finalized.

### Environment variables

Create a `.env` file at the repo root:

```bash
CLIENT_URL=http://localhost:3000
```

Used by `app/middleware/cors.py` to allow requests from the frontend.


## Endpoints

### `POST /detect_car`
Accepts an uploaded image. Runs YOLOv8, filters detections to `car`, `truck`, `bus`, `motorcycle` above a 0.5 confidence threshold, and returns each as a bounding box.

```json
{
  "detections": [
    {
      "class": "car",
      "confidence": 0.87,
      "box": { "x_min": 120, "y_min": 45, "x_max": 610, "y_max": 400 }
    }
  ]
}
```

### `POST /analyze_selected_car`
Accepts the original image file plus a `box` (JSON string, same shape as above) identifying which detected vehicle to identify. Crops the image to that box, base64-encodes the crop for the frontend preview, writes it to a temp file, and calls `predict()` (imported from `infer.py`) to classify it.

```json
{
  "brand": "...",
  "model": "...",
  "year": "...",
  "confidence": "high | medium | low",
  "notes": "...",
  "score": 0.0,
  "alternatives": [...],
  "cropped_image": "<base64 jpeg>"
}
```

## Known issues

**`/analyze_selected_car` cannot currently run.** `yolo_service.py` does:

```python
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from infer import predict
```

expecting `infer.py` at the backend repo root — but `infer.py` only exists in the separate `real-time-detection-ml` repo, under `scripts/`. Until it's copied in, this import fails and the app won't start `analyze_selected_car` calls at all.

Once `infer.py` is copied over, two more mismatches need fixing before it will actually work:

- `infer.py` hardcodes `MODEL_PATH = Path("car_classifier_best.pth")`, loading `torch.load(...)["model_state_dict"]` — but the checkpoint in `real-time-detection-ml` is named `car_classifier.pth` and isn't saved in that dict format.
- `infer.py` requires `car_classes.json` (the ordered list of Stanford Cars class labels) to map predictions back to real names — **this file doesn't exist anywhere in any of the three repos yet** and needs to be generated from the Stanford Cars devkit metadata.

**No `requirements.txt`.** Dependencies currently have to be inferred from imports (`fastapi`, `ultralytics`, `opencv-python`, `numpy`, `pydantic-settings`, `python-multipart` for file uploads).

**No live/streaming endpoint yet.** Both routes are single-image request/response — there's no video-stream or WebSocket endpoint for real-time camera detection.

## Related repos

- [`real-time-detection-frontend`](https://github.com/crsvalle/real-time-detection-frontend) — Next.js UI that calls these endpoints
- [`real-time-detection-ml`](https://github.com/crsvalle/real-time-detection-ml) — training scripts and the classifier `infer.py` this service depends on
