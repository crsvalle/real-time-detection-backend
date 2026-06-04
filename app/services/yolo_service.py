from ultralytics import YOLO
import cv2
import numpy as np
from fastapi import UploadFile, Form
import json
import base64
import tempfile
import os
import sys

# Add project root to path so infer.py can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from infer import predict

model = YOLO("app/models/yolov8n.pt")
class_names = model.names

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
CONF_THRESHOLD = 0.5



# 1️⃣ Detect all vehicles (unchanged)

async def detect_car(file: UploadFile):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    results = model(image)
    vehicle_detections = []

    if results and hasattr(results[0], "boxes"):
        for box in results[0].boxes:
            conf = box.conf[0].item()
            if conf < CONF_THRESHOLD:
                continue

            cls = int(box.cls[0].item())
            class_label = class_names.get(cls, f"Unknown Class ({cls})")

            if class_label not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0]

            vehicle_detections.append({
                "class": class_label,
                "confidence": conf,
                "box": {
                    "x_min": int(x1.item()),
                    "y_min": int(y1.item()),
                    "x_max": int(x2.item()),
                    "y_max": int(y2.item())
                }
            })

    return {"detections": vehicle_detections}



# 2️⃣ Analyze user-selected vehicle

async def analyze_selected_car(file: UploadFile, box: str = Form(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    box = json.loads(box)
    h, w, _ = image.shape

    x_min = max(0, box["x_min"])
    y_min = max(0, box["y_min"])
    x_max = min(w, box["x_max"])
    y_max = min(h, box["y_max"])

    cropped = image[y_min:y_max, x_min:x_max]

    # Encode cropped image to base64 for frontend preview
    _, buffer = cv2.imencode(".jpg", cropped)
    cropped_base64 = base64.b64encode(buffer).decode("utf-8")

    # Write crop to temp file for infer.py
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(buffer.tobytes())

    try:
        result = predict(tmp_path)
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}
    finally:
        os.unlink(tmp_path)

    return {
        "brand":        result["brand"],
        "model":        result["model"],
        "year":         result["year"],
        "confidence":   result["confidence"],
        "notes":        result["notes"],
        "score":        result["score"],
        "alternatives": result["alternatives"],
        "cropped_image": cropped_base64,   
    }