from ultralytics import YOLO
import cv2
import numpy as np
from fastapi import UploadFile
import base64

model = YOLO("app/models/yolov8n.pt")
class_names = model.names

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
CONF_THRESHOLD = 0.5


async def detect_car(file: UploadFile):
    # Read image from UploadFile
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

            detection = {
                "class": class_label,
                "confidence": conf,
                "box": {
                    "x_min": int(x1.item()),
                    "y_min": int(y1.item()),
                    "x_max": int(x2.item()),
                    "y_max": int(y2.item())
                }
            }
            vehicle_detections.append(detection)

    if not vehicle_detections:
        return {"detections": [], "cropped_image": None}

    # Select the largest vehicle
    largest_vehicle = max(
        vehicle_detections,
        key=lambda d: (
            (d["box"]["x_max"] - d["box"]["x_min"]) *
            (d["box"]["y_max"] - d["box"]["y_min"])
        )
    )

    # Crop the largest vehicle
    box = largest_vehicle["box"]
    h, w, _ = image.shape

    x_min = max(0, box["x_min"])
    y_min = max(0, box["y_min"])
    x_max = min(w, box["x_max"])
    y_max = min(h, box["y_max"])

    cropped = image[y_min:y_max, x_min:x_max]

    # Encode cropped image to base64 for frontend or API
    _, buffer = cv2.imencode(".jpg", cropped)
    cropped_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "detections": [largest_vehicle],
        "cropped_image": cropped_base64
    }
