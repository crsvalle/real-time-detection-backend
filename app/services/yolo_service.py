from ultralytics import YOLO
import cv2
import numpy as np
from fastapi import UploadFile

model = YOLO("app/models/yolov8n.pt")  
class_names = model.names  

async def detect_car(file: UploadFile):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    results = model(image)

    CONF_THRESHOLD = 0.5  # Only keep detections above 50% confidence
    detections = []

    if results and hasattr(results[0], 'boxes'):
        for box in results[0].boxes:
            conf = box.conf[0].item()  # Confidence
            if conf < CONF_THRESHOLD:
                continue  # Skip low-confidence boxes

            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box
            cls = box.cls[0]              # Class index

            class_label = class_names.get(int(cls), f"Unknown Class ({int(cls)})")

            detection = {
                "class": class_label,
                "box": {
                    "x_min": int(x1.item()),
                    "y_min": int(y1.item()),
                    "x_max": int(x2.item()),
                    "y_max": int(y2.item())
                }
            }
            detections.append(detection)


    if detections:
        largest_detection = max(
            detections,
            key=lambda d: (d['box']['x_max'] - d['box']['x_min']) * (d['box']['y_max'] - d['box']['y_min'])
        )
        detections = [largest_detection]

    return {"detections": detections}
