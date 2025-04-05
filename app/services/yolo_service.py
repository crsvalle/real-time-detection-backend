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

    detections = []

    if results and hasattr(results[0], 'boxes'):
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box
            conf = box.conf[0]           # Confidence
            cls = box.cls[0]             # Class index

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

    return {"detections": detections}
