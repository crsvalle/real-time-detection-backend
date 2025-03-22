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
            x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
            conf = box.conf[0]  # Get the confidence score
            cls = box.cls[0]  # Get the class index
            
            # Safe class label mapping
            if int(cls) in class_names:  
                class_label = class_names[int(cls)]  # Map class index to label
            else:
                class_label = f"Unknown Class ({int(cls)})"  
                
            detection = {
                "x1": x1.item(),  
                "y1": y1.item(),
                "x2": x2.item(),
                "y2": y2.item(),
                "confidence": conf.item(),
                "class": class_label
            }
            detections.append(detection)
    
    return {"detections": detections}
