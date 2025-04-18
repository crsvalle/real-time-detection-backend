from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.services.yolo_service import detect_car

router = APIRouter()

@router.post("/detect_car")
async def detect_car_route(file: UploadFile = File(...)):
    result = await detect_car(file)
    return JSONResponse(content=result)
