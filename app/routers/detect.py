from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from app.services.yolo_service import detect_car, analyze_selected_car

router = APIRouter()

# Detect all vehicles (for selection)
@router.post("/detect_car")
async def detect_car_route(file: UploadFile = File(...)):
    result = await detect_car(file)
    return JSONResponse(content=result)


# Analyze user-selected vehicle
@router.post("/analyze_selected_car")
async def analyze_selected_car_route(
    file: UploadFile = File(...),
    box: str = Form(...)
):
    result = await analyze_selected_car(file, box)
    return JSONResponse(content=result)
