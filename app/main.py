
from fastapi import FastAPI
from app.api.endpoints import detect  

app = FastAPI()


app.include_router(detect.router, prefix="/api")
