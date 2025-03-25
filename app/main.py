from fastapi import FastAPI
from app.routers import detect
from app.middleware.cors import setup_cors   

app = FastAPI()

setup_cors(app)  

@app.get("/")
async def main():
    return {"message": "Hello World"}

app.include_router(detect.router)
