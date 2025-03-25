from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.config import get_settings  

app = FastAPI()

def setup_cors(app):
    settings = get_settings()
    origins = [settings.CLIENT_URL]  # Use the cached CLIENT_URL

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

setup_cors(app)
