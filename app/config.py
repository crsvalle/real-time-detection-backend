from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    CLIENT_URL: str  

    model_config = {"env_file": ".env"} 

@lru_cache
def get_settings():
    return Settings() 
