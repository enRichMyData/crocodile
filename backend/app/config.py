from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    FASTAPI_APP_NAME: str = "Crocodile FastAPI"
    DEBUG: bool = False
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_SERVER_PORT: int = 27017
    FASTAPI_SERVER_PORT: int = 8000
    JWT_SECRET_KEY: str 
    ELASTIC_PASSWORD: str = ""  # Default empty, override in .env

    class Config:
        env_file = str(Path(__file__).parent.parent / ".env")  # Adjust path to .env

settings = Settings()
