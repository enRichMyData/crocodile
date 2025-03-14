from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    FASTAPI_APP_NAME: str = "Crocodile FastAPI"
    DEBUG: bool = False
    MONGO_URI: str = "mongodb://localhost:27017"  # Default value for DATABASE_URL
    MONGO_SERVER_PORT: int = 27017
    FASTAPI_SERVER_PORT: int = 8000

    class Config:
        env_file = str(Path(__file__).parent.parent / ".env")  # Adjust path to .env


settings = Settings()
