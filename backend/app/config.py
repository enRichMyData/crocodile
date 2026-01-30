from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    FASTAPI_APP_NAME: str = "Crocodile FastAPI"
    DEBUG: bool = False
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "crocdile_backend"
    MONGO_SERVER_PORT: int = 27017
    FASTAPI_SERVER_PORT: int = 8000
    ELASTIC_PASSWORD: str = ""  # Default empty, override in .env
    CROCODILE_API_KEY: str | None = None

    EL_DEFAULT_LIMIT: int = 200
    EL_MAX_LIMIT: int = 1000
    EL_INLINE_MAX_ROWS: int = 5000
    EL_PART_MAX_ROWS: int = 50000
    EL_MAX_COLUMNS: int = 200
    EL_MAX_REQUEST_BYTES: int = 10_000_000
    EL_RESULT_SEGMENT_SIZE: int = 500
    EL_INPUT_BATCH_SIZE: int = 200
    EL_WORKER_POLL_INTERVAL_S: float = 2.0
    EL_WORKER_LEASE_SECONDS: int = 60
    EL_LEASE_RENEW_EVERY_S: int = 20

    class Config:
        env_file = str(Path(__file__).parent.parent / ".env")  # Adjust path to .env

settings = Settings()
