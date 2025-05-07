from config import settings
from endpoints.crocodile_api import router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

app = FastAPI(title=settings.FASTAPI_APP_NAME, debug=settings.DEBUG)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS.split(",") if settings.CORS_ALLOWED_ORIGINS else [],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include the crocodile router
app.include_router(router)

@app.get("/")
def read_root():
    return {
        "app_name": settings.FASTAPI_APP_NAME,
        "debug": settings.DEBUG,
        "database_url": settings.MONGO_URI,
        "mongo_server_port": settings.MONGO_SERVER_PORT,
        "fastapi_server_port": settings.FASTAPI_SERVER_PORT,
    }
