from fastapi import FastAPI
from config import settings
from endpoints.crocodile_api import router

app = FastAPI(title=settings.FASTAPI_APP_NAME, debug=settings.DEBUG)

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
