import logging
import uuid

from config import settings
from endpoints.crocodile_api import health_router, router as crocodile_router
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

from el_jobs.db import init_indexes as init_el_indexes


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.FASTAPI_APP_NAME, debug=settings.DEBUG)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, modify this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include the crocodile router
app.include_router(crocodile_router)
app.include_router(health_router)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "{\"event\":\"request\",\"request_id\":\"%s\",\"method\":\"%s\",\"path\":\"%s\",\"status_code\":%s}",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


@app.on_event("startup")
def on_startup() -> None:
    init_el_indexes(get_db_from_settings())


def get_db_from_settings():
    from el_jobs.db import get_db

    return get_db()


@app.get("/")
def read_root():
    return {
        "app_name": settings.FASTAPI_APP_NAME,
        "debug": settings.DEBUG,
        "database_url": settings.MONGO_URI,
        "mongo_server_port": settings.MONGO_SERVER_PORT,
        "fastapi_server_port": settings.FASTAPI_SERVER_PORT,
    }
