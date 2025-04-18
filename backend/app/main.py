from typing import Dict, Any
from config import settings
from endpoints.crocodile_api import router
from jose import JWTError, jwt
from dependencies import verify_token
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI(title=settings.FASTAPI_APP_NAME, debug=settings.DEBUG)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, modify this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set up JWT authentication
bearer_scheme = HTTPBearer()

# Include the crocodile router with authentication
app.include_router(router, dependencies=[Depends(verify_token)])

@app.get("/protected")
def protected_route(token_payload: Dict[str, Any] = Depends(verify_token)):
    return {"message": f"Hello {token_payload['email']}"}

@app.get("/")
def read_root():
    return {
        "app_name": settings.FASTAPI_APP_NAME,
        "debug": settings.DEBUG,
        "database_url": settings.MONGO_URI,
        "mongo_server_port": settings.MONGO_SERVER_PORT,
        "fastapi_server_port": settings.FASTAPI_SERVER_PORT,
    }
