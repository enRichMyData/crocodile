from config import settings
from endpoints.crocodile_api import router
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

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

# Include the crocodile router
app.include_router(router)

@app.get("/protected")
def protected_route(token: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        payload = jwt.decode(token.credentials, settings.JWT_SECRET_KEY, algorithms=["HS256"])
    except JWTError:    
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"message": f"Hello {payload['email']}"}


@app.get("/")
def read_root():
    return {
        "app_name": settings.FASTAPI_APP_NAME,
        "debug": settings.DEBUG,
        "database_url": settings.MONGO_URI,
        "mongo_server_port": settings.MONGO_SERVER_PORT,
        "fastapi_server_port": settings.FASTAPI_SERVER_PORT,
    }
