import os
import time
from jose import JWTError, jwt
from config import settings
from typing import Dict, Any
from pymongo import ASCENDING, MongoClient, TEXT
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

def get_db():
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client["crocodile_backend_db"]

    # Ensure indexes are created with user_id as the first field for better performance
    db.datasets.create_index([("user_id", ASCENDING), ("dataset_name", ASCENDING)], unique=True)

    db.tables.create_index(
        [
            ("user_id", ASCENDING),
            ("dataset_name", ASCENDING),
            ("table_name", ASCENDING),
        ],
        unique=True,
    )

    db.input_data.create_index(
        [
            ("user_id", ASCENDING),
            ("dataset_name", ASCENDING),
            ("table_name", ASCENDING),
            ("row_id", ASCENDING),
        ],
        unique=True,
    )

    # Add new indexes for efficient type and confidence filtering/sorting
    db.input_data.create_index([
        ("user_id", ASCENDING),
        ("dataset_name", ASCENDING),
        ("table_name", ASCENDING),
        ("_id", ASCENDING)
    ])
    
    # Remove old text search logic since we use cell_data for search now
    
    # Keep index for avg_confidence as it's used for whole-row sorting
    db.input_data.create_index([
        ("user_id", ASCENDING),
        ("dataset_name", ASCENDING),
        ("table_name", ASCENDING),
        ("avg_confidence", ASCENDING)
    ])

    # Create indexes for the new cell_data collection
    try:
        # Primary compound index for queries
        db.cell_data.create_index([
            ("user_id", ASCENDING),
            ("dataset_name", ASCENDING),
            ("table_name", ASCENDING),
            ("row_id", ASCENDING),
            ("col_id", ASCENDING),
        ], unique=True)
        
        # Text search index
        db.cell_data.create_index([("cell_text", TEXT)], background=True)
        
        # Confidence sorting index
        db.cell_data.create_index([
            ("user_id", ASCENDING),
            ("dataset_name", ASCENDING),
            ("table_name", ASCENDING),
            ("col_id", ASCENDING),
            ("confidence", ASCENDING),
        ], background=True)
        
        # Types filtering index
        db.cell_data.create_index([
            ("user_id", ASCENDING),
            ("dataset_name", ASCENDING),
            ("table_name", ASCENDING),
            ("col_id", ASCENDING),
            ("types", ASCENDING),
        ], background=True)
        
        print("Created cell_data collection indexes")
    except Exception as e:
        print(f"Error creating cell_data indexes: {str(e)}")

    # Keep existing indexes
    db.input_data.create_index(
        [
            ("user_id", ASCENDING),
            ("dataset_name", ASCENDING),
            ("table_name", ASCENDING),
            ("status", ASCENDING),
        ],
    )

    db.input_data.create_index(
        [
            ("user_id", ASCENDING),
            ("dataset_name", ASCENDING),
            ("table_name", ASCENDING),
            ("ml_status", ASCENDING),
        ],
    )

    db.input_data.create_index(
        [
            ("user_id", ASCENDING),
            ("dataset_name", ASCENDING),
            ("table_name", ASCENDING),
            ("status", ASCENDING),
            ("ml_status", ASCENDING),
        ],
    )

    try:
        yield db
    finally:
        client.close()


def get_crocodile_db():
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client["crocodile_db"]
    try:
        yield db
    finally:
        pass
        # client.close()

bearer_scheme = HTTPBearer()

def verify_token(token: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> Dict[str, Any]:
    """
    Verify JWT token and return payload if valid.
    This should be used as a dependency in FastAPI routes.
    """
    try:
        payload = jwt.decode(token.credentials, settings.JWT_SECRET_KEY, algorithms=["HS256"])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
