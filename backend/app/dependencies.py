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
    
    # Create a single text index for global search if needed
    # MongoDB only allows one text index per collection
    try:
        # Check if we already have a text index
        has_text_index = False
        for idx in db.input_data.list_indexes():
            if "textIndexVersion" in idx:
                has_text_index = True
                break
                
        if not has_text_index:
            # Create individual field indexes for the first several data columns
            # which allows MongoDB to use them for regex searches
            for i in range(10):  # Index first 10 columns for better regex performance
                field_name = f"data.{i}"
                db.input_data.create_index([(field_name, ASCENDING)], background=True)
                
            print("Created column indexes for better text search performance")
    except Exception as e:
        print(f"Error creating indexes: {str(e)}")

    # Create B-tree indexes for commonly queried fields
    # These are the first few columns which are likely to be filtered by exact values
    for i in range(5):  # Index commonly accessed columns (0-4)
        try:
            field_name = f"data_{i}"
            db.input_data.create_index([(field_name, ASCENDING)], background=True)
        except Exception as e:
            print(f"Error creating index for {field_name}: {str(e)}")

    # Keep index for avg_confidence as it's used for whole-row sorting
    db.input_data.create_index([
        ("avg_confidence", ASCENDING)
    ])

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
