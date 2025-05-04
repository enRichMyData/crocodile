import os
import time
from jose import JWTError, jwt
from config import settings
from typing import Dict, Any
from pymongo import ASCENDING, MongoClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

# Elasticsearch configuration
ES_HOSTS = os.getenv("ES_HOSTS", "http://elastic:9200").split(",")
ES_INDEX = "table_rows"
ES_BODY = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
            "dataset_name": {"type": "keyword"},
            "table_name": {"type": "keyword"},
            "row_id": {"type": "integer"},
            "status": {"type": "keyword"},
            "ml_status": {"type": "keyword"},
            "manually_annotated": {"type": "boolean"},
            "created_at": {"type": "date"},
            "last_updated": {"type": "date"},
            "data": {
                "type": "nested",
                "properties": {
                    "col_index": {"type": "integer"},
                    "value": {"type": "text"},
                    "confidence_score": {"type": "float"},
                    "types": {"type": "keyword"},
                },
            },
            # store full EL results as opaque JSON
            "el_results": {"type": "object", "enabled": False},
        },
    },
}

# Initialize ES client and ensure index exists
es = Elasticsearch(hosts=ES_HOSTS, request_timeout=60)
# Wait for ES to respond and for cluster health green
for attempt in range(10):
    try:
        if es.ping() and es.cluster.health(wait_for_status="green", request_timeout=5):
            print("Connected to Elasticsearch with green cluster health")
            break
    except Exception:
        pass
    print(f"Waiting for Elasticsearch ({attempt+1}/10)...")
    time.sleep(2)
else:
    raise RuntimeError("Elasticsearch unavailable or cluster not green after retries")

try:
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(index=ES_INDEX, body=ES_BODY)
        print(f"Index '{ES_INDEX}' created")
    else:
        print(f"Index '{ES_INDEX}' already exists")
except Exception as e:
    print(f"Error creating index '{ES_INDEX}': {e}")

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
