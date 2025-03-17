import os

from pymongo import MongoClient


def get_db():
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "crocodile_db")]
    try:
        yield db
    finally:
        client.close()


def get_config():
    return {
        "entity_retrieval_endpoint": os.getenv("ENTITY_RETRIEVAL_ENDPOINT"),
        "entity_bow_endpoint": os.getenv("ENTITY_BOW_ENDPOINT"),
        "api_token": os.getenv("API_TOKEN"),
    }
