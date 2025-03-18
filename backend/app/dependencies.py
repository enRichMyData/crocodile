import os

from pymongo import MongoClient


def get_db():
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client["crocodile_backend_db"]
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
        client.close()