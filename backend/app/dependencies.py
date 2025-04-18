import os

from pymongo import ASCENDING, MongoClient  # added ASCENDING import


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
