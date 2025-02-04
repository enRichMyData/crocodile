import os
import sys

from pymongo import MongoClient

# Adding the level above to sys.path for crocodile module visibility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from crocodile import Crocodile

# MongoDB connection
client = MongoClient("mongodb://mongodb:27017/")
db = client["crocodile_db"]
input_data = db["input_data"]
model_path = "./trained_models/neural_ranker.h5"

# Create an instance of Crocodile
crocodile_instance = Crocodile(
    mongo_uri="mongodb://mongodb:27017/",
    db_name="crocodile_db",
    table_trace_collection_name="table_trace",
    dataset_trace_collection_name="dataset_trace",
    max_candidates=3,
    entity_retrieval_endpoint=os.environ[
        "ENTITY_RETRIEVAL_ENDPOINT"
    ],  # Access the entity retrieval endpoint directly from environment variables
    entity_bow_endpoint=os.environ[
        "ENTITY_BOW_ENDPOINT"
    ],  # Access the entity BoW endpoint directly from environment variables
    entity_retrieval_token=os.environ[
        "ENTITY_RETRIEVAL_TOKEN"
    ],  # Access the entity retrieval token directly from environment variables
    max_workers=8,
    candidate_retrieval_limit=10,
    model_path=model_path,
)


def process_entity_linking():
    try:
        # Run the Crocodile instance to process rows continuously
        crocodile_instance.run()
    except Exception as e:
        print(f"Error during entity linking process: {str(e)}")

    print("Finished processing input_data.")


if __name__ == "__main__":
    process_entity_linking()
