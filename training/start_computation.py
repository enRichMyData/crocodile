from pymongo import MongoClient
import sys
import os
# Adding the level above to sys.path for crocodile module visibility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crocodile import Crocodile
import time

# MongoDB connection
client = MongoClient("mongodb://mongodb:27017/")
db = client["crocodile_db"]
process_queue = db["process_queue"]

# Create an instance of Crocodile
crocodile_instance = Crocodile(
    mongo_uri="mongodb://mongodb:27017/",
    db_name="crocodile_db",
    table_trace_collection_name="table_trace",
    dataset_trace_collection_name="dataset_trace",
    max_candidates=3,
    entity_retrieval_endpoint=os.environ["ENTITY_RETRIEVAL_ENDPOINT"],  # Access the entity retrieval endpoint directly from environment variables
    entity_bow_endpoint=os.environ["ENTITY_BOW_ENDPOINT"],  # Access the entity BoW endpoint directly from environment variables
    entity_retrieval_token=os.environ["ENTITY_RETRIEVAL_TOKEN"]  # Access the entity retrieval token directly from environment variables
)

def process_entity_linking():
    """Fetch tasks from the process queue and run entity linking for each table."""
    while True:
        # Fetch the first QUEUED item from the process queue
        task = process_queue.find_one_and_update(
            {"status": "QUEUED"},
            {"$set": {"status": "PROCESSING"}},  # Update the status to PROCESSING
            return_document=True
        )

        if not task:
            print("No tasks in the queue!")
            break

        dataset_name = task["dataset_name"]
        table_name = task["table_name"]

        try:
            # Run the entity linking process using Crocodile
            print(f"Starting entity linking for dataset '{dataset_name}', table '{table_name}'...")
            crocodile_instance.run(dataset_name=dataset_name, table_name=table_name)

            # Update the task status to COMPLETED
            process_queue.update_one(
                {"dataset_name": dataset_name, "table_name": table_name},
                {"$set": {"status": "COMPLETED"}}
            )

            print(f"Entity linking completed for dataset '{dataset_name}', table '{table_name}'.")

        except Exception as e:
            # If there's an error, update the status to FAILED and log the error
            process_queue.update_one(
                {"dataset_name": dataset_name, "table_name": table_name},
                {"$set": {"status": "FAILED"}}
            )
            print(f"Error processing dataset '{dataset_name}', table '{table_name}': {str(e)}")

        # Short delay before processing the next task
        time.sleep(2)

if __name__ == "__main__":
    process_entity_linking()