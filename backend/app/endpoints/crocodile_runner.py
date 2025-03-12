import os
from pymongo.database import Database
from crocodile import Crocodile

def run_crocodile_task(dataset_name: str, table_name: str, db: Database):
    """
    This function replicates your example of how Crocodile is invoked.
    It initializes Crocodile with the same parameters and runs the process.
    """
    try:
        # Create an instance of Crocodile
        crocodile_instance = Crocodile(
            mongo_uri="mongodb://mongodb:27017/",
            db_name="crocodile_db",
            table_trace_collection_name="table_trace",
            dataset_trace_collection_name="dataset_trace",
            max_candidates=3,
            entity_retrieval_endpoint=os.environ["ENTITY_RETRIEVAL_ENDPOINT"],
            entity_bow_endpoint=os.environ["ENTITY_BOW_ENDPOINT"],
            entity_retrieval_token=os.environ["ENTITY_RETRIEVAL_TOKEN"],
            max_workers=8,
            candidate_retrieval_limit=10,
            model_path="./crocodile/models/default.h5"
        )
        # Run the entity linking process
        crocodile_instance.run()

        print("Entity linking process completed.")
    except Exception as e:
        # If an error occurs, mark the table as failed
        db["table_trace"].update_one(
            {"dataset_name": dataset_name, "table_name": table_name},
            {"$set": {"status": "FAILED"}}
        )
        print(f"Error running Crocodile: {e}")