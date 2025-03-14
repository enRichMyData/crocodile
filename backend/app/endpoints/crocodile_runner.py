import os
from pymongo import MongoClient
from crocodile import Crocodile

# Global Crocodile instance, created once at module load
global_crocodile_instance = Crocodile(
    max_candidates=3,
    entity_retrieval_endpoint=os.environ.get("ENTITY_RETRIEVAL_ENDPOINT"),
    entity_bow_endpoint=os.environ.get("ENTITY_BOW_ENDPOINT"),
    entity_retrieval_token=os.environ.get("ENTITY_RETRIEVAL_TOKEN"),
    max_workers=8,
    candidate_retrieval_limit=10,
    model_path="./crocodile/models/default.h5"
)

LOCK_COLLECTION = "crocodile_lock"

def is_crocodile_running(db):
    """Check if Crocodile is already running via the MongoDB lock."""
    return db[LOCK_COLLECTION].find_one({"status": "RUNNING"}) is not None

def set_crocodile_lock(db, running: bool):
    """Set or release the Crocodile lock in MongoDB."""
    lock_coll = db[LOCK_COLLECTION]
    if running:
        lock_coll.update_one({}, {"$set": {"status": "RUNNING"}}, upsert=True)
    else:
        lock_coll.delete_one({})

def run_crocodile_task():
    """
    Run the global Crocodile instance to process all TODO items.
    Uses a MongoDB lock to ensure that if a run is already in progress, a new one won't be started.
    """
    mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = mongo_client[os.getenv("DB_NAME", "crocodile_db")]
    print("Running Crocodile task...")
    try:
        # Check if another instance is already running via the MongoDB lock
        if is_crocodile_running(db):
            print("Crocodile run is already in progress. Skipping execution.")
            return

        # Set lock to prevent multiple instances
        set_crocodile_lock(db, True)

        print("Starting Crocodile processing...")
        global_crocodile_instance.run()
        print("Crocodile processing completed.")

    except Exception as e:
        print(f"Error running Crocodile: {e}")

    finally:
        # Always release the lock, even if there's an error
        set_crocodile_lock(db, False)
        mongo_client.close()