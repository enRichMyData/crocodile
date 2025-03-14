#!/usr/bin/env python
import argparse
import os
import json
import pandas as pd
from pymongo import MongoClient, ASCENDING
from crocodile import Crocodile
from dotenv import load_dotenv  # ✅ Load .env file

# ✅ Load environment variables from .env file in the root directory
load_dotenv()

def ensure_indexes(db):
    """Ensures MongoDB indexes exist for performance and uniqueness."""
    db["input_data"].create_index(
        [("dataset_name", ASCENDING), ("table_name", ASCENDING), ("row_id", ASCENDING)], unique=True
    )
    db["table_trace"].create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING)], unique=True)
    db["dataset_trace"].create_index([("dataset_name", ASCENDING)], unique=True)
    db["process_queue"].create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING)], unique=True)
    db["process_queue"].create_index([("status", ASCENDING)])  # Faster retrieval of pending tasks

def onboard_data(db, df, dataset_name, table_name, classified):
    """Onboards data into MongoDB and ensures dataset tracking in dataset_trace."""
    input_collection = db["input_data"]
    table_trace_collection = db["table_trace"]
    dataset_trace_collection = db["dataset_trace"]
    process_queue = db["process_queue"]

    # Ensure table trace entry exists
    table_trace_collection.update_one(
        {"dataset_name": dataset_name, "table_name": table_name},
        {
            "$setOnInsert": {
                "header": list(df.columns),
                "total_rows": len(df),
                "processed_rows": 0,
                "status": "PENDING"
            }
        },
        upsert=True
    )

    # Ensure dataset trace entry exists
    dataset_trace_collection.update_one(
        {"dataset_name": dataset_name},
        {
            "$setOnInsert": {
                "total_tables": 0,
                "processed_tables": 0,
                "total_rows": 0,
                "processed_rows": 0,
                "status": "PENDING"
            }
        },
        upsert=True
    )

    # Ensure process queue entry exists
    process_queue.update_one(
        {"dataset_name": dataset_name, "table_name": table_name},
        {
            "$setOnInsert": {
                "status": "PENDING",
                "total_rows": len(df),
                "processed_rows": 0
            }
        },
        upsert=True
    )

    # Define context columns as string indices
    context_columns = [str(i) for i in range(len(df.columns))]

    # Insert rows into input_data
    for index, row in df.iterrows():
        document = {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": index,
            "data": row.tolist(),
            "classified_columns": {
                "NE": classified.get("NE", {}),
                "LIT": classified.get("LIT", {}),
                "IGNORED": {}
            },
            "context_columns": context_columns,
            "correct_qids": {},
            "status": "TODO"
        }
        input_collection.insert_one(document)
    
    # Update dataset trace with new row and table count
    dataset_trace_collection.update_one(
        {"dataset_name": dataset_name},
        {
            "$inc": {
                "total_tables": 1,
                "total_rows": len(df)
            }
        }
    )

    print(f"✅ Data onboarded for dataset '{dataset_name}' and table '{table_name}'.")

def fetch_el_results(input_collection, dataset_name, table_name):
    """
    Retrieves processed documents from MongoDB including `el_results`.
    Extracts the first candidate per NE column (if available).
    Uses a **streaming approach** to avoid memory overload on large tables.
    """
    cursor = input_collection.find({"dataset_name": dataset_name, "table_name": table_name})

    extracted_rows = []
    table_trace = input_collection.database["table_trace"].find_one(
        {"dataset_name": dataset_name, "table_name": table_name}
    )
    header = table_trace.get("header") if table_trace else [f"col_{i}" for i in range(len(doc["data"]))]

    for doc in cursor:
        row_data = dict(zip(header, doc["data"]))  # Original row data as dict
        el_results = doc.get("el_results", {})

        # Extract first candidate for each NE column
        for col_idx, col_type in doc["classified_columns"].get("NE", {}).items():
            try:
                col_index = int(col_idx)
                col_header = header[col_index]
            except (ValueError, IndexError):
                col_header = f"col_{col_idx}"

            id_field = f"{col_header}_id"
            name_field = f"{col_header}_name"
            desc_field = f"{col_header}_desc"
            score_field = f"{col_header}_score"

            # Extract first candidate from el_results (if available)
            candidate = el_results.get(col_idx, [{}])[0]

            row_data[id_field] = candidate.get("id", "")
            row_data[name_field] = candidate.get("name", "")
            row_data[desc_field] = candidate.get("description", "")
            row_data[score_field] = candidate.get("score", "")

        extracted_rows.append(row_data)

    return extracted_rows, header

def main():
    parser = argparse.ArgumentParser(
        description="CLI for running the Crocodile entity linking system using a CSV and classified columns JSON."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--classified", type=str, required=True,
                        help="Path to the JSON file containing classified columns for NE and LIT types.")
    parser.add_argument("--db-uri", type=str, default="mongodb://localhost:27017/",
                        help="MongoDB URI to use for connections (override the default for non-Docker env)")
    args = parser.parse_args()
    
    # Load CSV file
    try:
        df = pd.read_csv(args.csv)
        print(f"📂 Loaded CSV file '{args.csv}' with {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return

    # Load classified columns from JSON file
    try:
        with open(args.classified, "r") as f:
            classified = json.load(f)
        print(f"📂 Loaded classified columns from '{args.classified}'.")
    except Exception as e:
        print(f"❌ Error loading classified columns JSON file: {e}")
        return

    dataset_name = "test"
    table_name = os.path.splitext(os.path.basename(args.csv))[0]
    
    # Connect to MongoDB
    print(f"🔗 Connecting to MongoDB at '{args.db_uri}'...")
    client = MongoClient(args.db_uri)
    db = client["crocodile_db"]

    ensure_indexes(db)

    # Drop old collections except caches
    collections_to_keep = ["bow_cache", "candidate_cache"]
    for collection in db.list_collection_names():
        if collection not in collections_to_keep:
            db[collection].drop()
            print(f"🗑️ Dropped collection: {collection}")

    onboard_data(db, df, dataset_name, table_name, classified)

    # ✅ Load environment variables
    entity_retrieval_endpoint = os.getenv("ENTITY_RETRIEVAL_ENDPOINT")
    entity_bow_endpoint = os.getenv("ENTITY_BOW_ENDPOINT")
    entity_retrieval_token = os.getenv("ENTITY_RETRIEVAL_TOKEN")

    if not all([entity_retrieval_endpoint, entity_bow_endpoint, entity_retrieval_token]):
        print("❌ Missing required environment variables. Please check your .env file.")
        return

    crocodile_instance = Crocodile(
        mongo_uri=args.db_uri,
        max_candidates=3,
        entity_retrieval_endpoint=entity_retrieval_endpoint,
        entity_bow_endpoint=entity_bow_endpoint,
        entity_retrieval_token=entity_retrieval_token,
        max_workers=8,
        candidate_retrieval_limit=10,
        model_path="./crocodile/models/default.h5"
    )
    
    print("🚀 Starting the entity linking process...")
    crocodile_instance.run()
    print("✅ Entity linking process completed.")

    input_collection = db["input_data"]
    extracted_rows, _ = fetch_el_results(input_collection, dataset_name, table_name)
    
    pd.DataFrame(extracted_rows).to_csv(f"annotated_{table_name}.csv", index=False)
    print(f"✅ Results saved to 'annotated_{table_name}.csv'.")

if __name__ == "__main__":
    main()