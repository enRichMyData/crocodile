#!/usr/bin/env python
import argparse
import os
import json
import pandas as pd
from pymongo import MongoClient
from crocodile import Crocodile

def onboard_data(db, df, dataset_name, table_name, classified):
    """
    Onboards data into MongoDB. It stores the CSV header in the table_trace collection,
    and for each row creates a document in input_data that contains the original row data,
    the classified columns (for NE and LIT), and other metadata.
    """
    input_collection = db["input_data"]
    table_trace_collection = db["table_trace"]
    
    # Save header info in table_trace
    table_trace_collection.insert_one({
        "dataset_name": dataset_name,
        "table_name": table_name,
        "header": list(df.columns),
        "total_rows": len(df),
        "processed_rows": 0,
        "status": "PENDING"
    })
    
    # Define context columns as string indices
    context_columns = [str(i) for i in range(len(df.columns))]
    
    # Insert each row as a document
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
    print(f"Data onboarded for dataset '{dataset_name}' and table '{table_name}'.")

def main():
    parser = argparse.ArgumentParser(
        description="CLI for running the Crocodile entity linking system using a CSV and classified columns JSON."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--classified", type=str, required=True,
                        help="Path to the JSON file containing classified columns for NE and LIT types.")
    args = parser.parse_args()
    
    # Load CSV file
    try:
        df = pd.read_csv(args.csv)
        print(f"Loaded CSV file '{args.csv}' with {len(df)} rows.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Load classified columns from JSON file
    try:
        with open(args.classified, "r") as f:
            classified = json.load(f)
        print(f"Loaded classified columns from '{args.classified}'.")
    except Exception as e:
        print(f"Error loading classified columns JSON file: {e}")
        return

    # Use default dataset name and derive table name from CSV filename
    dataset_name = "test"
    table_name = os.path.splitext(os.path.basename(args.csv))[0]
    
    # Connect to MongoDB
    client = MongoClient("mongodb://mongodb:27017/")
    db = client["crocodile_db"]

    # (Optional) Clean up: drop all collections except caches.
    collections_to_keep = ["bow_cache", "candidate_cache"]
    for collection in db.list_collection_names():
        if collection not in collections_to_keep:
            db[collection].drop()
            print(f"Dropped collection: {collection}")
    
    # Onboard the CSV data into MongoDB
    onboard_data(db, df, dataset_name, table_name, classified)
    
    # Retrieve endpoints and tokens from environment variables
    entity_retrieval_endpoint = os.environ.get("ENTITY_RETRIEVAL_ENDPOINT")
    entity_bow_endpoint = os.environ.get("ENTITY_BOW_ENDPOINT")
    entity_retrieval_token = os.environ.get("ENTITY_RETRIEVAL_TOKEN")
    if not (entity_retrieval_endpoint and entity_bow_endpoint and entity_retrieval_token):
        print("Warning: Missing one or more entity service endpoints/tokens from environment variables.")
    
    # Create an instance of Crocodile
    crocodile_instance = Crocodile(
        max_candidates=3,
        entity_retrieval_endpoint=entity_retrieval_endpoint,
        entity_bow_endpoint=entity_bow_endpoint,
        entity_retrieval_token=entity_retrieval_token,
        max_workers=8,
        candidate_retrieval_limit=10,
        model_path="./crocodile/models/default.h5"
    )
    
    # Run the entity linking process
    print("Starting the entity linking process...")
    crocodile_instance.run()
    print("Entity linking process completed.")
    
    # Retrieve processed documents from MongoDB
    input_collection = db["input_data"]
    docs = list(input_collection.find({"dataset_name": dataset_name, "table_name": table_name}))
    
    # Retrieve header info from table_trace
    table_trace = db["table_trace"].find_one({"dataset_name": dataset_name, "table_name": table_name})
    header = table_trace.get("header") if table_trace else [f"col_{i}" for i in range(len(df.columns))]
    
    # Determine which columns are NE from the classified JSON
    ne_cols = classified.get("NE", {})  # keys are string indices
    
    annotated_rows = []
    for doc in docs:
        # Map original row data using the header
        row_data = doc.get("data", [])
        row_dict = dict(zip(header, row_data))
        
        # For each NE column, add annotation columns from el_results
        el_results = doc.get("el_results", {})
        for col_idx in ne_cols:
            # Get the column header name for the NE column
            try:
                col_index = int(col_idx)
                col_header = header[col_index]
            except (ValueError, IndexError):
                col_header = f"col_{col_idx}"
            
            # Default field names for annotation
            id_field = f"{col_header}_id"
            name_field = f"{col_header}_name"
            desc_field = f"{col_header}_desc"
            score_field = f"{col_header}_score"
            
            candidate = None
            # If there are results for this column in el_results, take the first candidate.
            if col_idx in el_results and isinstance(el_results[col_idx], list) and el_results[col_idx]:
                candidate = el_results[col_idx][0]
            
            # Add candidate info or empty values if not found.
            row_dict[id_field] = candidate.get("id") if candidate else ""
            row_dict[name_field] = candidate.get("name") if candidate else ""
            # Use "description" field as "desc"
            row_dict[desc_field] = candidate.get("description") if candidate else ""
            row_dict[score_field] = candidate.get("score") if candidate else ""
        
        annotated_rows.append(row_dict)
    
    # Create DataFrame with original columns first and then appended annotation columns.
    annotated_df = pd.DataFrame(annotated_rows)
    output_file = f"annotated_{table_name}.csv"
    annotated_df.to_csv(output_file, index=False)
    print(f"Annotated table saved to '{output_file}'.")

if __name__ == "__main__":
    main()