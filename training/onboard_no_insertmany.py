import os
import pandas as pd
from column_classifier import ColumnClassifier
from pymongo import MongoClient, ASCENDING
from tqdm import tqdm

# MongoDB connection
client = MongoClient("mongodb://mongodb:27017/")
db = client["crocodile_db"]
input_collection = db["input_data"]
table_trace_collection = db["table_trace"]
dataset_trace_collection = db["dataset_trace"]
process_queue = db["process_queue"]

# Ensure indexes for uniqueness and performance
def ensure_indexes():
    input_collection.create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING), ("row_id", ASCENDING)], unique=True)
    table_trace_collection.create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING)], unique=True)
    dataset_trace_collection.create_index([("dataset_name", ASCENDING)], unique=True)
    process_queue.create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING)], unique=True)
    process_queue.create_index([("status", ASCENDING)])  # Ensure fast retrieval of items by status

ensure_indexes()

datasets = ["Round1_T2D", "Round3_2019", "2T_2020", "Round4_2020", "HardTablesR2", "HardTablesR3"]

# Initialize the column classifier
classifier = ColumnClassifier(model_type='fast')

# Function to get NE columns and correct QIDs from the GT file
def get_ne_cols_and_correct_qids(table_name, cea_gt):
    filtered_cea_gt = cea_gt[cea_gt[0] == table_name]
    ne_cols = {int(row[3]): None for row in filtered_cea_gt.itertuples()}  # Column index as integer
    correct_qids = {f"{int(row[1])-1}-{row[2]}": row[3].split("/")[-1] for _, row in filtered_cea_gt.iterrows()}
    return ne_cols, correct_qids

# Function to determine tag based on classification
def determine_tag(classification):
    return "NE" if classification in ["LOCATION", "ORGANIZATION", "PERSON", "OTHER"] else "LIT"

# Function to onboard data without using insert_many
def onboard_data_batch(dataset_name, table_name, df, ne_cols, lit_cols, correct_qids):
    all_columns = set([str(i) for i in range(len(df.columns))])
    classified_ne_columns = set(ne_cols.keys())
    classified_lit_columns = set(lit_cols.keys())
    unclassified_columns = all_columns - (classified_ne_columns | classified_lit_columns)

    for index, row in df.iterrows():
        document = {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": index,
            "data": row.tolist(),  # Storing data as array
            "classified_columns": {
                "NE": ne_cols,
                "LIT": lit_cols,
                "UNCLASSIFIED": list(unclassified_columns)
            },
            "context_columns": list(all_columns),
            "correct_qids": correct_qids,
            "status": "TODO"
        }
        
        try:
            input_collection.insert_one(document)  # Insert each document individually
        except Exception as e:
            print(f"Error inserting document for row {index}: {e}")

    # Store header separately in table_trace
    table_trace_collection.update_one(
        {"dataset_name": dataset_name, "table_name": table_name},
        {"$set": {
            "header": list(df.columns),  # Store the header
            "total_rows": len(df),
            "processed_rows": 0,
            "status": "PENDING"
        }},
        upsert=True
    )

    # Queue table for processing
    process_queue.update_one(
        {"dataset_name": dataset_name, "table_name": table_name},
        {"$set": {
            "status": "QUEUED",
            "table_name": table_name,
            "dataset_name": dataset_name
        }},
        upsert=True
    )

# Main processing loop for onboarding datasets with debug mode
def process_tables(datasets, max_tables_at_once=5, debug_n_tables=None):
    for dataset in datasets:
        cea_gt = pd.read_csv(f"./Datasets/{dataset}/gt/cea.csv", header=None)
        tables = os.listdir(f"./Datasets/{dataset}/tables")
        if debug_n_tables:
            tables = tables[:debug_n_tables]  # Limit the number of tables for debugging
        batch_tables_data = []
        batch_table_names = []

        for table in tqdm(tables, desc=f"Processing tables for dataset {dataset}..."):
            if table.endswith(".csv"):
                df = pd.read_csv(f"./Datasets/{dataset}/tables/{table}")
                table_name = table.split(".csv")[0]

                df_sampled = df.sample(n=min(100, len(df)), random_state=42)

                # Get NE columns and correct QIDs from GT
                ne_cols_gt, correct_qids = get_ne_cols_and_correct_qids(table_name, cea_gt)

                batch_tables_data.append(df_sampled)
                batch_table_names.append((dataset, table_name, df, ne_cols_gt, correct_qids))

                if len(batch_tables_data) >= max_tables_at_once:
                    process_table_batch(batch_tables_data, batch_table_names)
                    batch_tables_data = []
                    batch_table_names = []

        # Process remaining tables in the batch
        if batch_tables_data:
            process_table_batch(batch_tables_data, batch_table_names)

        # Initialize dataset-level trace after processing all tables
        dataset_trace_collection.update_one(
            {"dataset_name": dataset},
            {"$setOnInsert": {
                "total_tables": len(tables),
                "processed_tables": 0,
                "total_rows": 0,
                "processed_rows": 0,
                "status": "PENDING"
            }},
            upsert=True
        )

# Process a batch of tables
def process_table_batch(batch_tables_data, batch_table_names):
    ner_responses = classifier.classify_multiple_tables(batch_tables_data)

    if ner_responses:
        for table_idx, table_response in enumerate(ner_responses):
            dataset, table_name, df, ne_cols_gt, correct_qids = batch_table_names[table_idx]
            ne_cols_classified = {}
            lit_cols_classified = {}

            # Process each table's response
            table_key = list(table_response.keys())[0]
            ner_info_columns = table_response[table_key]

            for col_name, ner_info in ner_info_columns.items():
                classification = ner_info["classification"]
                tag = determine_tag(classification)

                col_idx = df.columns.get_loc(col_name)  # Get the integer index of the column
                col_idx_str = str(col_idx)

                if col_idx in ne_cols_gt:
                    # Trust GT for NE columns but use NER to update
                    if tag == "LIT":
                        classification = "OTHER"
                    ne_cols_classified[col_idx_str] = classification
                elif tag == "LIT":
                    lit_cols_classified[col_idx_str] = classification

            onboard_data_batch(dataset, table_name, df, ne_cols_classified, lit_cols_classified, correct_qids)

# Example of running the function with batching and debug mode
process_tables(datasets, max_tables_at_once=10)  # Use debug_n_tables to onboard 2 tables per dataset