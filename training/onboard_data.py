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
training_data_collection = db["training_data"]
timing_trace_collection = db["timing_trace"]

# Ensure indexes for uniqueness and performance
def ensure_indexes():
    input_collection.create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING)])  # Ensure fast retrieval of items by dataset and table
    input_collection.create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING), ("row_id", ASCENDING)], unique=True)
    input_collection.create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING), ("status", ASCENDING)])  # Ensure fast retrieval of items by status
    input_collection.create_index([("status", ASCENDING)])  # Ensure fast retrieval of items by status
    table_trace_collection.create_index([("dataset_name", ASCENDING)]) # Ensure unique dataset-level trace
    table_trace_collection.create_index([("table_name", ASCENDING)])  # Ensure fast retrieval of items by table_name
    table_trace_collection.create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING)], unique=True)
    dataset_trace_collection.create_index([("dataset_name", ASCENDING)], unique=True)
    training_data_collection.create_index([("dataset_name", ASCENDING)])  # Ensure fast retrieval of items by dataset
    training_data_collection.create_index([("table_name", ASCENDING)])  # Ensure fast retrieval of items by table
    training_data_collection.create_index([("ml_ranked", ASCENDING)])  # Ensure fast retrieval of items by ml_ranked
    training_data_collection.create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING)])  # Ensure fast retrieval of items by dataset and table
    training_data_collection.create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING), ("ml_ranked", ASCENDING)])  # Ensure fast retrieval of items by dataset, table, and ml_ranked
    timing_trace_collection.create_index([("duration_seconds", ASCENDING)])  # Ensure fast retrieval of items by duration_seconds

ensure_indexes()

datasets = ["Round4_2020", "2T_2020", "Round3_2019", "HardTablesR2", "HardTablesR3", "Round1_T2D"]

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

# Function to batch onboard data into MongoDB
def onboard_data_batch(dataset_name, table_name, df, ne_cols, lit_cols, correct_qids):
    all_columns = set([str(i) for i in range(len(df.columns))])
    classified_ne_columns = set(ne_cols.keys())
    classified_lit_columns = set(lit_cols.keys())
    unclassified_columns = all_columns - (classified_ne_columns | classified_lit_columns)

    documents = []
    for index, row in df.iterrows():
        # Filter correct QIDs relevant for the current row
        correct_qids_for_row = {key: value for key, value in correct_qids.items() if key.startswith(f"{index}-")}
        
        documents.append({
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
            "correct_qids": correct_qids_for_row,
            "status": "TODO"
        })
    
    if documents:
        input_collection.insert_many(documents)  # Batch insert the documents

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
def process_tables(datasets, max_tables_at_once=5, debug_n_tables=None, debug_tables=None):
    for dataset in datasets:
        cea_gt = pd.read_csv(f"./Datasets/{dataset}/gt/cea.csv", header=None)
        tables = os.listdir(f"./Datasets/{dataset}/tables")
        if debug_tables:
            # Use only the specified tables for debugging
            tables = [f"{table}.csv" for table in debug_tables if f"{table}.csv" in tables]
        elif debug_n_tables:
            # Limit the number of tables for debugging
            tables = tables[:debug_n_tables]
        batch_tables_data = []
        batch_table_names = []

        total_rows = 0
        total_tables = 0
        for table in tqdm(tables, desc=f"Processing tables for dataset {dataset}..."):
            if table.endswith(".csv"):
                df = pd.read_csv(f"./Datasets/{dataset}/tables/{table}")
                table_name = table.split(".csv")[0]
                total_rows += len(df)
                total_tables += 1

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
                "total_tables": total_tables,
                "processed_tables": 0,
                "total_rows": total_rows,
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
process_tables(datasets, max_tables_at_once=10, debug_n_tables=10000)  # Use debug_n_tables to onboard n tables per dataset