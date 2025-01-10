import pandas as pd
import sys
import os
from pymongo import MongoClient, ASCENDING
# Adding the level above to sys.path for crocodile module visibility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crocodile import Crocodile

# Load the CSV file into a DataFrame
file_path = '../tables/imdb_top_1000.csv'
df = pd.read_csv(file_path)

# MongoDB connection
client = MongoClient("mongodb://mongodb:27017/")
# Drop the entire crocodile_db database
#client.drop_database("crocodile_db")
db = client["crocodile_db"]

# Drop all collections except 'bow_cache' and 'candidate_cache'
collections_to_keep = ["bow_cache", "candidate_cache"]
all_collections = db.list_collection_names()

for collection in all_collections:
    if collection not in collections_to_keep:
        db[collection].drop()
        print(f"Dropped collection: {collection}")

print("All unwanted collections have been dropped.")


db = client["crocodile_db"]
input_collection = db["input_data"]
table_trace_collection = db["table_trace"]
dataset_trace_collection = db["dataset_trace"]


dataset_name = "test"
table_name = "imdb_top_1000_speed_test"


# Ensure indexes for uniqueness and performance
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

ensure_indexes()

# Define column classifications for NE and LIT types
ne_cols = {
    "0": "OTHER",    # Series_Title
    "7": "PERSON",   # Director
    "8": "PERSON"    # Star1
}

lit_cols = {
    "1": "NUMBER",   # Released_Year
    "2": "NUMBER",   # Runtime (min)
    "3": "STRING",    # Genre
    "4": "NUMBER",   # IMDB_Rating
    "5": "STRING",   # Overview
    "6": "NUMBER",   # Meta_score
    "9": "NUMBER",   # No_of_Votes
    "10": "NUMBER"   # Gross
}

# Store the header in table_trace_collection only once
table_trace_collection.insert_one({
    "dataset_name": dataset_name,
    "table_name": table_name,
    "header": list(df.columns),  # Store the header (column names)
    "total_rows": len(df),
    "processed_rows": 0,
    "status": "PENDING"
})

# Onboard data (values only, no headers)
for index, row in df.iterrows():
    document = {
        "dataset_name": dataset_name,
        "table_name": table_name,
        "row_id": index,
        "data": row.tolist(),  # Store row values as a list instead of a dictionary with headers
        "classified_columns": {
            "NE": ne_cols,
            "LIT": lit_cols
        },
        "context_columns": [str(i) for i in range(len(df.columns))],  # Context columns (by index)
        "correct_qids": {},  # Empty as GT is not available
        "status": "TODO"
    }
    input_collection.insert_one(document)

# Initialize dataset-level trace (if not done earlier)
dataset_trace_collection.update_one(
    {"dataset_name": dataset_name},
    {
        "$setOnInsert": {
            "total_tables": 1,  # Total number of tables
            "processed_tables": 0,
            "total_rows": len(df),  # This will be updated after processing
            "processed_rows": 0,
            "status": "PENDING"
        }
    },
    upsert=True
)

print(f"Data onboarded successfully for dataset '{dataset_name}' and table '{table_name}'.")

input_data = db["input_data"]
model_path = "./trained_models/neural_ranker.h5"



