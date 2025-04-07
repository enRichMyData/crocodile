import json
import os
import math  # Add import for handling special float values
import time  # Add import for time functions
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from bson import ObjectId
from column_classifier import ColumnClassifier  # added global import
from dependencies import get_crocodile_db, get_db
from endpoints.imdb_example import IMDB_EXAMPLE  # Example input
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError  # added import

from crocodile import Crocodile

router = APIRouter()


class TableUpload(BaseModel):
    table_name: str
    header: List[str]
    total_rows: int
    classified_columns: Optional[Dict[str, Dict[str, str]]] = {}
    data: List[dict]


# Add helper function to format classification results
def format_classification(raw_classification: dict, header: list) -> dict:
    ne_types = {"PERSON", "OTHER", "ORGANIZATION", "LOCATION"}
    ne, lit = {}, {}
    for i, col_name in enumerate(header):
        col_result = raw_classification.get(col_name, {})
        classification = col_result.get("classification", "UNKNOWN")
        if classification in ne_types:
            ne[str(i)] = classification
        else:
            lit[str(i)] = classification
    all_indexes = set(str(i) for i in range(len(header)))
    recognized = set(ne.keys()).union(lit.keys())
    ignored = list(all_indexes - recognized)
    return {"NE": ne, "LIT": lit, "IGNORED": ignored}


# Add helper function to sanitize JSON data with potentially invalid numeric values
def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize a Python object for JSON serialization,
    replacing any float infinity or NaN values with None.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        # Handle special float values that are not JSON compliant
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# Add a function to sync results from Crocodile to backend database
def sync_results_from_crocodile(
    user_id: str,
    dataset_name: str,
    table_name: str,
    total_rows: int = 0
):
    """
    Sync entity linking results from Crocodile to the backend database.
    Uses a continuous loop that will keep checking for updates until all rows are processed.
    
    Args:
        user_id: User ID for multi-tenant isolation
        dataset_name: Name of the dataset
        table_name: Name of the table
        total_rows: Total number of rows in the table
    """
    print(f"Starting result sync for {user_id}/{dataset_name}/{table_name}")

    # Create a direct MongoDB connection for this background task
    import os
    from pymongo import MongoClient
    
    # Get connection string from environment
    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
    client = MongoClient(mongo_uri)
    db = client["crocodile_backend_db"]
    
    try:
        # Create a Crocodile instance for fetching results only
        croco_sync = Crocodile(
            client_id=user_id,
            dataset_name=dataset_name,
            table_name=table_name,
            fetch_result_mode_only=True
        )
        
        # Sync until all rows are processed
        completed_rows = set()
        consecutive_failures = 0
        last_progress_time = time.time()
        max_failures = 5  # Safety limit for consecutive failures
        
        while True:
            # Get all row IDs not yet completed
            row_ids = list(set(range(total_rows)) - completed_rows)
            
            if not row_ids:
                print(f"All rows have been synced for {user_id}/{dataset_name}/{table_name}")
                break
                
            # Process in batches to not overload the system
            batch_size = 50
            batch_processed = False
            
            for i in range(0, len(row_ids), batch_size):
                batch_ids = row_ids[i:i + batch_size]
                
                try:
                    # Fetch results for this batch from Crocodile
                    results = croco_sync.fetch_results(batch_ids)
                    
                    if not results:
                        continue
                    
                    # Process each result
                    pre_completion_count = len(completed_rows)
                    for result in results:
                        row_id = result.get("row_id")
                        status = result.get("status")
                        ml_status = result.get("ml_status")
                        el_results = result.get("el_results", {})
                        
                        # Skip if no results yet
                        if not el_results:
                            continue
                        
                        # Update the backend database
                        db.input_data.update_one(
                            {
                                "user_id": user_id,
                                "dataset_name": dataset_name,
                                "table_name": table_name,
                                "row_id": row_id
                            },
                            {
                                "$set": {
                                    "status": status,
                                    "ml_status": ml_status,
                                    "el_results": el_results,
                                    "last_updated": datetime.now()
                                }
                            },
                            upsert=False
                        )
                        
                        # Mark as completed if done
                        if status == "DONE" and ml_status == "DONE":
                            completed_rows.add(row_id)
                    
                    # Check if we made progress
                    if len(completed_rows) > pre_completion_count:
                        batch_processed = True
                        consecutive_failures = 0
                        last_progress_time = time.time()
                        print(f"Synced batch of {len(results)} results, completed {len(completed_rows)}/{total_rows} rows")
                    
                except Exception as e:
                    print(f"Error syncing results: {str(e)}")
                    consecutive_failures += 1
                    
                    # Break out if we've had too many consecutive failures
                    if consecutive_failures >= max_failures:
                        print(f"Too many consecutive failures ({max_failures}), stopping sync")
                        raise
            
            # Break if all rows are processed
            if len(completed_rows) == total_rows:
                print(f"All {total_rows} rows have been synced successfully")
                break
                
            # If we processed no batches successfully, we should wait longer
            # to avoid hammering the database
            if not batch_processed:
                # Check if we haven't made progress in a long time (5 minutes)
                if time.time() - last_progress_time > 300:
                    print(f"No progress for 5 minutes, completing sync process")
                    break
                
                # Longer pause if no batches were processed
                time.sleep(30)
            else:
                # Normal pause between iterations to not overload the system
                time.sleep(5)
        
        # Update table status based on completion
        completion_percentage = len(completed_rows) / total_rows if total_rows > 0 else 0
        table_status = "completed" if completion_percentage >= 0.95 else "partially_completed"
        
        db.tables.update_one(
            {
                "user_id": user_id,
                "dataset_name": dataset_name, 
                "table_name": table_name
            },
            {"$set": {
                "status": table_status,
                "completion_percentage": round(completion_percentage * 100, 2)
            }}
        )
        print(f"Marked table {dataset_name}/{table_name} as {table_status} ({completion_percentage:.2%} complete)")
        
    except Exception as e:
        print(f"Sync process terminated with error: {str(e)}")
        # Update table with partial completion information if possible
        try:
            completion_percentage = len(completed_rows) / total_rows if total_rows > 0 else 0
            db.tables.update_one(
                {
                    "user_id": user_id,
                    "dataset_name": dataset_name, 
                    "table_name": table_name
                },
                {"$set": {
                    "status": "sync_error",
                    "completion_percentage": round(completion_percentage * 100, 2),
                    "error": str(e)
                }}
            )
        except Exception:
            print("Failed to update table status after error")
    finally:
        # Close our MongoDB connection when done
        client.close()
        print(f"Sync process finished for {dataset_name}/{table_name}")


@router.post("/datasets/{datasetName}/tables/json", status_code=status.HTTP_201_CREATED)
def add_table(
    datasetName: str,
    table_upload: TableUpload = Body(..., example=IMDB_EXAMPLE),
    background_tasks: BackgroundTasks = None,
    user_id: str = Query(None),  # Moved after required parameters
    db: Database = Depends(get_db),
):
    """
    Add a new table to an existing dataset and trigger Crocodile processing in the background.
    """
    # Require user_id for data isolation
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    # Check if dataset exists; if not, create it
    dataset = db.datasets.find_one({"user_id": user_id, "dataset_name": datasetName})  # user_id first
    if not dataset:
        try:
            dataset_id = db.datasets.insert_one(
                {
                    "user_id": user_id,  # Moved to first position
                    "dataset_name": datasetName,
                    "created_at": datetime.now(),
                    "total_tables": 0,
                    "total_rows": 0,
                }
            ).inserted_id
        except DuplicateKeyError:
            raise HTTPException(status_code=400, detail="Duplicate dataset insertion")
    else:
        dataset_id = dataset["_id"]

    if not table_upload.classified_columns:
        df = pd.DataFrame(table_upload.data)
        raw_classification = (
            ColumnClassifier(model_type="fast")
            .classify_multiple_tables([df.head(1024)])[0]
            .get("table_1", {})
        )
        classification = format_classification(raw_classification, table_upload.header)
    else:
        classification = table_upload.classified_columns

    # Create table metadata including classified_columns
    table_metadata = {
        "user_id": user_id,  # Moved to first position
        "dataset_name": datasetName,
        "table_name": table_upload.table_name,
        "header": table_upload.header,
        "total_rows": table_upload.total_rows,
        "created_at": datetime.now(),
        "status": "processing",
        "classified_columns": classification,
    }
    try:
        db.tables.insert_one(table_metadata)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=400, detail="Table with this name already exists in the dataset"
        )

    # Update dataset metadata
    db.datasets.update_one(
        {"_id": dataset_id}, {"$inc": {"total_tables": 1, "total_rows": table_upload.total_rows}}
    )

    # Store each row in the backend database input_data collection
    input_data = []
    for i, row_data in enumerate(table_upload.data):
        # Convert row data to list format if it's a dict
        if isinstance(row_data, dict):
            row_values = [row_data.get(col, None) for col in table_upload.header]
        else:
            row_values = row_data

        input_doc = {
            "user_id": user_id,  # Moved to first position
            "dataset_name": datasetName,
            "table_name": table_upload.table_name,
            "row_id": i,
            "data": row_values,
            "status": "TODO",  # Initial status
            "el_results": {},  # Empty results initially
            "ml_status": "TODO",  # Initial ML status
            "manually_annotated": False,  # Not manually annotated initially
            "created_at": datetime.now(),
        }
        input_data.append(input_doc)

    # Insert all row documents into the input_data collection
    if input_data:
        try:
            db.input_data.insert_many(input_data)
            print(f"Stored {len(input_data)} rows in backend database")
        except Exception as e:
            print(f"Error storing rows in backend database: {e}")
            # Continue anyway - we'll let Crocodile handle the processing

    # Trigger background task with classification passed to Crocodile
    def run_crocodile_task():
        croco = Crocodile(
            input_csv=pd.DataFrame(table_upload.data),
            client_id=user_id,  # Pass user_id as client_id to Crocodile
            dataset_name=datasetName,
            table_name=table_upload.table_name,
            max_candidates=3,
            entity_retrieval_endpoint=os.environ.get("ENTITY_RETRIEVAL_ENDPOINT"),
            entity_bow_endpoint=os.environ.get("ENTITY_BOW_ENDPOINT"),
            entity_retrieval_token=os.environ.get("ENTITY_RETRIEVAL_TOKEN"),
            max_workers=8,
            candidate_retrieval_limit=10,
            model_path="./crocodile/models/default.h5",
            save_output_to_csv=False,
            columns_type=classification,
        )
        croco.run()

    # Add a separate background task to sync results
    def sync_results_task():
        # Wait a moment before starting sync to allow initial processing
        time.sleep(5)
        sync_results_from_crocodile(
            user_id=user_id,
            dataset_name=datasetName, 
            table_name=table_upload.table_name,
            total_rows=table_upload.total_rows
        )

    # Add both tasks to background processing
    background_tasks.add_task(run_crocodile_task)
    background_tasks.add_task(sync_results_task)

    return {
        "message": "Table added successfully.",
        "tableName": table_upload.table_name,
        "datasetName": datasetName,
        "userId": user_id,  # Renamed from clientId to userId
    }


def parse_json_column_classification(column_classification: str = Form("")) -> Optional[dict]:
    # Parse the form field; return None if empty
    if not column_classification:
        return None
    return json.loads(column_classification)


@router.post("/datasets/{datasetName}/tables/csv", status_code=status.HTTP_201_CREATED)
def add_table_csv(
    datasetName: str,
    table_name: str,
    file: UploadFile = File(...),
    column_classification: Optional[dict] = Depends(parse_json_column_classification),
    background_tasks: BackgroundTasks = None,
    user_id: str = Query(None),  # Moved after required parameters
    db: Database = Depends(get_db),
):
    # Require user_id for data isolation
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    # Read CSV file and convert NaN values to None
    df = pd.read_csv(file.file)
    df = df.replace({np.nan: None})  # permanent fix for JSON serialization

    header = df.columns.tolist()
    total_rows = len(df)

    # Use the provided classification; if empty, call ColumnClassifier on a sample
    classification = column_classification if column_classification else {}
    if not classification:
        from column_classifier import ColumnClassifier

        classifier = ColumnClassifier(model_type="fast")
        classification_result = classifier.classify_multiple_tables([df.head(1024)])
        raw_classification = classification_result[0].get("table_1", {})
        classification = format_classification(raw_classification, header)

    # Check if dataset exists; if not, create it
    dataset = db.datasets.find_one({"user_id": user_id, "dataset_name": datasetName})  # user_id first
    if not dataset:
        try:
            dataset_id = db.datasets.insert_one(
                {
                    "user_id": user_id,  # Moved to first position
                    "dataset_name": datasetName,
                    "created_at": datetime.now(),
                    "total_tables": 0,
                    "total_rows": 0,
                }
            ).inserted_id
        except DuplicateKeyError:
            raise HTTPException(status_code=400, detail="Duplicate dataset insertion")
    else:
        dataset_id = dataset["_id"]

    # Create table metadata
    table_metadata = {
        "user_id": user_id,  # Moved to first position
        "dataset_name": datasetName,
        "table_name": table_name,
        "header": header,
        "total_rows": total_rows,
        "created_at": datetime.now(),
        "classified_columns": classification,
    }
    try:
        db.tables.insert_one(table_metadata)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=400, detail="Table with this name already exists in the dataset"
        )

    # Update dataset metadata
    db.datasets.update_one(
        {"_id": dataset_id}, {"$inc": {"total_tables": 1, "total_rows": total_rows}}
    )

    # Store each row in the backend database input_data collection
    input_data = []
    for i, (_, row) in enumerate(df.iterrows()):
        # Convert row to list, handling NaN/None values
        row_values = row.replace({np.nan: None}).tolist()

        input_doc = {
            "user_id": user_id,  # Moved to first position
            "dataset_name": datasetName,
            "table_name": table_name,
            "row_id": i,
            "data": row_values,
            "status": "TODO",
            "el_results": {},
            "ml_status": "TODO",
            "manually_annotated": False,
            "created_at": datetime.now(),
        }
        input_data.append(input_doc)

    # Insert all row documents into the input_data collection
    if input_data:
        try:
            db.input_data.insert_many(input_data)
            print(f"Stored {len(input_data)} rows in backend database")
        except Exception as e:
            print(f"Error storing rows in backend database: {e}")
            # Continue anyway - we'll let Crocodile handle the processing

    # Trigger background task with columns_type passed to Crocodile
    def run_crocodile_task():
        croco = Crocodile(
            input_csv=df,
            client_id=user_id,  # Pass user_id as client_id to Crocodile
            dataset_name=datasetName,
            table_name=table_name,
            entity_retrieval_endpoint=os.environ.get("ENTITY_RETRIEVAL_ENDPOINT"),
            entity_retrieval_token=os.environ.get("ENTITY_RETRIEVAL_TOKEN"),
            max_workers=8,
            candidate_retrieval_limit=10,
            model_path="./crocodile/models/default.h5",
            save_output_to_csv=False,
            columns_type=classification,
            entity_bow_endpoint=os.environ.get("ENTITY_BOW_ENDPOINT"),
        )
        croco.run()

    # Add a separate background task to sync results
    def sync_results_task():
        sync_results_from_crocodile(
            user_id=user_id,
            dataset_name=datasetName, 
            table_name=table_name,
            total_rows=total_rows
        )

    # Add both tasks to background processing
    background_tasks.add_task(run_crocodile_task)
    background_tasks.add_task(sync_results_task)

    return {
        "message": "CSV table added successfully.",
        "tableName": table_name,
        "datasetName": datasetName,
        "userId": user_id,  # Renamed from clientId to userId
    }


@router.get("/datasets")
def get_datasets(
    limit: int = Query(10),
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    user_id: str = Query(None),  # Default parameters are fine here
    db: Database = Depends(get_db),
):
    """
    Get datasets with bi-directional keyset pagination, using ObjectId as the cursor.
    Supports both forward (next_cursor) and backward (prev_cursor) navigation.
    """
    # Require user_id for data isolation
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    # Determine pagination direction and set up query
    query_filter = {"user_id": user_id}  # user_id is already first
    sort_direction = 1  # Default ascending (forward)

    if next_cursor and prev_cursor:
        raise HTTPException(
            status_code=400,
            detail="Only one of next_cursor or prev_cursor should be provided",
        )

    if next_cursor:
        # Forward pagination (get items after the cursor)
        try:
            query_filter["_id"] = {"$gt": ObjectId(next_cursor)}
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")
    elif prev_cursor:
        # Backward pagination (get items before the cursor)
        try:
            query_filter["_id"] = {"$lt": ObjectId(prev_cursor)}
            sort_direction = -1  # Sort descending for backward pagination
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")

    # Execute query with proper sorting
    results = db.datasets.find(query_filter).sort("_id", sort_direction).limit(limit + 1)
    datasets = list(results)

    # Handle pagination metadata
    has_more = len(datasets) > limit
    if has_more:
        datasets = datasets[:limit]  # Remove the extra item

    # If we did backward pagination, reverse the results to maintain consistent order
    if prev_cursor:
        datasets.reverse()

    # Get cursors for next and previous pages
    next_cursor = None
    prev_cursor = None

    if datasets:
        # For previous cursor, we need the ID of the first item
        # But only if we're not on the first page
        if not query_filter.get("_id", {}).get("$gt"):  # No forward pagination filter
            # Check if there are documents before the current page
            if prev_cursor:  # We came backwards, so there are previous items
                prev_cursor = str(datasets[0]["_id"])
            else:
                # We're on first page - check if this is a fresh query or already paginated
                first_id = datasets[0]["_id"]
                if db.datasets.count_documents({"_id": {"$lt": first_id}}) > 0:
                    prev_cursor = str(first_id)
                # Otherwise prev_cursor remains None (we're truly on first page)
        else:
            # We came from a next_cursor, there are previous items
            prev_cursor = str(datasets[0]["_id"])

        # For next cursor, we need the ID of the last item
        # But only if we have more items or we came backwards
        if has_more:
            next_cursor = str(datasets[-1]["_id"])
        elif query_filter.get("_id", {}).get("$lt"):  # We came backwards
            next_cursor = str(datasets[-1]["_id"])

    # Format the response
    for dataset in datasets:
        dataset["_id"] = str(dataset["_id"])
        if "created_at" in dataset:
            dataset["created_at"] = dataset["created_at"].isoformat()

    return {
        "data": datasets,
        "pagination": {"next_cursor": next_cursor, "prev_cursor": prev_cursor},
    }


@router.get("/datasets/{dataset_name}/tables")
def get_tables(
    dataset_name: str,
    limit: int = Query(10),
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    user_id: str = Query(None),  # Moved after required parameters
    db: Database = Depends(get_db),
):
    """
    Get tables for a dataset with bi-directional keyset pagination.
    """
    # Require user_id for data isolation
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    # Ensure dataset exists
    if not db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name}):  # user_id first
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Determine pagination direction
    query_filter = {"user_id": user_id, "dataset_name": dataset_name}  # user_id first
    sort_direction = 1  # Default ascending (forward)

    if next_cursor and prev_cursor:
        raise HTTPException(
            status_code=400,
            detail="Only one of next_cursor or prev_cursor should be provided",
        )

    if next_cursor:
        try:
            query_filter["_id"] = {"$gt": ObjectId(next_cursor)}
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")
    elif prev_cursor:
        try:
            query_filter["_id"] = {"$lt": ObjectId(prev_cursor)}
            sort_direction = -1  # Sort descending for backward pagination
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")

    # Execute query with proper sorting
    results = db.tables.find(query_filter).sort("_id", sort_direction).limit(limit + 1)
    tables = list(results)

    # Handle pagination metadata
    has_more = len(tables) > limit
    if has_more:
        tables = tables[:limit]  # Remove the extra item

    # If we did backward pagination, reverse the results
    if prev_cursor:
        tables.reverse()

    # Get cursors for next and previous pages
    next_cursor = None
    prev_cursor = None

    if tables:
        # For previous cursor, we need the ID of the first item
        # But only if we're not on the first page
        if not query_filter.get("_id", {}).get("$gt"):  # No forward pagination filter
            # Check if there are documents before the current page
            if prev_cursor:  # We came backwards, so there are previous items
                prev_cursor = str(tables[0]["_id"])
            else:
                # We're on first page - check if this is a fresh query or already paginated
                first_id = tables[0]["_id"]
                if db.tables.count_documents(
                    {"user_id": user_id, "dataset_name": dataset_name, "_id": {"$lt": first_id}}
                ) > 0:
                    prev_cursor = str(first_id)
                # Otherwise prev_cursor remains None (we're truly on first page)
        else:
            # We came from a next_cursor, there are previous items
            prev_cursor = str(tables[0]["_id"])

        # For next cursor, we need the ID of the last item
        # But only if we have more items or we came backwards
        if has_more:
            next_cursor = str(tables[-1]["_id"])
        elif query_filter.get("_id", {}).get("$lt"):  # We came backwards
            next_cursor = str(tables[-1]["_id"])

    # Format the response
    for table in tables:
        table["_id"] = str(table["_id"])
        if "created_at" in table:
            table["created_at"] = table["created_at"].isoformat()
        if "completed_at" in table:
            table["completed_at"] = table["completed_at"].isoformat()

    return {
        "dataset": dataset_name,
        "data": tables,
        "pagination": {"next_cursor": next_cursor, "prev_cursor": prev_cursor},
    }


@router.get("/datasets/{dataset_name}/tables/{table_name}")
def get_table(
    dataset_name: str,
    table_name: str,
    limit: int = Query(10),
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    user_id: str = Query(None),  # Moved after required parameters
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Get table data with bi-directional keyset pagination.
    First try to get data from backend database, fallback to crocodile_db if needed.
    """
    # Require user_id for data isolation
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    # Check dataset
    if not db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name}):  # user_id first
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Check table
    table = db.tables.find_one({"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name})
    if not table:
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    header = table.get("header", [])

    # Determine pagination direction
    query_filter = {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}  # user_id first
    sort_direction = 1  # Default ascending (forward)

    if next_cursor and prev_cursor:
        raise HTTPException(
            status_code=400,
            detail="Only one of next_cursor or prev_cursor should be provided",
        )

    if next_cursor:
        try:
            query_filter["_id"] = {"$gt": ObjectId(next_cursor)}
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")
    elif prev_cursor:
        try:
            query_filter["_id"] = {"$lt": ObjectId(prev_cursor)}
            sort_direction = -1  # Sort descending for backward pagination
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")

    # First try to find rows in the backend database
    results = db.input_data.find(query_filter).sort("row_id", sort_direction).limit(limit + 1)
    raw_rows = list(results)

    # If no rows found in backend db, fallback to crocodile_db
    if not raw_rows:
        results = crocodile_db.input_data.find(query_filter).sort("row_id", sort_direction).limit(
            limit + 1
        )
        raw_rows = list(results)

    # Handle pagination metadata
    has_more = len(raw_rows) > limit
    if has_more:
        raw_rows = raw_rows[:limit]  # Remove the extra item

    # If we did backward pagination, reverse the results
    if prev_cursor:
        raw_rows.reverse()

    # Get cursors for next and previous pages
    next_cursor = None
    prev_cursor = None

    if raw_rows:
        # For previous cursor, we need the ID of the first item
        # But only if we're not on the first page
        if not query_filter.get("_id", {}).get("$gt"):  # No forward pagination filter
            # Check if there are documents before the current page
            if prev_cursor:  # We came backwards, so there are previous items
                prev_cursor = str(raw_rows[0]["_id"])
            else:
                # We're on first page - check if this is a fresh query or already paginated
                first_id = raw_rows[0]["_id"]
                # Check both databases for previous pages
                prev_docs_count = db.input_data.count_documents(
                    {
                        "user_id": user_id,
                        "dataset_name": dataset_name,
                        "table_name": table_name,
                        "_id": {"$lt": first_id},
                    }
                )
                if prev_docs_count == 0:
                    prev_docs_count = crocodile_db.input_data.count_documents(
                        {
                            "user_id": user_id,
                            "dataset_name": dataset_name,
                            "table_name": table_name,
                            "_id": {"$lt": first_id},
                        }
                    )
                if prev_docs_count > 0:
                    prev_cursor = str(first_id)
                # Otherwise prev_cursor remains None (we're truly on first page)
        else:
            # We came from a next_cursor, there are previous items
            prev_cursor = str(raw_rows[0]["_id"])

        # For next cursor, we need the ID of the last item
        # But only if we have more items or we came backwards
        if has_more:
            next_cursor = str(raw_rows[-1]["_id"])
        elif query_filter.get("_id", {}).get("$lt"):  # We came backwards
            next_cursor = str(raw_rows[-1]["_id"])

    # Check if there are documents with status or ml_status = TODO or DOING
    table_status_filter = {
        "user_id": user_id,
        "dataset_name": dataset_name,
        "table_name": table_name,
        "$or": [
            {"status": {"$in": ["TODO", "DOING"]}},
            {"ml_status": {"$in": ["TODO", "DOING"]}},
        ],
    }

    # Check both databases for pending documents
    pending_docs_count = db.input_data.count_documents(table_status_filter)
    if pending_docs_count == 0:
        pending_docs_count = crocodile_db.input_data.count_documents(table_status_filter)

    status = "DOING"
    if pending_docs_count == 0:
        status = "DONE"

    # Build a cleaned-up response with *all* candidates
    rows_formatted = []
    for row in raw_rows:
        linked_entities = []
        el_results = row.get("el_results", {})

        # For each column, gather all candidates
        for col_index in range(len(header)):
            candidates = el_results.get(str(col_index), [])
            if candidates:
                # Sanitize candidate data to handle any special float values
                sanitized_candidates = sanitize_for_json(candidates)
                linked_entities.append({"idColumn": col_index, "candidates": sanitized_candidates})

        rows_formatted.append(
            {
                "idRow": row.get("row_id"),
                "data": sanitize_for_json(row.get("data", [])),
                "linked_entities": linked_entities,
            }
        )

    response_data = {
        "data": {
            "datasetName": dataset_name,
            "tableName": table.get("table_name"),
            "status": status,
            "header": header,
            "rows": rows_formatted,
        },
        "pagination": {"next_cursor": next_cursor, "prev_cursor": prev_cursor},
    }

    # Final sanity check - ensure the entire response is safe for JSON serialization
    response_data = sanitize_for_json(response_data)

    return response_data


@router.post("/datasets", status_code=status.HTTP_201_CREATED)
def create_dataset(
    dataset_data: dict = Body(..., example={"dataset_name": "test"}),  # updated example key
    user_id: str = Query(None),  # Moved after required parameters
    db: Database = Depends(get_db),
):
    """
    Create a new dataset.
    """
    # Require user_id for data isolation
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    existing = db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_data.get("dataset_name")}
    )  # updated query key
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset with dataset_name {dataset_data.get('dataset_name')} already exists",
        )

    dataset_data["created_at"] = datetime.now()
    dataset_data["total_tables"] = 0
    dataset_data["total_rows"] = 0
    dataset_data["user_id"] = user_id  # Renamed from client_id to user_id

    try:
        result = db.datasets.insert_one(dataset_data)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Dataset already exists")
    dataset_data["_id"] = str(result.inserted_id)

    return {"message": "Dataset created successfully", "dataset": dataset_data}


@router.delete("/datasets/{dataset_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_name: str,
    user_id: str = Query(None),  # Moved after required parameters
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Delete a dataset by name.
    """
    # Require user_id for data isolation
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    # Check existence using uniform dataset key
    existing = db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name})  # updated query key
    if not existing:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Delete all tables associated with this dataset
    db.tables.delete_many({"user_id": user_id, "dataset_name": dataset_name})

    # Delete dataset
    db.datasets.delete_one({"user_id": user_id, "dataset_name": dataset_name})  # updated query key

    # Delete data from crocodile_db
    deletion_result = crocodile_db.input_data.delete_many({"user_id": user_id, "dataset_name": dataset_name})

    return None


@router.delete(
    "/datasets/{dataset_name}/tables/{table_name}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_table(
    dataset_name: str,
    table_name: str,
    user_id: str = Query(None),  # Moved after required parameters
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Delete a table by name within a dataset.
    """
    # Require user_id for data isolation
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    # Ensure dataset exists using uniform dataset key
    dataset = db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name})  # updated query key
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    table = db.tables.find_one({"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name})
    if not table:
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    row_count = table.get("total_rows", 0)

    # Delete table
    db.tables.delete_one({"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name})

    # Update dataset metadata
    db.datasets.update_one(
        {"user_id": user_id, "dataset_name": dataset_name}, {"$inc": {"total_tables": -1, "total_rows": -row_count}}
    )

    # Delete data from crocodile_db
    deletion_result = crocodile_db.input_data.delete_many(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    )

    return None


class EntityType(BaseModel):
    """Type information for an entity"""
    id: str
    name: str


class EntityCandidate(BaseModel):
    """Complete entity candidate information without matching status"""
    id: str
    name: str
    description: str
    types: List[EntityType]
    # Note: score and match are handled at the annotation level


class AnnotationUpdate(BaseModel):
    """Request model for updating an annotation."""
    entity_id: str
    match: bool = True  # Whether this is the correct entity
    score: Optional[float] = 1.0  # Default to 1.0 for user selections
    notes: Optional[str] = None
    # If providing a new candidate not in the existing list
    candidate_info: Optional[EntityCandidate] = None


@router.put("/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}")
def update_annotation(
    dataset_name: str,
    table_name: str,
    row_id: int,
    column_id: int,
    annotation: AnnotationUpdate,
    user_id: str = Query(None),  # Moved after required parameters
    crocodile_db: Database = Depends(get_crocodile_db),
    db: Database = Depends(get_db),
):
    """
    Update the annotation for a specific cell by marking a candidate as matching.
    This allows users to manually correct or validate entity linking results.

    The annotation can either reference an existing candidate by ID or provide
    a completely new candidate that doesn't exist in the current list.
    """
    # Require user_id for data isolation
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    # Check if dataset and table exist
    if not db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    if not db.tables.find_one({"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}):
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    # Find the row in the database
    row = crocodile_db.input_data.find_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name, "row_id": row_id}
    )

    if not row:
        raise HTTPException(
            status_code=404, detail=f"Row {row_id} not found in table {table_name}"
        )

    # Get the current EL results
    el_results = row.get("el_results", {})
    column_candidates = el_results.get(str(column_id), [])

    # Find if the candidate already exists
    entity_found = False
    target_candidate = None

    for candidate in column_candidates:
        if candidate.get("id") == annotation.entity_id:
            entity_found = True
            target_candidate = candidate
            break

    # Create the updated candidates list
    updated_candidates = []

    # If we have a new candidate to add (not in existing list)
    if not entity_found:
        if not annotation.candidate_info:
            raise HTTPException(
                status_code=400,
                detail=f"Entity with ID {annotation.entity_id} not found in candidates. Please provide 'candidate_info' to add a new candidate.",
            )

        # Convert Pydantic model to dict for MongoDB storage and add annotation data
        new_candidate = annotation.candidate_info.dict()
        new_candidate["match"] = annotation.match
        new_candidate["score"] = annotation.score if annotation.match else None
        if annotation.notes:
            new_candidate["notes"] = annotation.notes

        target_candidate = new_candidate

        # Add all other candidates with match=False and score=null
        for candidate in column_candidates:
            # Skip if we already have this id (prevent duplicates)
            if candidate.get("id") == annotation.entity_id:
                continue

            candidate_copy = dict(candidate)
            candidate_copy["match"] = False
            candidate_copy["score"] = None
            updated_candidates.append(candidate_copy)

        # Add the new candidate
        updated_candidates.append(new_candidate)

    else:
        # Update existing candidates
        for candidate in column_candidates:
            # Skip if we already have this id (prevent duplicates)
            if candidate.get("id") in [c.get("id") for c in updated_candidates]:
                continue

            candidate_copy = dict(candidate)

            # If this is the target entity, update it
            if candidate_copy.get("id") == annotation.entity_id:
                candidate_copy["match"] = annotation.match
                candidate_copy["score"] = annotation.score if annotation.match else None
                if annotation.notes:
                    candidate_copy["notes"] = annotation.notes
            else:
                # Ensure other candidates are not matched
                candidate_copy["match"] = False
                candidate_copy["score"] = None

            updated_candidates.append(candidate_copy)

    # Sort candidates - matched candidate first
    updated_candidates.sort(key=lambda x: (0 if x.get("match") else 1))

    # Update the database with the modified candidates
    el_results[str(column_id)] = updated_candidates

    # Perform the update - set manually_annotated at cell level
    result = crocodile_db.input_data.update_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name, "row_id": row_id},
        {"$set": {"el_results": el_results, "manually_annotated": True}},
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to update annotation")

    # Return the updated entity with all its information
    matched_candidate = next(
        (c for c in updated_candidates if c["id"] == annotation.entity_id), None
    )

    # Sanitize the response to handle any JSON-incompatible values
    return sanitize_for_json(
        {
            "message": "Annotation updated successfully",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id,
            "column_id": column_id,
            "entity": matched_candidate,
            "manually_annotated": True,
        }
    )


@router.delete(
    "/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}/candidates/{entity_id}"
)
def delete_candidate(
    dataset_name: str,
    table_name: str,
    row_id: int,
    column_id: int,
    entity_id: str,
    user_id: str = Query(None),  # Moved after required parameters
    crocodile_db: Database = Depends(get_crocodile_db),
    db: Database = Depends(get_db),
):
    """
    Delete a specific candidate from the entity linking results for a cell.
    This allows users to remove unwanted or incorrect candidate entities.
    """
    # Require user_id for data isolation
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required")
        
    # Check if dataset and table exist
    if not db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    if not db.tables.find_one({"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}):
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    # Find the row in the database
    row = crocodile_db.input_data.find_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name, "row_id": row_id}
    )

    if not row:
        raise HTTPException(
            status_code=404, detail=f"Row {row_id} not found in table {table_name}"
        )

    # Get the current EL results
    el_results = row.get("el_results", {})
    column_candidates = el_results.get(str(column_id), [])

    if not column_candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No entity linking candidates found for column {column_id} in row {row_id}",
        )

    # Check if the entity exists in the candidates
    entity_exists = any(candidate.get("id") == entity_id for candidate in column_candidates)

    if not entity_exists:
        raise HTTPException(
            status_code=404,
            detail=f"Entity with ID {entity_id} not found in candidates for column {column_id}",
        )

    # Create updated candidates list without the specified entity
    updated_candidates = [c for c in column_candidates if c.get("id") != entity_id]

    # Check if we're removing a matched candidate, reorder if needed
    if updated_candidates and not any(c.get("match", False) for c in updated_candidates):
        # No matched candidates remain, potentially select the first one
        if updated_candidates:
            updated_candidates[0]["match"] = True
            updated_candidates[0]["score"] = 1.0  # Default score for manually selected

    # Update the database with the modified candidates
    el_results[str(column_id)] = updated_candidates

    # Perform the update
    result = crocodile_db.input_data.update_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name, "row_id": row_id},
        {
            "$set": {
                "el_results": el_results,
                "manually_annotated": True,  # Mark as manually annotated since we're modifying the candidates
            }
        },
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to delete candidate")

    # Sanitize the response to handle any JSON-incompatible values
    return sanitize_for_json(
        {
            "message": "Candidate deleted successfully",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id,
            "column_id": column_id,
            "entity_id": entity_id,
            "remaining_candidates": len(updated_candidates),
        }
    )
