# Import Build-in Libraries
import asyncio
from datetime import datetime
import json
import os
from typing import Optional

# Import Third-party Libraries
import numpy as np # type: ignore
import pandas as pd # type: ignore
from bson import ObjectId # type: ignore
from fastapi import ( # type: ignore
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
from fastapi.responses import StreamingResponse # type: ignore
from pymongo import MongoClient # type: ignore
from pymongo.database import Database # type: ignore
from pymongo.errors import DuplicateKeyError # type: ignore

# Import Local Libraries
from dependencies import get_crocodile_db, get_db, verify_token
from endpoints.imdb_example import IMDB_EXAMPLE
from schemas import (
    TableUpload,
    TableAddResponse,
    DatasetListResponse,
    DatasetCreateResponse,
    DatasetCreateRequest,
    AnnotationUpdate,
    DeleteResponse,
    CSVTableUpload,
)
from services.data_service import DataService
from services.result_sync import ResultSyncService
from services.utils import sanitize_for_json

# Import Crocodile Library
from crocodile import Crocodile

router = APIRouter()

# Dataset Endpoints
# -----------------

# GET /datasets
@router.get("/datasets", tags=["datasets"], response_model=DatasetListResponse)
def get_datasets(
    limit: int = Query(10),
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
):
    """
    Get datasets with bi-directional keyset pagination, using ObjectId as the cursor.
    Supports both forward (next_cursor) and backward (prev_cursor) navigation.
    """
    user_id = token_payload.get("email")

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
        dataset["_id"] = str(dataset["_id"]) # Ensure _id is a string for aliasing
        if "created_at" in dataset and isinstance(dataset["created_at"], datetime):
            dataset["created_at"] = dataset["created_at"].isoformat()
        # Ensure all fields required by DatasetResponseItem are present or have defaults
        dataset.setdefault("total_tables", 0)
        dataset.setdefault("total_rows", 0)
        # user_id is already in query_filter and should be in the dataset doc

    return {
        "data": datasets,
        "pagination": {"next_cursor": next_cursor, "prev_cursor": prev_cursor},
    }

# POST /datasets
@router.post("/datasets", tags=["datasets"], response_model=DatasetCreateResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(
    dataset_request: DatasetCreateRequest, # Changed from dataset_data: dict to use Pydantic model
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
):
    """
    Create a new dataset.
    """
    user_id = token_payload.get("email")

    existing = db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_request.dataset_name} # Use attribute access
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset with dataset_name {dataset_request.dataset_name} already exists", # Use attribute access
        )

    # Prepare the document to be inserted into MongoDB
    new_dataset_doc = {
        "dataset_name": dataset_request.dataset_name, # Use attribute access
        "created_at": datetime.now().isoformat(),
        "total_tables": 0,
        "total_rows": 0,
        "user_id": user_id,
    }

    try:
        result = db.datasets.insert_one(new_dataset_doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Dataset already exists")
    
    # Prepare the response dataset object, ensuring it matches DatasetResponseItem structure
    response_dataset = {
        "_id": str(result.inserted_id),
        "dataset_name": new_dataset_doc["dataset_name"],
        "total_tables": new_dataset_doc["total_tables"],
        "total_rows": new_dataset_doc["total_rows"],
        "user_id": new_dataset_doc["user_id"],
        "created_at": new_dataset_doc["created_at"],
    }

    return {"message": "Dataset created successfully", "dataset": response_dataset}

# DELETE /datasets/{dataset_name}
@router.delete("/datasets/{dataset_name}", tags=["datasets"], response_model=DeleteResponse, status_code=status.HTTP_200_OK) # Updated response_model
def delete_dataset(
    dataset_name: str,
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Delete a dataset by name.
    """
    user_id = token_payload.get("email")

    # Check existence using uniform dataset key
    existing = db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_name}
    )  # updated query key
    if not existing:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Delete all tables associated with this dataset
    db.tables.delete_many({"user_id": user_id, "dataset_name": dataset_name})

    # Delete dataset
    db.datasets.delete_one({"user_id": user_id, "dataset_name": dataset_name})  # updated query key

    # Delete data from crocodile_db
    crocodile_db.input_data.delete_many({"user_id": user_id, "dataset_name": dataset_name})

    return {"message": f"Dataset {dataset_name} and its associated tables deleted successfully"}

# Table Endpoints
# ---------------

# GET /tables
@router.get("/datasets/{dataset_name}/tables", tags=["tables"])
def get_tables(
    dataset_name: str,
    limit: int = Query(10),
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
):
    """
    Get tables for a dataset with bi-directional keyset pagination.
    """
    user_id = token_payload.get("email")

    # Ensure dataset exists
    if not db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_name}
    ):  # user_id first
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
                if (
                    db.tables.count_documents(
                        {
                            "user_id": user_id,
                            "dataset_name": dataset_name,
                            "_id": {"$lt": first_id},
                        }
                    )
                    > 0
                ):
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

# GET /tables/{table_name}
@router.get("/datasets/{dataset_name}/tables/{table_name}", tags=["tables"])
def get_table(
    dataset_name: str,
    table_name: str,
    limit: int = Query(10),
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Get table data with bi-directional keyset pagination.
    First try to get data from backend database, fallback to crocodile_db if needed.
    """
    user_id = token_payload.get("email")

    # Check dataset
    if not db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_name}
    ):  # user_id first
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Check table
    table = db.tables.find_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    )
    if not table:
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    header = table.get("header", [])

    # Determine pagination direction
    query_filter = {
        "user_id": user_id,
        "dataset_name": dataset_name,
        "table_name": table_name,
    }  # user_id first
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
        results = (
            crocodile_db.input_data.find(query_filter)
            .sort("row_id", sort_direction)
            .limit(limit + 1)
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

# GET /tables/{table_name}/status
async def get_table_status(
    dataset_name: str,
    table_name: str,
    token_payload: str,
):
    """
    Get the current status of a specific table (streaming).
    Manages its own DB connection for the stream duration.
    """
    user_id = token_payload.get("email")
    client = None  # Initialize client to None
    try:
        # Create a new client specifically for this stream
        client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        db = client["crocodile_backend_db"]

        # Check dataset
        if not db.datasets.find_one(
            {"user_id": user_id, "dataset_name": dataset_name}
        ):
            yield f"data: {json.dumps({'status': 'ERROR', 'detail': f'Dataset {dataset_name} not found'})}\n\n"
            return  # Stop the generator

        # Check table
        table = db.tables.find_one(
            {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
        )
        if not table:
            yield f"data: {json.dumps({'status': 'ERROR', 'detail': f'Table {table_name} not found in dataset {dataset_name}'})}\n\n"
            return  # Stop the generator

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

        rows = table.get("total_rows", 0)

        while True:  # Loop indefinitely until explicitly broken or returned
            # Check databases for pending documents
            pending_docs_count = db.input_data.count_documents(table_status_filter)

            # Send the current status
            pending_percent = (pending_docs_count / rows) * 100
            done_percent = f"{100 - pending_percent:.2f}%"
            pending_percent = f"{pending_percent:.2f}%"
            yield f"data: {json.dumps({'rows': rows, 'pending': pending_docs_count, 'pending_percent': pending_percent, 'done_percent': done_percent})}\n\n"

            if pending_docs_count == 0:
                break  # Exit the loop and finish the stream
            
            await asyncio.sleep(1)  # Check every second

    except Exception as e:
        yield f"data: {json.dumps({'status': 'ERROR', 'detail': str(e)})}\n\n"
    finally:
        if client:
            client.close()

@router.get("/datasets/{dataset_name}/tables/{table_name}/status", tags=["tables"])
async def stream_table_status(
    dataset_name: str,
    table_name: str,
    token_payload: str = Depends(verify_token),
):
    """
    Stream the current status of a specific table (streaming).
    """
    return StreamingResponse(
        get_table_status(dataset_name, table_name, token_payload),
        media_type="text/event-stream",
    )

# POST /tables/json
@router.post("/datasets/{datasetName}/tables/json", response_model=TableAddResponse, status_code=status.HTTP_201_CREATED, tags=["tables"])
def add_table(
    datasetName: str,
    table_upload: TableUpload = Body(..., example=IMDB_EXAMPLE),
    background_tasks: BackgroundTasks = None,
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
):
    """
    Add a new table to an existing dataset and trigger Crocodile processing in the background.
    """
    # Require user_id for data isolation
    user_id = token_payload.get("email")

    try:
        # Convert data to DataFrame for classification if needed
        df = pd.DataFrame(table_upload.data)

        # Get or create column classification
        classification = DataService.get_or_create_column_classification(
            data=df,
            header=table_upload.header,
            provided_classification=table_upload.classified_columns,
        )

        # Create the table and store data
        DataService.create_table(
            db=db,
            user_id=user_id,
            dataset_name=datasetName,
            table_name=table_upload.table_name,
            header=table_upload.header,
            total_rows=table_upload.total_rows,
            classification=classification,
            data_list=table_upload.data,
        )

        # Trigger background task with classification passed to Crocodile
        def run_crocodile_task():
            croco = Crocodile(
                input_csv=df,
                client_id=user_id,
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

        # Add a separate background task to sync results using the service
        def sync_results_task():
            # Create a result sync service and sync results
            mongo_uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
            sync_service = ResultSyncService(mongo_uri=mongo_uri)
            sync_service.sync_results(
                user_id=user_id, dataset_name=datasetName, table_name=table_upload.table_name
            )

        # Add both tasks to background processing
        background_tasks.add_task(run_crocodile_task)
        background_tasks.add_task(sync_results_task)

        return TableAddResponse(
            message="Table added successfully.",
            tableName=table_upload.table_name,
            datasetName=datasetName,
            userId=user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# POST /tables/csv
def parse_json_column_classification(column_classification: str = Form("")) -> Optional[dict]:
    # Parse the form field; return None if empty
    if not column_classification:
        return None
    return json.loads(column_classification)

@router.post("/datasets/{datasetName}/tables/csv", response_model=TableAddResponse, status_code=status.HTTP_201_CREATED, tags=["tables"])
def add_table_csv(
    datasetName: str,
    csv_upload: CSVTableUpload = Depends(),  # Using Pydantic model for validation
    file: UploadFile = File(...),
    column_classification: Optional[dict] = Depends(parse_json_column_classification),
    background_tasks: BackgroundTasks = None,
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
):
    """
    Add a new table from CSV file to an existing dataset and trigger Crocodile processing.
    """
    user_id = token_payload.get("email")
    table_name = csv_upload.table_name  # Using the validated table_name from Pydantic model

    # Validate dataset existence
    dataset = db.datasets.find_one(
        {"user_id": user_id, "dataset_name": datasetName}
    )
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {datasetName} not found")
    
    # Validate table name uniqueness
    existing_table = db.tables.find_one(
        {"user_id": user_id, "dataset_name": datasetName, "table_name": table_name}
    )
    if existing_table:
        raise HTTPException(
            status_code=400, 
            detail=f"Table {table_name} already exists in dataset {datasetName}"
        )

    try:
        # Validate file is a valid CSV
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Uploaded file must be a CSV file")
        
        # Read CSV file and convert NaN values to None
        try:
            df = pd.read_csv(file.file)
            df = df.replace({np.nan: None})  # permanent fix for JSON serialization
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")

        # Validate CSV has content
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file cannot be empty")

        header = df.columns.tolist()
        total_rows = len(df)
        
        # Validate header is not empty
        if not header:
            raise HTTPException(status_code=400, detail="CSV file must have column headers")

        # Get or create column classification
        classification = DataService.get_or_create_column_classification(
            data=df, header=header, provided_classification=column_classification
        )

        # Create the table and store data
        DataService.create_table(
            db=db,
            user_id=user_id,
            dataset_name=datasetName,
            table_name=table_name,
            header=header,
            total_rows=total_rows,
            classification=classification,
            data_df=df,
        )

        # Trigger background task with columns_type passed to Crocodile
        def run_crocodile_task():
            croco = Crocodile(
                input_csv=df,
                client_id=user_id,
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

        # Add a separate background task to sync results using the service
        def sync_results_task():
            # Create a result sync service and sync results
            mongo_uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
            sync_service = ResultSyncService(mongo_uri=mongo_uri)
            sync_service.sync_results(
                user_id=user_id, dataset_name=datasetName, table_name=table_name
            )

        # Add both tasks to background processing
        background_tasks.add_task(run_crocodile_task)
        background_tasks.add_task(sync_results_task)

        return TableAddResponse(
            message="CSV table added successfully.",
            tableName=table_name,
            datasetName=datasetName,
            userId=user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

# DELETE /tables/{table_name}
@router.delete("/datasets/{dataset_name}/tables/{table_name}", tags=["tables"], response_model=DeleteResponse, status_code=status.HTTP_200_OK)
def delete_table(
    dataset_name: str,
    table_name: str,
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Delete a table by name within a dataset.
    """
    user_id = token_payload.get("email")

    # Ensure dataset exists using uniform dataset key
    dataset = db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_name}
    )  # updated query key
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    table = db.tables.find_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    )
    if not table:
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    row_count = table.get("total_rows", 0)

    # Delete table
    db.tables.delete_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    )

    # Update dataset metadata
    db.datasets.update_one(
        {"user_id": user_id, "dataset_name": dataset_name},
        {"$inc": {"total_tables": -1, "total_rows": -row_count}},
    )

    # Delete data from crocodile_db
    crocodile_db.input_data.delete_many(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    )

    return {"message": f"Table {table_name} deleted successfully"} # Added response message

# Annotation Endpoints
# ------------------

# PUT /rows/{row_id}/columns/{column_id}
@router.put("/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}", tags=["annotations"])
def update_annotation(
    dataset_name: str,
    table_name: str,
    row_id: int,
    column_id: int,
    annotation: AnnotationUpdate,
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),  # Only using the backend database now
):
    """
    Update the annotation for a specific cell by marking a candidate as matching.
    This allows users to manually correct or validate entity linking results.

    The annotation can either reference an existing candidate by ID or provide
    a completely new candidate that doesn't exist in the current list.
    """
    user_id = token_payload.get("email")

    # Check if dataset and table exist
    if not db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    if not db.tables.find_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    ):
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    # Find the row in the backend database
    row = db.input_data.find_one(
        {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id,
        }
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

    for candidate in column_candidates:
        if candidate.get("id") == annotation.entity_id:
            entity_found = True
            break

    # Create the updated candidates list
    updated_candidates = []

    # If we have a new candidate to add (not in existing list)
    if not entity_found:
        if not annotation.candidate_info:
            raise HTTPException(
                status_code=400,
                detail=f"""Entity with ID {annotation.entity_id} not found in candidates.
                            Please provide 'candidate_info' to add a new candidate.""",
            )

        # Convert Pydantic model to dict for MongoDB storage and add annotation data
        new_candidate = annotation.candidate_info.dict()
        new_candidate["match"] = annotation.match
        new_candidate["score"] = annotation.score if annotation.match else None
        if annotation.notes:
            new_candidate["notes"] = annotation.notes

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

    # Perform the update in the backend database - set manually_annotated at cell level
    result = db.input_data.update_one(
        {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id,
        },
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

# DELETE /rows/{row_id}/columns/{column_id}/candidates/{entity_id}
@router.delete("/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}/candidates/{entity_id}", tags=["annotations"], response_model=DeleteResponse, status_code=status.HTTP_200_OK)
def delete_candidate(
    dataset_name: str,
    table_name: str,
    row_id: int,
    column_id: int,
    entity_id: str,
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),  # Only using the backend database now
):
    """
    Delete a specific candidate from the entity linking results for a cell.
    This allows users to remove unwanted or incorrect candidate entities.
    """
    user_id = token_payload.get("email")

    # Check if dataset and table exist
    if not db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    if not db.tables.find_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    ):
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    # Find the row in the backend database
    row = db.input_data.find_one(
        {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id,
        }
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

    # Perform the update in the backend database
    # Mark as manually annotated since we're modifying the candidates
    result = db.input_data.update_one(
        {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id,
        },
        {
            "$set": {
                "el_results": el_results,
                "manually_annotated": True,
            }
        },
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to delete candidate")

    # Sanitize the response to handle any JSON-incompatible values
    return sanitize_for_json(
        {
            "message": "Candidate deleted successfully",
            # "dataset_name": dataset_name, # Removed for generic response
            # "table_name": table_name, # Removed for generic response
            # "row_id": row_id, # Removed for generic response
            # "column_id": column_id, # Removed for generic response
            # "entity_id": entity_id, # Removed for generic response
            # "remaining_candidates": len(updated_candidates), # Removed for generic response
        }
    )
