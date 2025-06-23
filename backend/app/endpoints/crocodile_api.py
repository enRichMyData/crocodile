import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Union
import base64
import asyncio
import io
import csv
from pymongo import ASCENDING, TEXT

import numpy as np
import concurrent.futures
import pandas as pd
from bson import ObjectId
from dependencies import get_crocodile_db, get_db, verify_token
from endpoints.imdb_example import IMDB_EXAMPLE
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
from fastapi.responses import StreamingResponse
from pymongo import MongoClient
from pydantic import BaseModel
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError
from services.data_service import DataService
from services.result_sync import ResultSyncService
from services.utils import sanitize_for_json, log_info, log_error
from services.task_queue import task_queue

router = APIRouter()

class TableUpload(BaseModel):
    table_name: str
    header: List[str]
    total_rows: int
    classified_columns: Optional[Dict[str, Dict[str, str]]] = {}
    data: List[dict]

@router.post("/datasets/{datasetName}/tables/json", status_code=status.HTTP_201_CREATED)
def add_table(
    datasetName: str,
    table_upload: TableUpload = Body(..., example=IMDB_EXAMPLE),
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
):
    """
    Add a new table to an existing dataset and trigger Crocodile processing in the background.
    """
    # Require user_id for data isolation
    user_id = token_payload.get("email")

    try:
        # Check if dataset exists first
        if not db.datasets.find_one({"user_id": user_id, "dataset_name": datasetName}):
            raise HTTPException(status_code=404, detail=f"Dataset {datasetName} not found")

        # Check if table already exists
        if db.tables.find_one({"user_id": user_id, "dataset_name": datasetName, "table_name": table_upload.table_name}):
            raise HTTPException(status_code=400, detail=f"Table {table_upload.table_name} already exists in dataset {datasetName}")

        # Convert data to DataFrame for classification if needed
        df = pd.DataFrame(table_upload.data)

        # Get or create column classification using the same logic as CSV endpoint
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
            data_df=df,  # Use data_df instead of data_list for consistency
        )

        # Add task to queue instead of running immediately (same as CSV)
        task_data = {
            'user_id': user_id,
            'dataset_name': datasetName,
            'table_name': table_upload.table_name,
            'dataframe': df,
            'classification': classification,
        }
        
        task_id = task_queue.add_csv_task(task_data)

        return {
            "message": "Table queued for processing successfully.",
            "tableName": table_upload.table_name,
            "datasetName": datasetName,
            "userId": user_id,
            "taskId": task_id,
            "status": "queued",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_error(f"Error processing JSON upload for {user_id}/{datasetName}/{table_upload.table_name}", e)
        raise HTTPException(status_code=500, detail=f"Error processing JSON: {str(e)}")

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
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
):
    """
    Add a new table from CSV file to an existing dataset and trigger Crocodile processing.
    Uses a task queue to prevent resource exhaustion from concurrent uploads.
    """
    user_id = token_payload.get("email")

    try:
        # Check file size before processing (500MB limit)
        max_file_size = 500 * 1024 * 1024  # 500 MB in bytes
        file_size = 0
        
        # Read file content to check size
        file_content = file.file.read()
        file_size = len(file_content)
        
        if file_size > max_file_size:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size allowed is {max_file_size // (1024*1024)}MB. Your file is {file_size // (1024*1024):.1f}MB."
            )
        
        # Reset file pointer for pandas
        file.file.seek(0)

        # Check if dataset exists first
        if not db.datasets.find_one({"user_id": user_id, "dataset_name": datasetName}):
            raise HTTPException(status_code=404, detail=f"Dataset {datasetName} not found")

        # Check if table already exists
        if db.tables.find_one({"user_id": user_id, "dataset_name": datasetName, "table_name": table_name}):
            raise HTTPException(status_code=400, detail=f"Table {table_name} already exists in dataset {datasetName}")

        # Read CSV file and convert NaN values to None
        df = pd.read_csv(io.BytesIO(file_content))
        df = df.replace({np.nan: None})

        header = df.columns.tolist()
        total_rows = len(df)

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

        # Add task to queue instead of running immediately
        task_data = {
            'user_id': user_id,
            'dataset_name': datasetName,
            'table_name': table_name,
            'dataframe': df,
            'classification': classification,
        }
        
        task_id = task_queue.add_csv_task(task_data)

        return {
            "message": "CSV table queued for processing successfully.",
            "tableName": table_name,
            "datasetName": datasetName,
            "userId": user_id,
            "taskId": task_id,
            "status": "queued",
            "fileSize": f"{file_size / (1024*1024):.1f}MB"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_error(f"Error processing CSV upload for {user_id}/{datasetName}/{table_name}", e)
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@router.get("/datasets")
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

@router.get("/datasets/{dataset_name}/tables/{table_name}")
def get_table(
    dataset_name: str,
    table_name: str,
    limit: int = Query(10),
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    search: Optional[str] = Query(None, description="Text to match in cell values"),
    columns: Optional[List[int]] = Query(None, description="Restrict search to specific column indices"),
    include_types: Optional[List[str]] = Query(None, description="Include rows with these entity types"),
    exclude_types: Optional[List[str]] = Query(None, description="Exclude rows with these entity types"),
    column: Optional[int] = Query(None, description="Column index for type filtering and confidence sorting"),
    sort_by: Optional[str] = Query(None, description="Sort by: 'confidence' or 'confidence_avg'"),
    sort_direction: str = Query("desc", description="Sort direction: 'asc' or 'desc'"),
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Get table data with bi-directional pagination.
    
    - When search is provided, performs text search across all data columns
    - When columns is provided with search, restricts search to those columns
    - When types are provided, filters by entity types (requires 'column' parameter)
    - Can sort by confidence scores at column level (requires 'column' parameter) or row level
    """
    user_id = token_payload.get("email")

    # Check dataset
    if not db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_name}
    ):
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

    # Get type information for the table
    table_types = {}
    if "column_types" in table:
        table_types = table["column_types"]
        
    # Get column classification information
    classified_columns = {}
    if "classified_columns" in table:
        classified_columns = table["classified_columns"]

    # If advanced filtering is requested, use the cell_data collection
    if search or include_types or exclude_types or sort_by == "confidence":
        # Ensure mutual exclusivity of pagination cursors
        if next_cursor and prev_cursor:
            raise HTTPException(
                status_code=400,
                detail="Only one of next_cursor or prev_cursor should be provided"
            )

        # Build cell_data query filters
        cell_filters = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
        }
        
        # Add text search functionality
        if search:
            if columns is not None and len(columns) > 0:
                # Search in specific columns
                cell_filters["col_id"] = {"$in": columns}
                cell_filters["$text"] = {"$search": search}
            else:
                # Global search across all columns
                cell_filters["$text"] = {"$search": search}

        # Add type filters if provided
        if include_types or exclude_types:
            if column is None:
                raise HTTPException(
                    status_code=400,
                    detail="Must specify 'column' parameter when filtering by types"
                )
            
            cell_filters["col_id"] = column
            
            if include_types:
                cell_filters["types"] = {"$in": include_types}
            
            if exclude_types:
                cell_filters["types"] = {"$nin": exclude_types}

        # Set up sorting
        sort_criteria = []
        if sort_by == "confidence":
            if column is None:
                raise HTTPException(
                    status_code=400,
                    detail="Must specify 'column' parameter when sorting by confidence"
                )
            
            cell_filters["col_id"] = column
            sort_criteria = [
                ("confidence", -1 if sort_direction == "desc" else 1),
                ("row_id", 1)  # Secondary sort for stable pagination
            ]
        else:
            sort_criteria = [("row_id", 1)]

        # Handle pagination with cursors
        if next_cursor:
            try:
                cursor_data = json.loads(base64.b64decode(next_cursor).decode('utf-8'))
                if sort_by == "confidence":
                    last_confidence = cursor_data.get("confidence")
                    last_row_id = cursor_data.get("row_id")
                    if sort_direction == "desc":
                        cell_filters["$or"] = [
                            {"confidence": {"$lt": last_confidence}},
                            {"confidence": last_confidence, "row_id": {"$gt": last_row_id}}
                        ]
                    else:
                        cell_filters["$or"] = [
                            {"confidence": {"$gt": last_confidence}},
                            {"confidence": last_confidence, "row_id": {"$gt": last_row_id}}
                        ]
                else:
                    cell_filters["row_id"] = {"$gt": cursor_data.get("row_id")}
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid next_cursor format")

        # Execute cell_data query to get matching row_ids
        if sort_by == "confidence":
            # For confidence sorting, we work directly with cells and then map to rows
            cell_query = db.cell_data.find(cell_filters).sort([("confidence", -1 if sort_direction == "desc" else 1)]).limit(limit + 1)
            cell_results = list(cell_query)
            
            # Determine if there are more results
            has_more = len(cell_results) > limit
            if has_more:
                cell_results = cell_results[:limit]

            # Extract row_ids (maintaining the confidence sort order)
            matching_row_ids = [result["row_id"] for result in cell_results]
            
        else:
            # For other cases, use the aggregation pipeline to group by row_id
            cell_pipeline = [
                {"$match": cell_filters},
                {"$sort": dict(sort_criteria)},
                {"$group": {
                    "_id": "$row_id",
                }},
                {"$sort": {"_id": 1}},  # Sort by row_id for consistent ordering
                {"$limit": limit + 1}
            ]
            
            cell_results = list(db.cell_data.aggregate(cell_pipeline))
            
            # Determine if there are more results
            has_more = len(cell_results) > limit
            if has_more:
                cell_results = cell_results[:limit]

            # Extract row_ids
            matching_row_ids = [result["_id"] for result in cell_results]
        
        if not matching_row_ids:
            # No matching cells found
            return {
                "data": {
                    "datasetName": dataset_name,
                    "tableName": table_name,
                    "status": "DONE",
                    "header": header,
                    "rows": [],
                    "total_matches": 0,
                    "column_types": table_types,
                    "classified_columns": classified_columns,
                },
                "pagination": {"next_cursor": None, "prev_cursor": None},
            }

        # Fetch full row data from input_data using the matching row_ids
        row_filter = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": {"$in": matching_row_ids}
        }
        
        # For confidence sorting, we need to preserve the order from cell_data
        if sort_by == "confidence":
            # Create a map for row ordering based on confidence sort
            row_order_map = {row_id: idx for idx, row_id in enumerate(matching_row_ids)}
            
            raw_rows = list(db.input_data.find(row_filter))
            if not raw_rows:
                raw_rows = list(crocodile_db.input_data.find(row_filter))
            
            # Sort rows according to the confidence order
            raw_rows.sort(key=lambda row: row_order_map.get(row["row_id"], float('inf')))
        else:
            raw_rows = list(db.input_data.find(row_filter).sort("row_id", 1))
            if not raw_rows:
                raw_rows = list(crocodile_db.input_data.find(row_filter).sort("row_id", 1))

        # Calculate pagination cursors
        new_next_cursor = new_prev_cursor = None
        
        if len(raw_rows) > 0:
            if has_more:
                if sort_by == "confidence":
                    # For confidence sorting, get the last confidence value
                    last_row = raw_rows[-1]
                    last_confidence = next(
                        (r["confidence"] for r in cell_results if r["row_id"] == last_row["row_id"]),
                        0.0
                    )
                    cursor_data = {"confidence": last_confidence, "row_id": last_row["row_id"]}
                else:
                    cursor_data = {"row_id": raw_rows[-1]["row_id"]}
                new_next_cursor = base64.b64encode(json.dumps(cursor_data).encode('utf-8')).decode('utf-8')
            
            # For confidence sorting, we don't support backward pagination
            if sort_by != "confidence":
                # Check if we're on the first page for other sorting types
                first_check_filter = {
                    "user_id": user_id,
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                }
                
                # Add the same filters that were applied to cell_data
                if search:
                    if columns is not None and len(columns) > 0:
                        first_check_filter["col_id"] = {"$in": columns}
                        first_check_filter["$text"] = {"$search": search}
                    else:
                        first_check_filter["$text"] = {"$search": search}
                
                if include_types or exclude_types:
                    first_check_filter["col_id"] = column
                    if include_types:
                        first_check_filter["types"] = {"$in": include_types}
                    if exclude_types:
                        first_check_filter["types"] = {"$nin": exclude_types}
                
                # Check if there are cells before the current first row
                first_check_filter["row_id"] = {"$lt": raw_rows[0]["row_id"]}
                
                if db.cell_data.count_documents(first_check_filter) > 0:
                    new_prev_cursor = base64.b64encode(json.dumps({"row_id": raw_rows[0]["row_id"]}).encode('utf-8')).decode('utf-8')

        # Format rows for response
        rows_formatted = []
        for row in raw_rows:
            linked_entities = []
            el_results = row.get("el_results", {})

            for col_index in range(len(header)):
                candidates = el_results.get(str(col_index), [])
                if candidates:
                    sanitized_candidates = sanitize_for_json(candidates)
                    linked_entities.append({"idColumn": col_index, "candidates": sanitized_candidates})

            rows_formatted.append(
                {
                    "idRow": row.get("row_id"),
                    "data": sanitize_for_json(row.get("data", [])),
                    "linked_entities": linked_entities,
                }
            )

        # Determine table status
        table_status_filter = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "$or": [
                {"status": {"$in": ["TODO", "DOING"]}},
                {"ml_status": {"$in": ["TODO", "DOING"]}},
            ],
        }

        pending_docs_count = db.input_data.count_documents(table_status_filter)
        if pending_docs_count == 0:
            pending_docs_count = crocodile_db.input_data.count_documents(table_status_filter)

        status = "DOING" if pending_docs_count > 0 else "DONE"

        return {
            "data": {
                "datasetName": dataset_name,
                "tableName": table_name,
                "status": status,
                "header": header,
                "rows": rows_formatted,
                "total_matches": len(matching_row_ids),
                "column_types": table_types,
                "classified_columns": classified_columns,
            },
            "pagination": {"next_cursor": new_next_cursor, "prev_cursor": new_prev_cursor},
        }
    
    # Handle row-level average confidence sorting
    elif sort_by == "confidence_avg":
        # Use input_data collection for row-level confidence sorting
        query_filter = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
        }
        
        # Ensure mutual exclusivity of pagination cursors
        if next_cursor and prev_cursor:
            raise HTTPException(
                status_code=400,
                detail="Only one of next_cursor or prev_cursor should be provided"
            )

        # Set up sorting criteria
        sort_criteria = [
            ("avg_confidence", -1 if sort_direction == "desc" else 1),
            ("_id", 1)  # Secondary sort for stable pagination
        ]

        # Handle pagination with cursors
        if next_cursor:
            try:
                cursor_data = json.loads(base64.b64decode(next_cursor).decode('utf-8'))
                last_avg_confidence = cursor_data.get("avg_confidence")
                last_id = cursor_data.get("_id")
                
                if sort_direction == "desc":
                    query_filter["$or"] = [
                        {"avg_confidence": {"$lt": last_avg_confidence}},
                        {"avg_confidence": last_avg_confidence, "_id": {"$gt": ObjectId(last_id)}}
                    ]
                else:
                    query_filter["$or"] = [
                        {"avg_confidence": {"$gt": last_avg_confidence}},
                        {"avg_confidence": last_avg_confidence, "_id": {"$gt": ObjectId(last_id)}}
                    ]
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid next_cursor format")

        # Execute query
        results = db.input_data.find(query_filter).sort(sort_criteria).limit(limit + 1)
        raw_rows = list(results)

        if not raw_rows:
            results = crocodile_db.input_data.find(query_filter).sort(sort_criteria).limit(limit + 1)
            raw_rows = list(results)

        # Determine if there are more results
        has_more = len(raw_rows) > limit
        if has_more:
            raw_rows = raw_rows[:limit]

        # Calculate pagination cursors
        new_next_cursor = new_prev_cursor = None
        
        if len(raw_rows) > 0 and has_more:
            last_row = raw_rows[-1]
            cursor_data = {
                "avg_confidence": last_row.get("avg_confidence", 0.0),
                "_id": str(last_row["_id"])
            }
            new_next_cursor = base64.b64encode(json.dumps(cursor_data).encode('utf-8')).decode('utf-8')

        # Format rows for response
        rows_formatted = []
        for row in raw_rows:
            linked_entities = []
            el_results = row.get("el_results", {})

            for col_index in range(len(header)):
                candidates = el_results.get(str(col_index), [])
                if candidates:
                    sanitized_candidates = sanitize_for_json(candidates)
                    linked_entities.append({"idColumn": col_index, "candidates": sanitized_candidates})

            rows_formatted.append(
                {
                    "idRow": row.get("row_id"),
                    "data": sanitize_for_json(row.get("data", [])),
                    "linked_entities": linked_entities,
                }
            )

        # Determine table status
        table_status_filter = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "$or": [
                {"status": {"$in": ["TODO", "DOING"]}},
                {"ml_status": {"$in": ["TODO", "DOING"]}},
            ],
        }

        pending_docs_count = db.input_data.count_documents(table_status_filter)
        if pending_docs_count == 0:
            pending_docs_count = crocodile_db.input_data.count_documents(table_status_filter)

        status = "DOING" if pending_docs_count > 0 else "DONE"

        return {
            "data": {
                "datasetName": dataset_name,
                "tableName": table_name,
                "status": status,
                "header": header,
                "rows": rows_formatted,
                "total_matches": len(raw_rows),
                "column_types": table_types,
                "classified_columns": classified_columns,
            },
            "pagination": {"next_cursor": new_next_cursor, "prev_cursor": new_prev_cursor},
        }

    # Standard query when no advanced filtering is needed
    else:
        query_filter = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
        }
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

        results = db.input_data.find(query_filter).sort("row_id", sort_direction).limit(limit + 1)
        raw_rows = list(results)

        if not raw_rows:
            results = (
                crocodile_db.input_data.find(query_filter)
                .sort("row_id", sort_direction)
                .limit(limit + 1)
            )
            raw_rows = list(results)

        has_more = len(raw_rows) > limit
        if has_more:
            raw_rows = raw_rows[:limit]

        if prev_cursor:
            raw_rows.reverse()

        next_cursor = None
        prev_cursor = None

        if raw_rows:
            if not query_filter.get("_id", {}).get("$gt"):
                if prev_cursor:
                    prev_cursor = str(raw_rows[0]["_id"])
                else:
                    first_id = raw_rows[0]["_id"]
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
            else:
                prev_cursor = str(raw_rows[0]["_id"])

            if has_more:
                next_cursor = str(raw_rows[-1]["_id"])
            elif query_filter.get("_id", {}).get("$lt"):
                next_cursor = str(raw_rows[-1]["_id"])

        table_status_filter = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "$or": [
                {"status": {"$in": ["TODO", "DOING"]}},
                {"ml_status": {"$in": ["TODO", "DOING"]}},
            ],
        }

        pending_docs_count = db.input_data.count_documents(table_status_filter)
        if pending_docs_count == 0:
            pending_docs_count = crocodile_db.input_data.count_documents(table_status_filter)

        status = "DOING" if pending_docs_count > 0 else "DONE"

        rows_formatted = []
        for row in raw_rows:
            linked_entities = []
            el_results = row.get("el_results", {})

            for col_index in range(len(header)):
                candidates = el_results.get(str(col_index), [])
                if candidates:
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
                "column_types": table_types,
                "classified_columns": classified_columns,
            },
            "pagination": {"next_cursor": next_cursor, "prev_cursor": prev_cursor},
        }

        response_data = sanitize_for_json(response_data)

        return response_data

@router.post("/datasets", status_code=status.HTTP_201_CREATED)
def create_dataset(
    dataset_data: dict = Body(..., example={"dataset_name": "test"}),  # updated example key
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
):
    """
    Create a new dataset.
    """
    user_id = token_payload.get("email")

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
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Delete a dataset by name and all its data from MongoDB and Elasticsearch.
    """
    user_id = token_payload.get("email")

    # Check existence using uniform dataset key
    existing = db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_name}
    )
    if not existing:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Delete all tables associated with this dataset
    db.tables.delete_many({"user_id": user_id, "dataset_name": dataset_name})

    # Delete dataset metadata
    db.datasets.delete_one({"user_id": user_id, "dataset_name": dataset_name})

    # Delete all rows associated with this dataset
    db.input_data.delete_many({"user_id": user_id, "dataset_name": dataset_name})

    # Delete data from crocodile_db
    crocodile_db.input_data.delete_many({"client_id": user_id, "dataset_name": dataset_name})

    return None

@router.delete("/datasets/{dataset_name}/tables/{table_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_table(
    dataset_name: str,
    table_name: str,
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Delete a table by name within a dataset and remove all its data from MongoDB and Elasticsearch.
    """
    user_id = token_payload.get("email")

    # Ensure dataset exists
    dataset = db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_name}
    )
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Ensure table exists
    table = db.tables.find_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    )
    if not table:
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    row_count = table.get("total_rows", 0)

    # Delete table metadata
    db.tables.delete_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    )

    # Update dataset metadata
    db.datasets.update_one(
        {"user_id": user_id, "dataset_name": dataset_name},
        {"$inc": {"total_tables": -1, "total_rows": -row_count}},
    )

    # Delete all rows associated with this table
    db.input_data.delete_many(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    )

    # Delete data from crocodile_db
    crocodile_db.input_data.delete_many(
        {"client_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
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

    if not column_candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No entity linking candidates found for column {column_id} in row {row_id}",
        )

    # Check if the entity exists in the candidates
    entity_found = any(candidate.get("id") == annotation.entity_id for candidate in column_candidates)

    updated_candidates = []

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

@router.delete("/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}/candidates/{entity_id}")
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
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id,
            "column_id": column_id,
            "entity_id": entity_id,
            "remaining_candidates": len(updated_candidates),
        }
    )

def sort_rows_by_confidence(
    rows: List[Dict], 
    sort_by: Optional[str] = None,
    sort_column: Optional[int] = None,
    sort_direction: str = "desc",
    header: List[str] = None
) -> List[Dict]:
    """
    Sort rows by confidence scores.
    """
    if not sort_by or sort_by != "confidence" or not rows:
        return rows  # No sorting if not confidence

    reverse = sort_direction == "desc"  # True for descending order
    
    if sort_column is not None and header:
        # Column-level sorting: sort by the confidence of a specific column
        def get_column_confidence(row):
            # Get confidence score from the first candidate in the specified column
            el_results = row.get("el_results", {})
            candidates = el_results.get(str(sort_column), [])
            if candidates and len(candidates) > 0:
                return candidates[0].get("score", 0.0) or 0.0
            return 0.0
        
        return sorted(rows, key=get_column_confidence, reverse=reverse)
    else:
        # Row-level sorting: sort by the average confidence across all columns
        def get_avg_confidence(row):
            # Calculate average confidence score across all columns with candidates
            el_results = row.get("el_results", {})
            total_score = 0.0
            count = 0
            
            # Only consider valid column indices
            if header:
                col_range = range(len(header))
            else:
                # If no header, try to determine columns from data
                data = row.get("data", [])
                col_range = range(len(data))
                
            for col_idx in col_range:
                candidates = el_results.get(str(col_idx), [])
                if candidates and len(candidates) > 0:
                    score = candidates[0].get("score", 0.0)
                    if score is not None:  # Skip None scores
                        total_score += score
                        count += 1
            
            # Avoid division by zero
            return total_score / count if count > 0 else 0.0
        
        return sorted(rows, key=get_avg_confidence, reverse=reverse)

async def get_table_status_stream(
    dataset_name: str,
    table_name: str,
    token_payload: str,
):
    """
    Get the current status of a specific table (streaming).
    Provides distinct progress for the prediction and ML phases with a 'phase' attribute.
    """
    user_id = token_payload.get("email")
    client = None  # Initialize client to None
    try:
        # Create a new client specifically for this stream
        client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        db = client["crocodile_backend_db"]
        crocodile_db = client["crocodile_db"]
        
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

        total_rows = table.get("total_rows", 0)
        
        # Initial response with current state
        stored_completion = table.get("completion_percentage")
        last_synced = table.get("last_synced")
        if last_synced:
            last_synced = last_synced.isoformat()

        while True:  # Loop until processing is complete
            # Check both databases for pending documents in the prediction phase
            prediction_filter = {
                "user_id": user_id,
                "dataset_name": dataset_name,
                "table_name": table_name,
                "status": {"$in": ["TODO", "DOING"]},
            }
            pending_prediction_count = db.input_data.count_documents(prediction_filter)
            if pending_prediction_count == 0:
                pending_prediction_count = crocodile_db.input_data.count_documents(prediction_filter)

            # Check both databases for pending documents in the ML phase
            ml_filter = {
                "user_id": user_id,
                "dataset_name": dataset_name,
                "table_name": table_name,
                "ml_status": {"$in": ["TODO", "DOING"]},
            }
            pending_ml_count = db.input_data.count_documents(ml_filter)
            if pending_ml_count == 0:
                pending_ml_count = crocodile_db.input_data.count_documents(ml_filter)

            # Calculate completion percentages
            prediction_completion = 100 - ((pending_prediction_count / total_rows) * 100) if total_rows > 0 else 100
            ml_completion = 100 - ((pending_ml_count / total_rows) * 100) if total_rows > 0 else 100

            # Determine overall phase and status
            if pending_prediction_count > 0:
                phase = "PREDICTION"
                status = "PROCESSING"
            elif pending_ml_count > 0:
                phase = "ML_PREDICTION"
                status = "PROCESSING"
            else:
                phase = "DONE"
                status = "DONE"

            # Send status update
            response = {
                "dataset_name": dataset_name,
                "table_name": table_name,
                "status": status,
                "phase": phase,
                "total_rows": total_rows,
                "pending_rows": pending_prediction_count if phase == "PREDICTION" else pending_ml_count,
                "completed_rows": total_rows - (pending_prediction_count if phase == "PREDICTION" else pending_ml_count),
                "completion_percentage": round(prediction_completion if phase == "PREDICTION" else ml_completion, 2),
                "last_synced": last_synced,
            }
            
            # Include error if present
            if "error" in table:
                response["error"] = table["error"]
                
            yield f"data: {json.dumps(response)}\n\n"

            # If processing is complete, exit the stream
            if status == "DONE":
                break
                
            # Wait before checking again
            await asyncio.sleep(1)

    except Exception as e:
        yield f"data: {json.dumps({'status': 'ERROR', 'detail': str(e)})}\n\n"
    finally:
        if client:
            client.close()

@router.get("/datasets/{dataset_name}/tables/{table_name}/status")
async def stream_table_status(
    dataset_name: str,
    table_name: str,
    token_payload: str = Depends(verify_token),
):
    """
    Stream the current status of a specific table with real-time updates.
    Returns Server-Sent Events (SSE) with status information.
    """
    return StreamingResponse(
        get_table_status_stream(dataset_name, table_name, token_payload),
        media_type="text/event-stream",
    )

@router.get("/datasets/{dataset_name}/tables/{table_name}/export")
def export_enriched_csv(
    dataset_name: str,
    table_name: str,
    token_payload: str = Depends(verify_token),
    fields: List[str] = Query(default=["id", "name"], description="Fields to include for enriched columns"),
):
    """
    Stream an enriched CSV file combining the original data and the selected fields
    of the top candidate for each NE‐classified column.
    """
    user_id = token_payload.get("email")
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    input_data = client["crocodile_backend_db"]["input_data"]
    tables = client["crocodile_backend_db"]["tables"]

    table_doc = tables.find_one({
        "user_id": user_id,
        "dataset_name": dataset_name,
        "table_name": table_name
    })

    if not table_doc:
        raise HTTPException(status_code=404, detail="Table not found")

    # validate requested fields
    allowed = {"id", "name", "description", "types", "score"}
    if any(f not in allowed for f in fields):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid field requested. Allowed fields: {allowed}"
        )

    # only NE‐classified columns
    ne_cols = sorted(int(idx) for idx in table_doc.get("classified_columns", {}).get("NE", {}))
    header = table_doc.get("header", [])

    def csv_generator() -> Generator[bytes, None, None]:
        buffer = io.StringIO()
        writer = None
        cursor = input_data.find(
            {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
        ).sort("row_id", 1)

        for doc in cursor:
            data = doc.get("data", [])
            enriched = []
            for idx in ne_cols:
                top = (doc.get("el_results", {}).get(str(idx), []) or [{}])[0]
                for fld in fields:
                    if fld == "types":
                        enriched.append(json.dumps(top.get("types", [])))
                    else:
                        enriched.append(top.get(fld, ""))

            full_row = data + enriched

            if writer is None:
                # build header: original + one per NE‐col per field
                hdr_extra = [f"{header[idx]}_{fld}" for idx in ne_cols for fld in fields]
                full_header = header + hdr_extra
                writer = csv.writer(buffer)
                writer.writerow(full_header)
                yield buffer.getvalue().encode()
                buffer.seek(0); buffer.truncate(0)

            writer.writerow(full_row)
            yield buffer.getvalue().encode()
            buffer.seek(0); buffer.truncate(0)

    filename = f"{dataset_name}_{table_name}_enriched.csv"
    return StreamingResponse(
        csv_generator(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )