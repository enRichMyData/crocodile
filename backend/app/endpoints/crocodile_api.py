import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Union
import base64
import asyncio

import numpy as np
import pandas as pd
from bson import ObjectId
from dependencies import get_crocodile_db, get_db, verify_token, es, ES_INDEX
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
from services.utils import sanitize_for_json

# Import services and utilities
from crocodile import Crocodile

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
            # Wait a moment before starting sync to allow initial processing
            time.sleep(5)

            # Create a result sync service and sync results
            mongo_uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
            sync_service = ResultSyncService(mongo_uri=mongo_uri)
            sync_service.sync_results(
                user_id=user_id, dataset_name=datasetName, table_name=table_upload.table_name
            )

        # Add both tasks to background processing
        background_tasks.add_task(run_crocodile_task)
        background_tasks.add_task(sync_results_task)

        return {
            "message": "Table added successfully.",
            "tableName": table_upload.table_name,
            "datasetName": datasetName,
            "userId": user_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
):
    """
    Add a new table from CSV file to an existing dataset and trigger Crocodile processing.
    """
    user_id = token_payload.get("email")

    try:
        # Read CSV file and convert NaN values to None
        df = pd.read_csv(file.file)
        df = df.replace({np.nan: None})  # permanent fix for JSON serialization

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

        return {
            "message": "CSV table added successfully.",
            "tableName": table_name,
            "datasetName": datasetName,
            "userId": user_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
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
    column: Optional[int] = Query(None, description="Column index to use for search, types filtering or sorting"),
    search_columns: Optional[List[int]] = Query(None, description="Restrict search to specific column indices"),
    include_types: Optional[List[str]] = Query(None, description="Include rows with these entity types"),
    exclude_types: Optional[List[str]] = Query(None, description="Exclude rows with these entity types"),
    sort_by: Optional[str] = Query(None, description="Sort by: 'confidence' or 'confidence_avg'"),
    sort_direction: str = Query("desc", description="Sort direction: 'asc' or 'desc'"),
    token_payload: str = Depends(verify_token),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Get table data with bi-directional pagination.
    
    - When search is provided, uses Elasticsearch for text search
    - When types are provided, filters by entity types
    - Can sort by confidence scores at column or row level
    - Sorting options:
      - 'confidence': Sort by confidence score of a specific column (requires 'column' parameter)
      - 'confidence_avg': Sort by average confidence across all columns in each row
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

    # Get type information for the table
    table_types = {}
    if "column_types" in table:
        table_types = table["column_types"]
        
    # Get column classification information
    classified_columns = {}
    if "classified_columns" in table:
        classified_columns = table["classified_columns"]

    # If search, type filters or confidence sorting is requested, use Elasticsearch
    if search is not None or include_types or exclude_types or sort_by is not None:
        # Ensure mutual exclusivity of pagination cursors
        if next_cursor and prev_cursor:
            raise HTTPException(
                status_code=400,
                detail="Only one of next_cursor or prev_cursor should be provided"
            )

        # Parse cursors if provided
        search_after = None
        search_before = None
        is_backward = False
        
        if next_cursor:
            try:
                # Try to decode the cursor as JSON first
                try:
                    cursor_data = json.loads(base64.b64decode(next_cursor).decode('utf-8'))
                    search_after = cursor_data.get("sort")
                except (json.JSONDecodeError, ValueError):
                    # For backward compatibility, try parsing as a simple row_id
                    row_id = int(next_cursor)
                    search_after = [row_id]  # Default to row_id only
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid next_cursor format")
        
        elif prev_cursor:
            is_backward = True
            try:
                # Try to decode the cursor as JSON first
                try:
                    cursor_data = json.loads(base64.b64decode(prev_cursor).decode('utf-8'))
                    search_before = cursor_data.get("sort")
                except (json.JSONDecodeError, ValueError):
                    # For backward compatibility, try parsing as a simple row_id
                    row_id = int(prev_cursor)
                    search_before = [row_id]  # Default to row_id only
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid prev_cursor format")
            
        # Build base query filters
        filters = [
            {"term": {"user_id": user_id}},
            {"term": {"dataset_name": dataset_name}},
            {"term": {"table_name": table_name}},
        ]
        
        # Add text search filter if provided
        if search:
            if search_columns:
                # Search in specified columns only (OR condition across columns)
                nested_queries = []
                for col_idx in search_columns:
                    nested_queries.append({
                        "bool": {
                            "must": [
                                {"match": {"data.value": search}},
                                {"term": {"data.col_index": col_idx}}
                            ]
                        }
                    })
                
                filters.append({
                    "nested": {
                        "path": "data",
                        "query": {"bool": {"should": nested_queries, "minimum_should_match": 1}}
                    }
                })
            elif column is not None:
                # Search in specified column
                filters.append({
                    "nested": {
                        "path": "data",
                        "query": {
                            "bool": {
                                "must": [
                                    {"match": {"data.value": search}},
                                    {"term": {"data.col_index": column}}
                                ]
                            }
                        }
                    }
                })
            else:
                # Search in all columns
                filters.append({
                    "nested": {
                        "path": "data",
                        "query": {"bool": {"must": [{"match": {"data.value": search}}]}}
                    }
                })
        
        # Add type filters if provided - must specify a column
        if (include_types or exclude_types):
            type_column = column  # Use the unified column parameter
            
            if type_column is None:
                raise HTTPException(
                    status_code=400,
                    detail="Must specify 'column' parameter when filtering by types"
                )
                
            # Include specific types (ANY of the specified types must match)
            if include_types:
                type_filter_query = {
                    "nested": {
                        "path": "data",
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"data.col_index": type_column}},
                                    {"bool": {"should": [{"term": {"data.types": type_name}} for type_name in include_types]}}
                                ]
                            }
                        }
                    }
                }
                filters.append(type_filter_query)
            
            # Exclude specific types (NONE of the specified types should match)
            if exclude_types:
                for type_name in exclude_types:
                    exclude_filter = {
                        "bool": {
                            "must_not": {
                                "nested": {
                                    "path": "data",
                                    "query": {
                                        "bool": {
                                            "must": [
                                                {"term": {"data.col_index": type_column}},
                                                {"term": {"data.types": type_name}}
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    }
                    filters.append(exclude_filter)
        
        # Build sort criteria
        sort_criteria = []
        
        # Set up sorting based on confidence if requested
        if sort_by == "confidence":
            # Column-level confidence sort - requires column parameter
            if column is None:
                raise HTTPException(
                    status_code=400,
                    detail="Must specify 'column' parameter when sorting by column confidence"
                )
                
            # Use the working format for nested sort
            sort_criteria.append({
                "data.confidence": {
                    "order": "asc" if is_backward else sort_direction,
                    "nested": {
                        "path": "data",
                        "filter": {"term": {"data.col_index": column}}
                    },
                    "missing": "_last" if sort_direction == "desc" else "_first"
                }
            })
        elif sort_by == "confidence_avg":
            # Row-level average confidence sort
            sort_criteria.append({
                "avg_confidence": {
                    "order": "asc" if is_backward else sort_direction,
                    "missing": "_last" if sort_direction == "desc" else "_first"
                }
            })
        
        # Always add row_id as secondary sort for stable pagination
        sort_criteria.append({
            "row_id": {
                "order": "asc" if is_backward else "asc"  # always ascending for row_id
            }
        })
            
        # Prepare the search body
        body = {
            "query": {"bool": {"filter": filters}},
            "sort": sort_criteria,
            "_source": ["row_id", "avg_confidence"],  # Include avg_confidence in results
            "size": limit + 1,      # Get one extra to check for more results
            "track_total_hits": True  # Get accurate total hit count
        }
        
        # Add search_after for forward pagination
        if search_after:
            body["search_after"] = search_after
        
        # For backward pagination, we need to reverse the sort order temporarily
        # and then reverse the results afterward
        if is_backward and search_before:
            # Reverse sort orders for backward pagination
            for sort_item in body["sort"]:
                for key in sort_item:
                    if isinstance(sort_item[key], dict) and "order" in sort_item[key]:
                        sort_item[key]["order"] = "desc" if sort_item[key]["order"] == "asc" else "asc"
            
            body["search_after"] = search_before
        
        # Execute search and get results
        res = es.search(index=ES_INDEX, body=body)
        hits = res["hits"]["hits"]
        total_hits = res["hits"]["total"]["value"]
        
        # Determine if there are more results
        has_more = len(hits) > limit
        if has_more:
            hits = hits[:limit]  # Remove the extra item
            
        # If we did backward pagination, we need to reverse the results
        # to maintain consistent ordering for the user
        if is_backward:
            hits.reverse()
            
        # Calculate next_cursor and prev_cursor for pagination
        new_next_cursor = new_prev_cursor = None
        
        # Helper function to create encoded cursor
        def create_cursor(sort_values: List[Any]) -> str:
            cursor_data = {"sort": sort_values}
            return base64.b64encode(json.dumps(cursor_data).encode('utf-8')).decode('utf-8')
            
        if len(hits) > 0:
            # For next_cursor:
            # - If we have more results after this page, create next_cursor
            # - If we came from prev_cursor, create next_cursor unless we're on first page
            if has_more:
                last_hit = hits[-1]
                new_next_cursor = create_cursor(last_hit["sort"])
            elif is_backward:
                # Coming backward and no more results means we're approaching first page
                # Need to check if this is actually the first page
                first_page_check = {
                    "query": {
                        "bool": {"filter": filters}
                    },
                    "sort": sort_criteria,
                    "size": 1,
                    "_source": False
                }
                
                # Reverse sort directions back to normal
                for sort_item in first_page_check["sort"]:
                    for key in sort_item:
                        if isinstance(sort_item[key], dict) and "order" in sort_item[key]:
                            sort_item[key]["order"] = "desc" if sort_item[key]["order"] == "asc" else "asc"
                
                # Check if there are any documents before this "batch"
                first_hit = hits[0]
                first_page_check["search_after"] = first_hit["sort"]
                
                check_res = es.search(index=ES_INDEX, body=first_page_check)
                if len(check_res["hits"]["hits"]) > 0:
                    # There are results after the first item in our current batch
                    # So we're not on the first page
                    new_next_cursor = create_cursor(first_hit["sort"])
            
            # For prev_cursor:
            # - If we're not on the first page, create prev_cursor
            if len(hits) > 0:
                first_hit = hits[0]
                
                # Check if we're on the first page
                first_page_check = {
                    "query": {
                        "bool": {"filter": filters}
                    },
                    "sort": [{key: {"order": "desc" if item[key]["order"] == "asc" else "asc"} 
                              if isinstance(item[key], dict) and "order" in item[key] else item[key] 
                              for key in item} 
                            for item in sort_criteria],
                    "size": 1,
                    "_source": False
                }
                
                # If we have a first hit, check if anything comes before it
                try:
                    first_page_check["search_after"] = first_hit["sort"]
                    check_res = es.search(index=ES_INDEX, body=first_page_check)
                    if len(check_res["hits"]["hits"]) > 0:
                        new_prev_cursor = create_cursor(first_hit["sort"])
                except Exception as e:
                    # Log the error but don't fail the request
                    print(f"Error checking for previous page: {str(e)}")
                
                # If we came via next_cursor, we definitely have a previous page
                if next_cursor:
                    new_prev_cursor = create_cursor(first_hit["sort"])
        
        # Extract row IDs from search hits
        row_ids = [hit["_source"]["row_id"] for hit in hits]
        
        # Fetch full data from MongoDB using the row_ids
        mongo_query = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": {"$in": row_ids}
        }
        
        # Try backend DB first, then crocodile DB if needed
        raw_rows = list(db.input_data.find(mongo_query))
        if not raw_rows:
            raw_rows = list(crocodile_db.input_data.find(mongo_query))
            
        # Sort rows to match the order from ES results
        row_id_to_idx = {row_id: i for i, row_id in enumerate(row_ids)}
        raw_rows.sort(key=lambda row: row_id_to_idx.get(row.get("row_id"), float('inf')))

        # Format rows
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

        status = "DOING"
        if pending_docs_count == 0:
            status = "DONE"

        return {
            "data": {
                "datasetName": dataset_name,
                "tableName": table_name,
                "status": status,
                "header": header,
                "rows": rows_formatted,
                "total_matches": total_hits,
                "column_types": table_types,
                "classified_columns": classified_columns,
            },
            "pagination": {"next_cursor": new_next_cursor, "prev_cursor": new_prev_cursor},
        }

    # MongoDB-only logic when no search/type filters or special sorting provided
    else:
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

        status = "DOING"
        if pending_docs_count == 0:
            status = "DONE"

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

    # Delete data from Elasticsearch
    try:
        es.delete_by_query(
            index=ES_INDEX,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"user_id": user_id}},
                            {"term": {"dataset_name": dataset_name}}
                        ]
                    }
                }
            },
            refresh=True  # Refresh the index immediately
        )
    except Exception as e:
        # Log error but don't fail the operation if ES deletion fails
        print(f"Error deleting dataset from Elasticsearch: {str(e)}")

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

    # Delete data from Elasticsearch
    try:
        es.delete_by_query(
            index=ES_INDEX,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"user_id": user_id}},
                            {"term": {"dataset_name": dataset_name}},
                            {"term": {"table_name": table_name}}
                        ]
                    }
                }
            },
            refresh=True  # Refresh the index immediately
        )
    except Exception as e:
        # Log error but don't fail the operation if ES deletion fails
        print(f"Error deleting table from Elasticsearch: {str(e)}")

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
    Manages its own DB connection for the stream duration.
    """
    user_id = token_payload.get("email")
    client = None  # Initialize client to None
    try:
        # Create a new client specifically for this stream
        client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        db = client["crocodile_backend_db"]
        crocodile_db = client["crocodile_db"]
        crocodile_backend_db = client["crocodile_backend_db"]

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

        total_rows = table.get("total_rows", 0)
        
        # Initial response with current state
        stored_completion = table.get("completion_percentage")
        last_synced = table.get("last_synced")
        if last_synced:
            last_synced = last_synced.isoformat()

        while True:  # Loop until processing is complete
            # Check both databases for pending documents
            pending_docs_count = db.input_data.count_documents(table_status_filter)
            if pending_docs_count == 0:
                pending_docs_count = crocodile_db.input_data.count_documents(table_status_filter)

            # Calculate completion percentage
            if total_rows == 0:
                completion_percentage = 100  # If no rows, consider it complete
            else:
                remaining_percentage = (pending_docs_count / total_rows) * 100
                completion_percentage = 100 - remaining_percentage

            # If the table record has a completion percentage and we're not fully done,
            # use the stored value as it's likely more accurate (from the sync process)
            if stored_completion is not None and pending_docs_count > 0:
                completion_percentage = stored_completion

            # Determine overall status
            if pending_docs_count == 0:
                status = "DONE"
            else:
                status = "PROCESSING"

            # Send status update
            response = {
                "dataset_name": dataset_name,
                "table_name": table_name,
                "status": status,
                "total_rows": total_rows,
                "pending_rows": pending_docs_count,
                "completed_rows": total_rows - pending_docs_count,
                "completion_percentage": round(completion_percentage, 2),
                "last_synced": last_synced,
            }
            
            # Include error if present
            if "error" in table:
                response["error"] = table["error"]
                
            yield f"data: {json.dumps(response)}\n\n"

            # If processing is complete, exit the stream
            if status == "DONE":
                crocodile_backend_db.tables.update_one(
                    {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name},
                    {"$set": {"status": "DONE", "completion_percentage": 100, "last_synced": datetime.now()}}
                )
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
