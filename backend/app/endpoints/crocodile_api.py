import json
import os
from datetime import datetime
from typing import Dict, List, Optional

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


@router.post("/datasets/{datasetName}/tables/json", status_code=status.HTTP_201_CREATED)
def add_table(
    datasetName: str,
    table_upload: TableUpload = Body(..., example=IMDB_EXAMPLE),
    background_tasks: BackgroundTasks = None,
    db: Database = Depends(get_db),
):
    """
    Add a new table to an existing dataset and trigger Crocodile processing in the background.
    """
    # Check if dataset exists; if not, create it
    dataset = db.datasets.find_one({"dataset_name": datasetName})  # updated query key
    if not dataset:
        try:
            dataset_id = db.datasets.insert_one(
                {
                    "dataset_name": datasetName,  # updated field key
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
        "dataset_name": datasetName,
        "table_name": table_upload.table_name,
        "header": table_upload.header,
        "total_rows": table_upload.total_rows,
        "created_at": datetime.now(),
        "status": "processing",
        "classified_columns": classification,  # added classification field
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

    # Trigger background task with classification passed to Crocodile
    def run_crocodile_task():
        croco = Crocodile(
            input_csv=pd.DataFrame(table_upload.data),
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

    background_tasks.add_task(run_crocodile_task)

    return {
        "message": "Table added successfully.",
        "tableName": table_upload.table_name,
        "datasetName": datasetName,
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
    column_classification: Optional[dict] = Depends(parse_json_column_classification),  # SON/dict
    background_tasks: BackgroundTasks = None,
    db: Database = Depends(get_db),
):
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
    dataset = db.datasets.find_one({"dataset_name": datasetName})  # updated query key
    if not dataset:
        try:
            dataset_id = db.datasets.insert_one(
                {
                    "dataset_name": datasetName,  # updated field key
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
        "dataset_name": datasetName,
        "table_name": table_name,
        "header": header,
        "total_rows": total_rows,
        "created_at": datetime.now(),
        "classified_columns": classification,  # updated field for CSV input
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

    # Trigger background task with columns_type passed to Crocodile
    def run_crocodile_task():
        croco = Crocodile(
            input_csv=df,
            dataset_name=datasetName,
            table_name=table_name,
            entity_retrieval_endpoint=os.environ.get("ENTITY_RETRIEVAL_ENDPOINT"),
            entity_retrieval_token=os.environ.get("ENTITY_RETRIEVAL_TOKEN"),
            max_workers=8,
            candidate_retrieval_limit=10,
            model_path="./crocodile/models/default.h5",
            save_output_to_csv=False,
            columns_type=classification,
            entity_bow_endpoint=os.environ.get("ENTITY_BOW_ENDPOINT")
            
        )
        croco.run()

    background_tasks.add_task(run_crocodile_task)

    return {
        "message": "CSV table added successfully.",
        "tableName": table_name,
        "datasetName": datasetName,
    }


@router.get("/datasets")
def get_datasets(
    limit: int = Query(10), 
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    db: Database = Depends(get_db)
):
    """
    Get datasets with bi-directional keyset pagination, using ObjectId as the cursor.
    Supports both forward (next_cursor) and backward (prev_cursor) navigation.
    """
    # Determine pagination direction and set up query
    query_filter = {}
    sort_direction = 1  # Default ascending (forward)
    
    if next_cursor and prev_cursor:
        raise HTTPException(
            status_code=400, 
            detail="Only one of next_cursor or prev_cursor should be provided"
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
        "pagination": {
            "next_cursor": next_cursor,
            "prev_cursor": prev_cursor
        },
    }


@router.get("/datasets/{dataset_name}/tables")
def get_tables(
    dataset_name: str,
    limit: int = Query(10),
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    db: Database = Depends(get_db),
):
    """
    Get tables for a dataset with bi-directional keyset pagination.
    """
    # Ensure dataset exists
    if not db.datasets.find_one({"dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Determine pagination direction
    query_filter = {"dataset_name": dataset_name}
    sort_direction = 1  # Default ascending (forward)
    
    if next_cursor and prev_cursor:
        raise HTTPException(
            status_code=400, 
            detail="Only one of next_cursor or prev_cursor should be provided"
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
                if db.tables.count_documents({"dataset_name": dataset_name, "_id": {"$lt": first_id}}) > 0:
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
        "pagination": {
            "next_cursor": next_cursor,
            "prev_cursor": prev_cursor
        },
    }


@router.get("/datasets/{dataset_name}/tables/{table_name}")
def get_table(
    dataset_name: str,
    table_name: str,
    limit: int = Query(10),
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Get table data with bi-directional keyset pagination.
    """
    # Check dataset
    if not db.datasets.find_one({"dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Check table
    table = db.tables.find_one({"dataset_name": dataset_name, "table_name": table_name})
    if not table:
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    header = table.get("header", [])

    # Determine pagination direction
    query_filter = {"dataset_name": dataset_name, "table_name": table_name}
    sort_direction = 1  # Default ascending (forward)
    
    if next_cursor and prev_cursor:
        raise HTTPException(
            status_code=400, 
            detail="Only one of next_cursor or prev_cursor should be provided"
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
    results = crocodile_db.input_data.find(query_filter).sort("_id", sort_direction).limit(limit + 1)
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
                if crocodile_db.input_data.count_documents({
                    "dataset_name": dataset_name, 
                    "table_name": table_name, 
                    "_id": {"$lt": first_id}
                }) > 0:
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

    # Check if there are documents with ML_STATUS = TODO or DOING
    table_status_filter = {"dataset_name": dataset_name, "table_name": table_name, "ml_status": {"$in": ["TODO", "DOING"]}}
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
                linked_entities.append({"idColumn": col_index, "candidates": candidates})

        rows_formatted.append(
            {
                "idRow": row.get("row_id"),
                "data": row.get("data", []),
                "linked_entities": linked_entities,
            }
        )

    return {
        "data": {
            "datasetName": dataset_name,
            "tableName": table.get("table_name"),
            "status": status,
            "header": header,
            "rows": rows_formatted,
        },
        "pagination": {
            "next_cursor": next_cursor,
            "prev_cursor": prev_cursor
        },
    }


@router.post("/datasets", status_code=status.HTTP_201_CREATED)
def create_dataset(
    dataset_data: dict = Body(..., example={"dataset_name": "test"}),  # updated example key
    db: Database = Depends(get_db),
):
    """
    Create a new dataset.
    """
    existing = db.datasets.find_one(
        {"dataset_name": dataset_data.get("dataset_name")}
    )  # updated query key
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset with dataset_name {dataset_data.get('dataset_name')} already exists",
        )

    dataset_data["created_at"] = datetime.now()
    dataset_data["total_tables"] = 0
    dataset_data["total_rows"] = 0

    try:
        result = db.datasets.insert_one(dataset_data)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Dataset already exists")
    dataset_data["_id"] = str(result.inserted_id)

    return {"message": "Dataset created successfully", "dataset": dataset_data}


@router.delete("/datasets/{dataset_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_name: str,
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    """
    Delete a dataset by name.
    """
    # Check existence using uniform dataset key
    existing = db.datasets.find_one({"dataset_name": dataset_name})  # updated query key
    if not existing:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Delete all tables associated with this dataset
    db.tables.delete_many({"dataset_name": dataset_name})

    # Delete dataset
    db.datasets.delete_one({"dataset_name": dataset_name})  # updated query key

    # Optionally delete data from crocodile_db if needed
    return None


@router.delete(
    "/datasets/{dataset_name}/tables/{table_name}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_table(dataset_name: str, table_name: str, db: Database = Depends(get_db)):
    """
    Delete a table by name within a dataset.
    """
    # Ensure dataset exists using uniform dataset key
    dataset = db.datasets.find_one({"dataset_name": dataset_name})  # updated query key
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    table = db.tables.find_one({"dataset_name": dataset_name, "table_name": table_name})
    if not table:
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    row_count = table.get("total_rows", 0)

    # Delete table
    db.tables.delete_one({"dataset_name": dataset_name, "table_name": table_name})

    # Update dataset metadata
    db.datasets.update_one(
        {"name": dataset_name}, {"$inc": {"total_tables": -1, "total_rows": -row_count}}
    )

    # Optionally delete data from crocodile_db if needed
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
    crocodile_db: Database = Depends(get_crocodile_db),
    db: Database = Depends(get_db),
):
    """
    Update the annotation for a specific cell by marking a candidate as matching.
    This allows users to manually correct or validate entity linking results.
    
    The annotation can either reference an existing candidate by ID or provide
    a completely new candidate that doesn't exist in the current list.
    """
    # Check if dataset and table exist
    if not db.datasets.find_one({"dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    
    if not db.tables.find_one({"dataset_name": dataset_name, "table_name": table_name}):
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )
    
    # Find the row in the database
    row = crocodile_db.input_data.find_one({
        "dataset_name": dataset_name,
        "table_name": table_name,
        "row_id": row_id
    })
    
    if not row:
        raise HTTPException(
            status_code=404, 
            detail=f"Row {row_id} not found in table {table_name}"
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
                detail=f"Entity with ID {annotation.entity_id} not found in candidates. Please provide 'candidate_info' to add a new candidate."
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
        {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id
        },
        {"$set": {
            "el_results": el_results,
            "manually_annotated": True
        }}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=500, 
            detail="Failed to update annotation"
        )
    
    # Return the updated entity with all its information
    matched_candidate = next((c for c in updated_candidates if c["id"] == annotation.entity_id), None)
    
    return {
        "message": "Annotation updated successfully",
        "dataset_name": dataset_name,
        "table_name": table_name,
        "row_id": row_id,
        "column_id": column_id,
        "entity": matched_candidate,
        "manually_annotated": True
    }


@router.delete("/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}/candidates/{entity_id}")
def delete_candidate(
    dataset_name: str,
    table_name: str,
    row_id: int,
    column_id: int,
    entity_id: str,
    crocodile_db: Database = Depends(get_crocodile_db),
    db: Database = Depends(get_db),
):
    """
    Delete a specific candidate from the entity linking results for a cell.
    This allows users to remove unwanted or incorrect candidate entities.
    """
    # Check if dataset and table exist
    if not db.datasets.find_one({"dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    
    if not db.tables.find_one({"dataset_name": dataset_name, "table_name": table_name}):
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )
    
    # Find the row in the database
    row = crocodile_db.input_data.find_one({
        "dataset_name": dataset_name,
        "table_name": table_name,
        "row_id": row_id
    })
    
    if not row:
        raise HTTPException(
            status_code=404, 
            detail=f"Row {row_id} not found in table {table_name}"
        )
    
    # Get the current EL results
    el_results = row.get("el_results", {})
    column_candidates = el_results.get(str(column_id), [])
    
    if not column_candidates:
        raise HTTPException(
            status_code=404, 
            detail=f"No entity linking candidates found for column {column_id} in row {row_id}"
        )
    
    # Check if the entity exists in the candidates
    entity_exists = any(candidate.get("id") == entity_id for candidate in column_candidates)
    
    if not entity_exists:
        raise HTTPException(
            status_code=404, 
            detail=f"Entity with ID {entity_id} not found in candidates for column {column_id}"
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
        {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id
        },
        {"$set": {
            "el_results": el_results,
            "manually_annotated": True  # Mark as manually annotated since we're modifying the candidates
        }}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=500, 
            detail="Failed to delete candidate"
        )
    
    return {
        "message": "Candidate deleted successfully",
        "dataset_name": dataset_name,
        "table_name": table_name,
        "row_id": row_id,
        "column_id": column_id,
        "entity_id": entity_id,
        "remaining_candidates": len(updated_candidates)
    }
