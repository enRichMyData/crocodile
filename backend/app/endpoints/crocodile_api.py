import os
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from dependencies import get_db, get_crocodile_db
from endpoints.imdb_example import IMDB_EXAMPLE  # Example input
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query, status, UploadFile, File, Form
from pydantic import BaseModel
from pymongo.database import Database
from bson import ObjectId
import json

router = APIRouter()


class TableUpload(BaseModel):
    table_name: str
    header: List[str]
    total_rows: int
    classified_columns: Optional[Dict[str, Dict[str, str]]] = {}
    data: List[dict]


@router.post("/dataset/{datasetName}/table/json", status_code=status.HTTP_201_CREATED)
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
    dataset = db.datasets.find_one({"name": datasetName})
    if not dataset:
        dataset_id = db.datasets.insert_one({
            "name": datasetName,
            "created_at": datetime.now(),
            "total_tables": 0,
            "total_rows": 0
        }).inserted_id
    else:
        dataset_id = dataset["_id"]
    
    # Create table metadata
    table_metadata = {
        "dataset_name": datasetName,
        "table_name": table_upload.table_name,
        "header": table_upload.header,
        "total_rows": table_upload.total_rows,
        "created_at": datetime.now(),
        "status": "processing"
    }
    db.tables.insert_one(table_metadata)
    
    # Update dataset metadata
    db.datasets.update_one(
        {"_id": dataset_id},
        {"$inc": {"total_tables": 1, "total_rows": table_upload.total_rows}}
    )
    
    # Trigger background task
    def run_crocodile_task():
        from crocodile import Crocodile

        croco = Crocodile(
            input_csv=pd.DataFrame(table_upload.data),
            dataset_name=datasetName,
            table_name=table_upload.table_name,
            max_candidates=3,
            entity_retrieval_endpoint=os.environ.get("ENTITY_RETRIEVAL_ENDPOINT"),
            entity_retrieval_token=os.environ.get("ENTITY_RETRIEVAL_TOKEN"),
            max_workers=8,
            candidate_retrieval_limit=10,
            model_path="./crocodile/models/default.h5",
            save_output_to_csv=False,
        )
        croco.run()
        
        # Update table status to completed
        db.tables.update_one(
            {"dataset_name": datasetName, "table_name": table_upload.table_name},
            {"$set": {"status": "completed", "completed_at": datetime.now()}}
        )

    background_tasks.add_task(run_crocodile_task)

    return {
        "message": "Table added successfully.",
        "tableName": table_upload.table_name,
        "datasetName": datasetName,
    }


@router.post("/dataset/{datasetName}/table/csv", status_code=status.HTTP_201_CREATED)
def add_table_csv(
    datasetName: str,
    file: UploadFile = File(...),
    table_name: str = Form(...),
    column_classification: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    db: Database = Depends(get_db)
):
    # Read CSV file and convert NaN values to None
    df = pd.read_csv(file.file)
    df = df.replace({np.nan: None})  # permanent fix for JSON serialization
    
    header = df.columns.tolist()
    total_rows = len(df)
    
    # Parse column_classification if provided
    classification = json.loads(column_classification) if column_classification else {}
    
    # Check if dataset exists; if not, create it
    dataset = db.datasets.find_one({"name": datasetName})
    if not dataset:
        dataset_id = db.datasets.insert_one({
            "name": datasetName,
            "created_at": datetime.now(),
            "total_tables": 0,
            "total_rows": 0
        }).inserted_id
    else:
        dataset_id = dataset["_id"]
    
    # Create table metadata
    table_metadata = {
        "dataset_name": datasetName,
        "table_name": table_name,
        "header": header,
        "total_rows": total_rows,
        "created_at": datetime.now(),
        "status": "processing",
        "classified_columns": classification  # updated field for CSV input
    }
    db.tables.insert_one(table_metadata)
    
    # Update dataset metadata
    db.datasets.update_one(
        {"_id": dataset_id},
        {"$inc": {"total_tables": 1, "total_rows": total_rows}}
    )
    
    # Trigger background task
    def run_crocodile_task():
        from crocodile import Crocodile

        croco = Crocodile(
            input_csv=df,
            dataset_name=datasetName,
            table_name=table_name,
            max_candidates=3,
            entity_retrieval_endpoint=os.environ.get("ENTITY_RETRIEVAL_ENDPOINT"),
            entity_retrieval_token=os.environ.get("ENTITY_RETRIEVAL_TOKEN"),
            max_workers=8,
            candidate_retrieval_limit=10,
            model_path="./crocodile/models/default.h5",
            save_output_to_csv=False,
        )
        croco.run()
        
        # Update table status to completed
        db.tables.update_one(
            {"dataset_name": datasetName, "table_name": table_name},
            {"$set": {"status": "completed", "completed_at": datetime.now()}}
        )

    background_tasks.add_task(run_crocodile_task)
    
    return {
        "message": "CSV table added successfully.",
        "tableName": table_name,
        "datasetName": datasetName,
    }


@router.get("/datasets")
def get_datasets(
    limit: int = Query(10),
    cursor: Optional[str] = Query(None),
    db: Database = Depends(get_db)
):
    """
    Get datasets with keyset pagination, using ObjectId as the cursor.
    """
    query_filter = {}
    if cursor:
        try:
            query_filter["_id"] = {"$gt": ObjectId(cursor)}
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")

    results = db.datasets.find(query_filter).sort("_id", 1).limit(limit)
    datasets = list(results)
    next_cursor = datasets[-1]["_id"] if datasets else None

    for dataset in datasets:
        dataset["_id"] = str(dataset["_id"])
        if "created_at" in dataset:
            dataset["created_at"] = dataset["created_at"].isoformat()

    return {
        "data": datasets,
        "pagination": {
            "next_cursor": str(next_cursor) if next_cursor else None,
            "limit": limit
        }
    }


@router.get("/datasets/{dataset_name}/tables")
def get_tables(
    dataset_name: str,
    limit: int = Query(10),
    cursor: Optional[str] = Query(None),
    db: Database = Depends(get_db)
):
    """
    Get tables for a dataset with keyset pagination, using ObjectId as the cursor.
    """
    # Ensure dataset exists
    if not db.datasets.find_one({"name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    
    query_filter = {"dataset_name": dataset_name}
    if cursor:
        try:
            query_filter["_id"] = {"$gt": ObjectId(cursor)}
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")

    results = db.tables.find(query_filter).sort("_id", 1).limit(limit)
    tables = list(results)
    next_cursor = tables[-1]["_id"] if tables else None

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
            "next_cursor": str(next_cursor) if next_cursor else None,
            "limit": limit
        }
    }


@router.get("/datasets/{dataset_name}/tables/{table_name}")
def get_table(
    dataset_name: str,
    table_name: str,
    limit: int = Query(10),
    cursor: Optional[str] = Query(None),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db)
):
    """
    Get table data with keyset pagination, using ObjectId as the cursor.
    Returns *all* candidate entities for each column that has any.
    """
    # Check dataset
    if not db.datasets.find_one({"name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Check table
    table = db.tables.find_one({"dataset_name": dataset_name, "table_name": table_name})
    if not table:
        raise HTTPException(status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}")

    header = table.get("header", [])

    # Pagination filter
    query_filter = {"dataset_name": dataset_name, "table_name": table_name}
    if cursor:
        try:
            query_filter["_id"] = {"$gt": ObjectId(cursor)}
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")

    # Fetch rows from the Crocodile-processed data
    results = crocodile_db.input_data.find(query_filter).sort("_id", 1).limit(limit)
    raw_rows = list(results)

    # Build a cleaned-up response with *all* candidates
    rows_formatted = []
    for row in raw_rows:
        linked_entities = []
        el_results = row.get("el_results", {})

        # For each column, gather all candidates
        for col_index in range(len(header)):
            candidates = el_results.get(str(col_index), [])
            if candidates:
                linked_entities.append({
                    "idColumn": col_index,
                    "candidates": candidates
                })

        rows_formatted.append({
            "idRow": row.get("row_id"),
            "data": row.get("data", []),
            "linked_entities": linked_entities
        })
    
    print(rows_formatted)
    # Determine next cursor
    next_cursor = str(raw_rows[-1]["_id"]) if raw_rows else None
    return {
        "data": {
            "datasetName": dataset_name,
            "tableName": table.get("table_name"),
            "header": header,
            "rows": rows_formatted
        },
        "pagination": {
            "next_cursor": next_cursor
        }
    }


@router.post("/datasets", status_code=status.HTTP_201_CREATED)
def create_dataset(
    dataset_data: dict = Body(..., example={"name": "test"}),
    db: Database = Depends(get_db)
):
    """
    Create a new dataset.
    """
    existing = db.datasets.find_one({"name": dataset_data.get("name")})
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset with name {dataset_data.get('name')} already exists"
        )

    dataset_data["created_at"] = datetime.now()
    dataset_data["total_tables"] = 0
    dataset_data["total_rows"] = 0

    result = db.datasets.insert_one(dataset_data)
    dataset_data["_id"] = str(result.inserted_id)

    return {"message": "Dataset created successfully", "dataset": dataset_data}


@router.delete("/datasets/{dataset_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_name: str,
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db)
):
    """
    Delete a dataset by name.
    """
    existing = db.datasets.find_one({"name": dataset_name})
    if not existing:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Delete all tables associated with this dataset
    db.tables.delete_many({"dataset_name": dataset_name})

    # Delete dataset
    db.datasets.delete_one({"name": dataset_name})

    # Optionally delete data from crocodile_db if needed
    return None


@router.delete("/datasets/{dataset_name}/tables/{table_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_table(
    dataset_name: str,
    table_name: str,
    db: Database = Depends(get_db)
):
    """
    Delete a table by name within a dataset.
    """
    dataset = db.datasets.find_one({"name": dataset_name})
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    table = db.tables.find_one({"dataset_name": dataset_name, "table_name": table_name})
    if not table:
        raise HTTPException(status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}")

    row_count = table.get("total_rows", 0)

    # Delete table
    db.tables.delete_one({"dataset_name": dataset_name, "table_name": table_name})

    # Update dataset metadata
    db.datasets.update_one(
        {"name": dataset_name},
        {"$inc": {"total_tables": -1, "total_rows": -row_count}}
    )

    # Optionally delete data from crocodile_db if needed
    return None