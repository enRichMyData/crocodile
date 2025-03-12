from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from pymongo.database import Database
from datetime import datetime
from dependencies import get_db
from endpoints.imdb_example import IMDB_EXAMPLE  # Dynamically loaded example
from endpoints.crocodile_runner import run_crocodile_task

router = APIRouter()

# ------------------------------------------------------------------
# Dataset Management Endpoints
# ------------------------------------------------------------------

@router.post("/dataset", status_code=status.HTTP_201_CREATED)
def create_dataset(datasets: List[dict], db: Database = Depends(get_db)):
    """
    Create one or more datasets.

    Expected payload: List of objects each containing at least 'dataset_name' and 'table_name'.
    This will initialize dataset trace entries.
    """
    dataset_trace_coll = db["dataset_trace"]
    processed = []
    for d in datasets:
        dataset_name = d.get("dataset_name")
        table_name = d.get("table_name")
        # Initialize dataset trace if not exists.
        dataset_trace_coll.update_one(
            {"dataset_name": dataset_name},
            {"$setOnInsert": {"dataset_name": dataset_name, "status": "PENDING", "total_tables": 0, "total_rows": 0}},
            upsert=True
        )
        processed.append({"datasetName": dataset_name, "tablesProcessed": [table_name]})
    return {"message": "Dataset created successfully.", "datasetsProcessed": processed}


@router.get("/dataset")
def list_datasets(cursor: Optional[str] = None, limit: int = 10, db: Database = Depends(get_db)):
    """
    List datasets.

    Returns a paginated list of dataset trace documents.
    """
    dataset_trace_coll = db["dataset_trace"]
    query = {}
    if cursor:
        from bson import ObjectId
        query["_id"] = {"$gt": ObjectId(cursor)}
    datasets = list(dataset_trace_coll.find(query).limit(limit))
    next_cursor = str(datasets[-1]["_id"]) if datasets else None

    formatted = []
    for ds in datasets:
        formatted.append({
            "datasetName": ds.get("dataset_name"),
            "status": ds.get("status"),
            "total_tables": ds.get("total_tables", 0),
            "total_rows": ds.get("total_rows", 0)
        })

    return {"datasets": formatted, "next_cursor": next_cursor}


@router.delete("/dataset/{datasetName}")
def delete_dataset(datasetName: str, db: Database = Depends(get_db)):
    """
    Delete a dataset and all its related entries.
    """
    dataset_trace_coll = db["dataset_trace"]
    table_trace_coll = db["table_trace"]
    input_data_coll = db["input_data"]

    dataset_trace_coll.delete_one({"dataset_name": datasetName})
    table_trace_coll.delete_many({"dataset_name": datasetName})
    input_data_coll.delete_many({"dataset_name": datasetName})

    return {"message": f"Dataset '{datasetName}' deleted successfully."}


# ------------------------------------------------------------------
# Table Management Endpoints
# ------------------------------------------------------------------

class TableUpload(BaseModel):
    table_name: str
    header: List[str]
    total_rows: int
    classified_columns: Optional[Dict[str, Dict[str, str]]] = {}
    data: List[dict]

@router.post("/dataset/{datasetName}/table", status_code=status.HTTP_201_CREATED)
def add_table(
    datasetName: str,
    table_upload: TableUpload = Body(..., example=IMDB_EXAMPLE),
    db: Database = Depends(get_db)
):
    """
    Add a new table to an existing dataset.

    This endpoint accepts a JSON payload with the following fields:
      - table_name: The name of the table.
      - header: A list of column names.
      - total_rows: The total number of rows.
      - classified_columns: (Optional) An object specifying column classifications (e.g., NE, LIT).
      - data: An array of row objects mapping header names to their values.

    The payload is validated and used for further processing by the Crocodile library.
    """
    table_trace_coll = db["table_trace"]
    input_data_coll = db["input_data"]
    dataset_trace_coll = db["dataset_trace"]

    if table_trace_coll.find_one({
        "dataset_name": datasetName,
        "table_name": table_upload.table_name
    }):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Table '{table_upload.table_name}' already exists in dataset '{datasetName}'."
        )

    table_trace_doc = {
        "dataset_name": datasetName,
        "table_name": table_upload.table_name,
        "header": table_upload.header,
        "total_rows": table_upload.total_rows,
        "status": "PENDING"
    }
    table_trace_coll.insert_one(table_trace_doc)

    context_columns = [str(i) for i in range(len(table_upload.header))]
    for index, row in enumerate(table_upload.data):
        row_list = [row.get(col, None) for col in table_upload.header]
        row_doc = {
            "dataset_name": datasetName,
            "table_name": table_upload.table_name,
            "row_id": index,
            "data": row_list,
            "classified_columns": table_upload.classified_columns or {},
            "context_columns": context_columns,
            "correct_qids": {},
            "status": "TODO"
        }
        input_data_coll.insert_one(row_doc)

    dataset_trace_coll.update_one(
        {"dataset_name": datasetName},
        {
            "$setOnInsert": {"dataset_name": datasetName, "status": "PENDING"},
            "$inc": {"total_tables": 1, "total_rows": table_upload.total_rows}
        },
        upsert=True
    )

    return {
        "message": "Table added successfully.",
        "tableName": table_upload.table_name,
        "datasetName": datasetName
    }


@router.get("/dataset/{datasetName}/table")
def list_tables(datasetName: str, cursor: Optional[str] = None, limit: int = 10, db: Database = Depends(get_db)):
    """
    List tables for a specific dataset.

    Returns a paginated list from the table_trace collection.
    """
    table_trace_coll = db["table_trace"]
    query = {"dataset_name": datasetName}
    if cursor:
        from bson import ObjectId
        query["_id"] = {"$gt": ObjectId(cursor)}
    tables = list(table_trace_coll.find(query).limit(limit))
    next_cursor = str(tables[-1]["_id"]) if tables else None

    formatted = []
    for t in tables:
        formatted.append({
            "tableName": t.get("table_name"),
            "status": t.get("status"),
            "total_rows": t.get("total_rows")
        })

    return {"tables": formatted, "next_cursor": next_cursor}


@router.get("/dataset/{datasetName}/table/{tableName}")
def get_table_results(
    datasetName: str,
    tableName: str,
    cursor: Optional[int] = None,
    limit: int = 10,
    db: Database = Depends(get_db)
):
    """
    Get table results for the specified table.

    If available, the result for each row includes a 'linked_entities' array.
    Each element in this array corresponds to a column and is set to the first entity
    candidate from 'el_results' for that column or null if none exist.
    """
    input_data_coll = db["input_data"]
    table_trace_coll = db["table_trace"]

    table_info = table_trace_coll.find_one({"dataset_name": datasetName, "table_name": tableName})
    if not table_info:
        raise HTTPException(status_code=404, detail="Table not found.")
    header = table_info.get("header", [])

    query = {"dataset_name": datasetName, "table_name": tableName}
    if cursor is not None:
        query["row_id"] = {"$gt": cursor}
    rows = list(input_data_coll.find(query).sort("row_id", 1).limit(limit))
    next_cursor = rows[-1]["row_id"] if rows else None

    formatted_rows = []
    for row in rows:
        row_data = row.get("data", [])
        el_results = row.get("el_results", {})
        linked_entities = []
        for i in el_results:
            entities = el_results[i]
            for entity in entities:
                del entity["features"]
            linked_entities.append({"idColumn": i, "entities": entities})
        formatted_rows.append({
            "idRow": row.get("row_id"),
            "data": row_data,
            "linked_entities": linked_entities
        })

    data = {
        "datasetName": datasetName,
        "tableName": tableName,
        "header": header,
        "rows": formatted_rows
    }
    pagination = {"next_cursor": next_cursor}
    return {"data": data, "pagination": pagination}


@router.delete("/dataset/{datasetName}/table/{tableName}")
def delete_table(datasetName: str, tableName: str, db: Database = Depends(get_db)):
    """
    Delete a specific table and its associated rows from the dataset.
    """
    table_trace_coll = db["table_trace"]
    input_data_coll = db["input_data"]

    table_trace_coll.delete_one({"dataset_name": datasetName, "table_name": tableName})
    input_data_coll.delete_many({"dataset_name": datasetName, "table_name": tableName})

    return {"message": f"Table '{tableName}' deleted from dataset '{datasetName}'."}


# ------------------------------------------------------------------
# Background Processing Endpoint
# ------------------------------------------------------------------
@router.post("/dataset/{datasetName}/table/{tableName}/process", status_code=status.HTTP_202_ACCEPTED)
def process_table_in_background(
    datasetName: str,
    tableName: str,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_db)
):
    """
    Kick off the long-running Crocodile process in the background.
    """
    db["table_trace"].update_one(
        {"dataset_name": datasetName, "table_name": tableName},
        {"$set": {"status": "PROCESSING"}}
    )

    background_tasks.add_task(run_crocodile_task, datasetName, tableName, db)
    return {"message": "Crocodile processing started in the background."}


# ------------------------------------------------------------------
# Utility Endpoint
# ------------------------------------------------------------------
@router.get("/db_name")
def example_endpoint(db: Database = Depends(get_db)):
    return {"database": db.name}