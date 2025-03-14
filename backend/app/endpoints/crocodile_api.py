from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from pymongo.database import Database
from dependencies import get_db
from endpoints.imdb_example import IMDB_EXAMPLE  # Example input
from endpoints.crocodile_runner import run_crocodile_task  # The function we'll call in background

router = APIRouter()

# ------------------------------------------------------------------
# Dataset Management Endpoints
# ------------------------------------------------------------------

@router.post("/dataset", status_code=status.HTTP_201_CREATED)
def create_dataset(datasets: List[dict], db: Database = Depends(get_db)):
    """
    Create one or more datasets.
    """
    dataset_trace_coll = db["dataset_trace"]
    processed = []
    for d in datasets:
        dataset_name = d.get("dataset_name")
        table_name = d.get("table_name")

        dataset_trace_coll.update_one(
            {"dataset_name": dataset_name},
            {
                "$setOnInsert": {
                    "dataset_name": dataset_name,
                    "status": "PENDING",
                    "total_tables": 0,
                    "total_rows": 0
                }
            },
            upsert=True
        )
        processed.append({"datasetName": dataset_name, "tablesProcessed": [table_name]})

    return {
        "message": "Dataset created successfully.",
        "datasetsProcessed": processed
    }


@router.get("/dataset")
def list_datasets(cursor: Optional[str] = None, limit: int = 10, db: Database = Depends(get_db)):
    """
    List datasets.
    """
    dataset_trace_coll = db["dataset_trace"]
    query = {}
    if cursor:
        from bson import ObjectId
        query["_id"] = {"$gt": ObjectId(cursor)}

    datasets = list(dataset_trace_coll.find(query).limit(limit))
    next_cursor = str(datasets[-1]["_id"]) if datasets else None

    formatted = [
        {
            "datasetName": ds.get("dataset_name"),
            "status": ds.get("status"),
            "total_tables": ds.get("total_tables", 0),
            "total_rows": ds.get("total_rows", 0)
        }
        for ds in datasets
    ]
    
    return {
        "datasets": formatted,
        "next_cursor": next_cursor
    }


@router.delete("/dataset/{datasetName}")
def delete_dataset(datasetName: str, db: Database = Depends(get_db)):
    """
    Delete a dataset and all its related entries.
    """
    db["dataset_trace"].delete_one({"dataset_name": datasetName})
    db["table_trace"].delete_many({"dataset_name": datasetName})
    db["input_data"].delete_many({"dataset_name": datasetName})

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
    background_tasks: BackgroundTasks = None,  # <-- Use BackgroundTasks
    db: Database = Depends(get_db)
):
    """
    Add a new table to an existing dataset.
    After onboarding the data, trigger the Crocodile process in the background.
    """
    # Check if the table already exists
    if db["table_trace"].find_one({"dataset_name": datasetName, "table_name": table_upload.table_name}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Table '{table_upload.table_name}' already exists in dataset '{datasetName}'."
        )

    # Insert table trace document
    db["table_trace"].insert_one({
        "dataset_name": datasetName,
        "table_name": table_upload.table_name,
        "header": table_upload.header,
        "total_rows": table_upload.total_rows,
        "status": "PENDING"
    })

    # Prepare a list of context columns (if needed)
    context_columns = [str(i) for i in range(len(table_upload.header))]
    
    # Insert each row of data into input_data collection
    for index, row in enumerate(table_upload.data):
        db["input_data"].insert_one({
            "dataset_name": datasetName,
            "table_name": table_upload.table_name,
            "row_id": index,
            "data": [row.get(col, None) for col in table_upload.header],
            "classified_columns": table_upload.classified_columns or {},
            "context_columns": context_columns,
            "correct_qids": {},
            "status": "TODO"
        })

    # Update the dataset trace with new table and rows info
    db["dataset_trace"].update_one(
        {"dataset_name": datasetName},
        {
            "$setOnInsert": {
                "dataset_name": datasetName,
                "status": "PENDING"
            },
            "$inc": {
                "total_tables": 1,
                "total_rows": table_upload.total_rows
            }
        },
        upsert=True
    )

    # Immediately trigger the global Crocodile processing task
    print("Triggering Crocodile task...")
    background_tasks.add_task(run_crocodile_task)

    return {
        "message": "Table added successfully.",
        "tableName": table_upload.table_name,
        "datasetName": datasetName
    }