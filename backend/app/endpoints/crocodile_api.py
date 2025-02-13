from fastapi import APIRouter, Depends, HTTPException, status, Query
from pymongo.database import Database
from typing import List, Optional
from dependencies import get_db
from schemas import DatasetItem, TableItem  # Import models from schemas

router = APIRouter()

@router.get("/dashboard/{collection_name}")
def get_collection_data(collection_name: str, db=Depends(get_db)):
    """Fetch data from a given MongoDB collection."""
    if collection_name not in db.list_collection_names():
        return {"error": "Collection not found"}

    collection = db[collection_name]
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB _id field
    return {"collection": collection_name, "data": data}

# --------- Dataset Management Endpoints ---------
@router.post("/dataset", status_code=status.HTTP_201_CREATED)
def create_dataset(datasets: List[DatasetItem], db: Database = Depends(get_db)):
    """
    Creates one or more datasets and their associated tables in the database.

    this function takes a list of datasets, inserts each datasest into the "dataset_trace" 
    collection (if it doesn't already exist), and then inserts the associated tables
    into the "table_trace" collection. Each dataset and table starts with a "PENDING" status.

    args:
        datasets (List[DatasetItem]): A list of dataset objects containing tables.
        db (Database): The MongoDB database instance.
    
    returns:
        dict: A confirmation message and details about the processed datasets.
    """
    dataset_trace = db["dataset_trace"]
    table_trace = db["table_trace"]

    processed_datasets = []
    for dataset in datasets:
        # Insert dataset into `dataset_trace`
        dataset_trace.update_one(
            {"dataset_name": dataset.datasetName},
            {"$setOnInsert": {"total_tables": 0, "processed_tables": 0, "status": "PENDING"}},
            upsert=True,
        )

        # Insert tables associated initial tables into `table_trace`
        for table in dataset.tables:
            table_trace.insert_one(
                {
                    "dataset_name": dataset.datasetName,
                    "table_name": table.tableName,
                    "status": "PENDING",
                    "header": table.header,
                    "total_rows": len(table.rows),
                    "processed_rows": 0,
                }
            )
        # Append dataset details to the response
        processed_datasets.append({"datasetName": dataset.datasetName, "tablesProcessed": len(dataset.tables)})
    # Return confirmation message and details about the processed datasets
    return {"message": "Datasets created successfully.", "datasetsProcessed": processed_datasets}


@router.get("/dataset")
def list_datasets(cursor: Optional[str] = None, limit: int = 10, db: Database = Depends(get_db)):
    """
    Retrieves a list of datasets from the database with pagination support.

    this function fetches datasets stored in the ""dataset_trace"" collection, applying
    pagination using a cursor-bassed approach. The "_id" field is sued for pagination, 
    allowing efficient retrieval of datasets in chunks.

    args:
        cursor (Optional[str]): A string the last seen dataset's "_id" for pagination.
                                If provided, only datasets with and "_id" greater than this value
                                will be retrieved.
        limit (int): The maximum number of datasets to return. default is 10.
        db (Database): The MongoDB database instance.

    returns:
        dict: A dictionary containing:
            - "Datasets": A list of dataset documents.
            - "pagination": A dictionary with the next cursor value ("nextCursor")
                            if more datasets exist.
    """
    # Get the "dataset_trace" collection
    dataset_trace = db["dataset_trace"]
    
    # Query to retrieve datasets with pagination
    query = {}
    if cursor:
        query["_id"] = {"$gt": cursor}  # Pagination logic

    # Retrieve datasets with pagination
    datasets = list(dataset_trace.find(query).limit(limit))
    
    # Convert ObjectId to string
    for dataset in datasets:
        dataset["_id"] = str(dataset["_id"])  # Convert ObjectId to string
    # Return datasets and pagination information
    return {"datasets": datasets, "pagination": {"nextCursor": datasets[-1]["_id"] if datasets else None}}


@router.delete("/dataset/{datasetName}")
def delete_dataset(datasetName: str, db: Database = Depends(get_db)):
    """
    Deletes a dataset and all its associated tables from the database.

    this function removes the dataset entry from the "table_trace" collection
    and delete all related tables stored in the "table_trace" collection.

    args:
        datasetName (str): The name of the dataset to be deleted.
        db (Database): The MongoDB database instance.
    
    returns:
        dict: A confirmation message indicating the dataset has been deleted.
    """
    # Get the "dataset_trace" and "table_trace" collections
    dataset_trace = db["dataset_trace"]
    table_trace = db["table_trace"]

    # Delete dataset and tables
    dataset_trace.delete_one({"dataset_name": datasetName})
    table_trace.delete_many({"dataset_name": datasetName})

    return {"message": f"Dataset '{datasetName}' deleted successfully."}


# --------- Table Management Endpoints ---------
@router.post("/dataset/{datasetName}/table", status_code=status.HTTP_201_CREATED)
def add_table(datasetName: str, table: TableItem, db: Database = Depends(get_db)):
    """
    Adds a new table to an existing dataset in the database.

    this function inserts a new table entry into the "table_trace" collection,
    linking it to the specified dataset. The table starts with a "PENDING" 
    and tracks column headers and row counts.

    args:
        datasetName(str): The name of the dataset to which the table belongs.
        table (TableItem): The table object containing its name, headers, and rows.
        db (Database): The MongoDB database instance.

    returns:
        dict: A confirmation message with the added table name it's dataset.c  
    """
    table_trace = db["table_trace"]
    
    # Insert table into `table_trace`
    table_trace.insert_one(
        {
            "dataset_name": datasetName,
            "table_name": table.tableName,
            "status": "PENDING",
            "header": table.header,
            "total_rows": len(table.rows),
            "processed_rows": 0,
        }
    )

    return {"message": "Table added successfully.", "tableName": table.tableName, "datasetName": datasetName}


@router.get("/dataset/{datasetName}/table")
def list_tables(datasetName: str, cursor: Optional[str] = None, limit: int = 10, db: Database = Depends(get_db)):
    # ...retrieve tables with pagination...
    tables = [
        {"tableName": "DirectorsAndMovies", "status": "completed"},
        {"tableName": "ActorsAndMovies", "status": "processing"}
    ]
    pagination = {"nextCursor": "next_cursor_example", "previousCursor": None}
    return {"tables": tables, "pagination": pagination}

@router.get("/dataset/{datasetName}/table/{tableName}")
def get_table_results(datasetName: str, tableName: str, cursor: Optional[int] = None, limit: int = 10, db: Database = Depends(get_db)):
    # ...retrieve processing results for the specified table...
    data = {
        "datasetName": datasetName,
        "tableName": tableName,
        "header": ["Director", "Movie", "Genre", "Release Year"],
        "rows": [
            {
                "idRow": 6,
                "data": ["Christopher Nolan", "Inception", "Sci-Fi", "2010"],
                "linked_entities": [
                    {"idColumn": 0, "entity": {"id": "Q25191", "name": "Christopher Nolan", "description": "British-American filmmaker"}},
                    {"idColumn": 1, "entity": {"id": "Q25188", "name": "Inception", "description": "2010 science fiction film"}},
                    {"idColumn": 2, "entity": None},
                    {"idColumn": 3, "entity": None}
                ]
            },
            {
                "idRow": 7,
                "data": ["Quentin Tarantino", "Pulp Fiction", "Crime", "1994"],
                "linked_entities": [
                    {"idColumn": 0, "entity": {"id": "Q3772", "name": "Quentin Tarantino", "description": "American filmmaker"}},
                    {"idColumn": 1, "entity": {"id": "Q104123", "name": "Pulp Fiction", "description": "1994 film"}},
                    {"idColumn": 2, "entity": None},
                    {"idColumn": 3, "entity": None}
                ]
            }
        ]
    }
    pagination = {"nextCursor": "7", "previousCursor": "5"}
    return {"data": data, "pagination": pagination}

@router.delete("/dataset/{datasetName}/table/{tableName}")
def delete_table(datasetName: str, tableName: str, db: Database = Depends(get_db)):
    # ...delete table logic...
    return {"message": f"Table '{tableName}' deleted from dataset '{datasetName}'."}

# ...existing endpoints...
@router.get("/db_name")
def example_endpoint(db: Database = Depends(get_db)):
    return {"database": db.name}
