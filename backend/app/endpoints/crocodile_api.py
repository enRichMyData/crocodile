from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from pymongo.database import Database
from typing import List, Optional
from dependencies import get_db
from schemas import DatasetItem, TableItem  # Import models from schemas 
import pandas as pd
import json
from io import StringIO
from bson import ObjectId

router = APIRouter()

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
    """

    #Get the "dataset_trace" collection
    dataset_trace = db["dataset_trace"]
    table_trace = db["table_trace"]

    #Query to retrieve datasets with pagination
    query = {}
    if cursor:
        try:
            query["_id"] = {"$gt": ObjectId(cursor)}
        except:
            raise HTTPException(status_code=400, detail="Invalid cursor format")
        
    #Get datasets with pagination
    datasets_cursor = dataset_trace.find(query).limit(limit + 1) # get one extra document to check if there's a next page
    datasets_list = list(datasets_cursor)

    #Check if there's a next page
    has_next_page = len(datasets_list) > limit
    if has_next_page:
        datasets_list = datasets_list[:limit] #remove the extra document
    
    #Format the response
    formatted_datasets = []
    for dataset in datasets_list:
        #Count tables for this dataset
        tables_count = table_trace.count_documents({"dataset_name": dataset["dataset_name"]})
        formatted_datasets.append({
            "datasetName": dataset["dataset_name"],
            "tablesCount": tables_count,
            "status": dataset.get("status", "pending") #default to "pending" if status is not set
        })
    
    #set up pagination
    next_cursor = None
    if has_next_page and datasets_list:
        next_cursor = str(datasets_list[-1]["_id"])
    
    return{
        "datasets": formatted_datasets,
        "pagination": {
            "nextCursor": next_cursor,
            "previousCursor": None
        }
    }

@router.delete("/dataset/{datasetName}")
def delete_dataset(datasetName: str, db: Database = Depends(get_db)):
    """
    Deletes a dataset and all its associated tables from the database.
    """

    #get the colelctions
    dataset_trace = db["dataset_trace"]
    table_trace = db["table_trace"]
    input_data = db["input_data"]

    #Check if dataset exists
    dataset = dataset_trace.find_one({"dataset_name": datasetName})
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
    
    #Delete dataset from dataset_trace
    dataset_trace.delete_one({"dataset_name": datasetName})

    #Delete all tables for this dataset from table_trace
    table_trace.delete_many({"dataset_name": datasetName})

    #Delete all rows for this dataset from input_data
    input_data.delete_many({"dataset_name": datasetName})

    return {"message": f"Dataset '{datasetName}' deleted successfully."}



# --------- Table Management Endpoints ---------
@router.post("/dataset/{datasetName}/table/upload/csv", status_code=status.HTTP_201_CREATED)
async def upload_csv_table(
    datasetName: str, 
    file: UploadFile = File(...), 
    db: Database = Depends(get_db)
):
    """
    Add a new table from a CSV file to an existing dataset for processing.
    """
    # Check if dataset exists
    dataset_trace = db["dataset_trace"]
    dataset = dataset_trace.find_one({"dataset_name": datasetName})
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
    
    try:
        # Read CSV content
        content = await file.read()
        csv_text = content.decode()
        df = pd.read_csv(StringIO(csv_text))
        
        # Get table name from filename (without extension)
        tableName = file.filename.rsplit('.', 1)[0]
        
        # Prepare data for insertion
        table_trace = db["table_trace"]
        input_data = db["input_data"]
        
        # Check if table already exists
        existing_table = table_trace.find_one({
            "dataset_name": datasetName,
            "table_name": tableName
        })
        if existing_table:
            raise HTTPException(
                status_code=400, 
                detail=f"Table '{tableName}' already exists in dataset '{datasetName}'"
            )
        
        # Extract header and rows
        header = df.columns.tolist()
        
        # Insert table into table_trace
        table_trace.insert_one({
            "dataset_name": datasetName,
            "table_name": tableName,
            "status": "PENDING",
            "header": header,
            "total_rows": len(df),
            "processed_rows": 0
        })
        
        # Insert rows into input_data
        rows_to_insert = []
        for idx, row_data in enumerate(df.values.tolist()):
            # Convert any non-string values to strings
            row_data_str = [str(item) for item in row_data]
            
            rows_to_insert.append({
                "dataset_name": datasetName,
                "table_name": tableName,
                "row_id": idx,
                "data": row_data_str,
                "status": "TODO"
            })
        
        if rows_to_insert:
            input_data.insert_many(rows_to_insert)
        
        # Update dataset's table count
        dataset_trace.update_one(
            {"dataset_name": datasetName},
            {"$inc": {"total_tables": 1}}
        )
        
        return {
            "message": "Table added successfully.",
            "tableName": tableName,
            "datasetName": datasetName
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@router.post("/dataset/{datasetName}/table/upload/json", status_code=status.HTTP_201_CREATED)
async def upload_json_table(
    datasetName: str,
    table_data: dict,
    db: Database = Depends(get_db)
):
    """
    Add a new table in JSON format to an existing dataset for processing.
    """
    # Check if dataset exists
    dataset_trace = db["dataset_trace"]
    dataset = dataset_trace.find_one({"dataset_name": datasetName})
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
    
    try:
        # Validate required fields
        required_fields = ["tableName", "header", "rows"]
        for field in required_fields:
            if field not in table_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        tableName = table_data["tableName"]
        header = table_data["header"]
        rows = table_data["rows"]
        
        # Prepare data for insertion
        table_trace = db["table_trace"]
        input_data = db["input_data"]
        
        # Check if table already exists
        existing_table = table_trace.find_one({
            "dataset_name": datasetName,
            "table_name": tableName
        })
        if existing_table:
            raise HTTPException(
                status_code=400, 
                detail=f"Table '{tableName}' already exists in dataset '{datasetName}'"
            )
        
        # Insert table into table_trace
        table_trace.insert_one({
            "dataset_name": datasetName,
            "table_name": tableName,
            "status": "PENDING",
            "header": header,
            "total_rows": len(rows),
            "processed_rows": 0
        })
        
        # Insert rows into input_data
        rows_to_insert = []
        for row in rows:
            # Ensure each row has the required fields
            if "idRow" not in row or "data" not in row:
                raise HTTPException(
                    status_code=400,
                    detail=f"Row missing required fields (idRow or data)"
                )
            
            rows_to_insert.append({
                "dataset_name": datasetName,
                "table_name": tableName,
                "row_id": row["idRow"],
                "data": row["data"],
                "status": "TODO"
            })
        
        if rows_to_insert:
            input_data.insert_many(rows_to_insert)
        
        # Update dataset's table count
        dataset_trace.update_one(
            {"dataset_name": datasetName},
            {"$inc": {"total_tables": 1}}
        )
        
        return {
            "message": "Table added successfully.",
            "tableName": tableName,
            "datasetName": datasetName
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing JSON: {str(e)}")

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
def list_tables(
    datasetName: str, 
    cursor: Optional[str] = None, 
    limit: int = 10, 
    db: Database = Depends(get_db)
):
    """
    Retrieve a list of tables in dataset with pagination.
    """

    #Check if datasest exists
    dataset_trace = db["dataset_trace"]
    dataset = dataset_trace.find_one({"dataset_name": datasetName})
    if not dataset:
        raise HTTPException (status_code=404, detail=f"Dataset '{datasetName}' not found")
    
    #Build query 
    table_trace = db["table_trace"]
    query = {"dataset_name": datasetName}
    if cursor: 
        try:
            query["_id"] = {"$gt": ObjectId(cursor)}
        except:
            raise HTTPException (status_code=400, detail="Invalid cursor format")
    
    #Get tables with pagination 
    tables_cursor = table_trace.find(query).limit(limit + 1) #get one extra document to check if there's a next page
    tables = list(tables_cursor)

    #check if there's a next page
    has_next_page = len(tables) > limit
    if has_next_page:
        tables = tables[:limit] #remove the extra document
    
    #format the response
    tables_list = []
    for table in tables:
        tables_list.append({
            "tableName": table["table_name"],
            "status": table["status"]
        })
    #set up pagination
    next_cursor = None
    if has_next_page and tables:
        next_cursor = str(tables[-1]["_id"])
    
    return {
        "tables": tables_list,
        "pagination": {
            "nextCursor": next_cursor,
            "previousCursor": None
        }
    }

@router.get("/dataset/{datasetName}/table/{tableName}")
def get_table_results(
    datasetName: str, 
    tableName: str, 
    cursor: Optional[int] = None, 
    limit: int = 10, 
    db: Database = Depends(get_db)
):
    """
    Retrieve the processing results for a specific table, including entity linking.
    """
    # Check if dataset exists
    dataset_trace = db["dataset_trace"]
    dataset = dataset_trace.find_one({"dataset_name": datasetName})
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
    
    # Check if table exists
    table_trace = db["table_trace"]
    table = table_trace.find_one({
        "dataset_name": datasetName,
        "table_name": tableName
    })
    if not table:
        raise HTTPException(
            status_code=404, 
            detail=f"Table '{tableName}' not found in dataset '{datasetName}'"
        )
    
    # Query the input_data collection
    input_data = db["input_data"]
    query = {
        "dataset_name": datasetName,
        "table_name": tableName
    }
    
    if cursor is not None:
        query["row_id"] = {"$gt": cursor}
    
    # Get rows with pagination
    rows_cursor = input_data.find(query).sort("row_id", 1).limit(limit + 1)
    rows = list(rows_cursor)
    
    # Check if there's a next page
    has_next_page = len(rows) > limit
    if has_next_page:
        rows = rows[:limit]  # Remove the extra document
    
    # Format the response
    formatted_rows = []
    for row in rows:
        # Get linked entities (el_results) if available
        linked_entities = []
        el_results = row.get("el_results", {})
        
        for col_idx in range(len(row["data"])):
            # Check if we have entity linking for this column
            col_key = str(col_idx)
            if col_key in el_results and el_results[col_key]:
                # Use the top candidate (highest score)
                top_candidate = el_results[col_key][0]
                linked_entities.append({
                    "idColumn": col_idx,
                    "entity": {
                        "id": top_candidate.get("id"),
                        "name": top_candidate.get("name"),
                        "description": top_candidate.get("description", "")
                    }
                })
            else:
                # No entity for this column
                linked_entities.append({
                    "idColumn": col_idx,
                    "entity": None
                })
        
        formatted_rows.append({
            "idRow": row["row_id"],
            "data": row["data"],
            "linked_entities": linked_entities
        })
    
    # Set up pagination
    next_cursor = None
    previous_cursor = None
    
    if has_next_page and rows:
        next_cursor = str(rows[-1]["row_id"])
    
    if cursor is not None:
        # Find the previous page's last row
        prev_query = {
            "dataset_name": datasetName,
            "table_name": tableName,
            "row_id": {"$lt": cursor}
        }
        prev_rows = list(input_data.find(prev_query).sort("row_id", -1).limit(1))
        if prev_rows:
            previous_cursor = str(prev_rows[0]["row_id"])
    
    return {
        "data": {
            "datasetName": datasetName,
            "tableName": tableName,
            "header": table["header"],
            "rows": formatted_rows
        },
        "pagination": {
            "nextCursor": next_cursor,
            "previousCursor": previous_cursor
        }
    }

@router.delete("/dataset/{datasetName}/table/{tableName}")
def delete_table(
    datasetName: str, 
    tableName: str, 
    db: Database = Depends(get_db)
):
    """
    Delete a specific table from a dataset.
    """
    # Check if dataset exists
    dataset_trace = db["dataset_trace"]
    dataset = dataset_trace.find_one({"dataset_name": datasetName})
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
    
    # Check if table exists
    table_trace = db["table_trace"]
    table = table_trace.find_one({
        "dataset_name": datasetName,
        "table_name": tableName
    })
    if not table:
        raise HTTPException(
            status_code=404, 
            detail=f"Table '{tableName}' not found in dataset '{datasetName}'"
        )
    
    # Delete table from table_trace
    table_trace.delete_one({
        "dataset_name": datasetName,
        "table_name": tableName
    })
    
    # Delete related data from input_data
    input_data = db["input_data"]
    input_data.delete_many({
        "dataset_name": datasetName,
        "table_name": tableName
    })
    
    # Update dataset's table count
    dataset_trace.update_one(
        {"dataset_name": datasetName},
        {"$inc": {"total_tables": -1}}
    )
    
    return {
        "message": f"Table '{tableName}' deleted from dataset '{datasetName}'."
    }

# ...existing endpoints...
@router.get("/db_name")
def example_endpoint(db: Database = Depends(get_db)):
    return {"database": db.name}
