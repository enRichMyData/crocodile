from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from pymongo.database import Database
from typing import List, Optional
from dependencies import get_db
from schemas import DatasetItem, TableItem  # Import models from schemas 
#import pandas as pd
import json
from io import StringIO
from bson import ObjectId

#Import GraphQL resolvers
from get_graphql_methods.crocodile_getmethods import Query as GraphQLQuery

#Initialize the GraphQL query instance
graphql_query = GraphQLQuery()

router = APIRouter()

# --------- Dataset Management Endpoints ---------
@router.post("/dataset", status_code=status.HTTP_201_CREATED)
async def create_dataset(datasets: List[DatasetItem], db: Database = Depends(get_db)):
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
async def list_datasets(cursor: Optional[str] = None, limit: int = 10, db: Database = Depends(get_db)):
    """
    Retrieves a list of datasets from the database with pagination support.
    Using GraphQL query for pagination.
    """
    try:
        # Use the GraphQL query directly for pagination
        result = graphql_query.get_datasets(
            cursor=cursor,
            page_size=limit,
            direction="next"
        )
        
        # Format the response to match the REST API structure
        formatted_datasets = []
        for dataset in result.datasets:
            tables_count = db["table_trace"].count_documents({"dataset_name": dataset.dataset_name})
            
            formatted_datasets.append({
                "datasetName": dataset.dataset_name,
                "tablesCount": tables_count,
                "status": dataset.status
            })
        
        return {
            "datasets": formatted_datasets,
            "pagination": {
                "nextCursor": result.next_cursor,
                "previousCursor": result.previous_cursor
            }
        }
    except Exception as e:
        # Fall back to direct database query if GraphQL call fails
        dataset_trace = db["dataset_trace"]
        table_trace = db["table_trace"]

        query = {}
        if cursor:
            try:
                query["_id"] = {"$gt": ObjectId(cursor)}
            except:
                raise HTTPException(status_code=400, detail="Invalid cursor format")
            
        datasets_cursor = dataset_trace.find(query).limit(limit + 1)
        datasets_list = list(datasets_cursor)

        has_next_page = len(datasets_list) > limit
        if has_next_page:
            datasets_list = datasets_list[:limit]
        
        formatted_datasets = []
        for dataset in datasets_list:
            tables_count = table_trace.count_documents({"dataset_name": dataset["dataset_name"]})
            formatted_datasets.append({
                "datasetName": dataset["dataset_name"],
                "tablesCount": tables_count,
                "status": dataset.get("status", "pending")
            })
        
        next_cursor = None
        if has_next_page and datasets_list:
            next_cursor = str(datasets_list[-1]["_id"])
        
        return {
            "datasets": formatted_datasets,
            "pagination": {
                "nextCursor": next_cursor,
                "previousCursor": None
            }
        }


@router.delete("/dataset/{datasetName}")
async def delete_dataset(datasetName: str, db: Database = Depends(get_db)):
    """
    Deletes a dataset and all its associated tables from the database.
    """
    try:
        # Check if dataset exists using GraphQL query
        dataset_info = graphql_query.get_dataset_info(dataset_name=datasetName)
        if not dataset_info:
            raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
        
        # Delete from databases
        dataset_trace = db["dataset_trace"]
        table_trace = db["table_trace"]
        input_data = db["input_data"]
        
        dataset_trace.delete_one({"dataset_name": datasetName})
        table_trace.delete_many({"dataset_name": datasetName})
        input_data.delete_many({"dataset_name": datasetName})
        
        return {"message": f"Dataset '{datasetName}' deleted successfully."}
    except HTTPException:
        # Re-raise HTTPExceptions to maintain error status codes
        raise
    except Exception as e:
        # Check dataset directly in database as fallback
        dataset_trace = db["dataset_trace"]
        dataset = dataset_trace.find_one({"dataset_name": datasetName})
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
        
        table_trace = db["table_trace"]
        input_data = db["input_data"]
        
        dataset_trace.delete_one({"dataset_name": datasetName})
        table_trace.delete_many({"dataset_name": datasetName})
        input_data.delete_many({"dataset_name": datasetName})
        
        return {"message": f"Dataset '{datasetName}' deleted successfully."}


# --------- Table Management Endpoints ---------
# @router.post("/dataset/{datasetName}/table/upload/csv", status_code=status.HTTP_201_CREATED)
# async def upload_csv_table(
#     datasetName: str, 
#     file: UploadFile = File(...), 
#     db: Database = Depends(get_db)
# ):
#     """
#     Add a new table from a CSV file to an existing dataset for processing.
#     """
#     # Check if dataset exists
#     dataset_trace = db["dataset_trace"]
#     dataset = dataset_trace.find_one({"dataset_name": datasetName})
#     if not dataset:
#         raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
    
#     try:
#         # Read CSV content
#         content = await file.read()
#         csv_text = content.decode()
#         df = pd.read_csv(StringIO(csv_text))
        
#         # Get table name from filename (without extension)
#         tableName = file.filename.rsplit('.', 1)[0]
        
#         # Prepare data for insertion
#         table_trace = db["table_trace"]
#         input_data = db["input_data"]
        
#         # Check if table already exists
#         existing_table = table_trace.find_one({
#             "dataset_name": datasetName,
#             "table_name": tableName
#         })
#         if existing_table:
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Table '{tableName}' already exists in dataset '{datasetName}'"
#             )
        
#         # Extract header and rows
#         header = df.columns.tolist()
        
#         # Insert table into table_trace
#         table_trace.insert_one({
#             "dataset_name": datasetName,
#             "table_name": tableName,
#             "status": "PENDING",
#             "header": header,
#             "total_rows": len(df),
#             "processed_rows": 0
#         })
        
#         # Insert rows into input_data
#         rows_to_insert = []
#         for idx, row_data in enumerate(df.values.tolist()):
#             # Convert any non-string values to strings
#             row_data_str = [str(item) for item in row_data]
            
#             rows_to_insert.append({
#                 "dataset_name": datasetName,
#                 "table_name": tableName,
#                 "row_id": idx,
#                 "data": row_data_str,
#                 "status": "TODO"
#             })
        
#         if rows_to_insert:
#             input_data.insert_many(rows_to_insert)
        
#         # Update dataset's table count
#         dataset_trace.update_one(
#             {"dataset_name": datasetName},
#             {"$inc": {"total_tables": 1}}
#         )
        
#         return {
#             "message": "Table added successfully.",
#             "tableName": tableName,
#             "datasetName": datasetName
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

# @router.post("/dataset/{datasetName}/table/upload/json", status_code=status.HTTP_201_CREATED)
# async def upload_json_table(
#     datasetName: str,
#     table_data: dict,
#     db: Database = Depends(get_db)
# ):
#     """
#     Add a new table in JSON format to an existing dataset for processing.
#     """
#     # Check if dataset exists
#     dataset_trace = db["dataset_trace"]
#     dataset = dataset_trace.find_one({"dataset_name": datasetName})
#     if not dataset:
#         raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
    
#     try:
#         # Validate required fields
#         required_fields = ["tableName", "header", "rows"]
#         for field in required_fields:
#             if field not in table_data:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Missing required field: {field}"
#                 )
        
#         tableName = table_data["tableName"]
#         header = table_data["header"]
#         rows = table_data["rows"]
        
#         # Prepare data for insertion
#         table_trace = db["table_trace"]
#         input_data = db["input_data"]
        
#         # Check if table already exists
#         existing_table = table_trace.find_one({
#             "dataset_name": datasetName,
#             "table_name": tableName
#         })
#         if existing_table:
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Table '{tableName}' already exists in dataset '{datasetName}'"
#             )
        
#         # Insert table into table_trace
#         table_trace.insert_one({
#             "dataset_name": datasetName,
#             "table_name": tableName,
#             "status": "PENDING",
#             "header": header,
#             "total_rows": len(rows),
#             "processed_rows": 0
#         })
        
#         # Insert rows into input_data
#         rows_to_insert = []
#         for row in rows:
#             # Ensure each row has the required fields
#             if "idRow" not in row or "data" not in row:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Row missing required fields (idRow or data)"
#                 )
            
#             rows_to_insert.append({
#                 "dataset_name": datasetName,
#                 "table_name": tableName,
#                 "row_id": row["idRow"],
#                 "data": row["data"],
#                 "status": "TODO"
#             })
        
#         if rows_to_insert:
#             input_data.insert_many(rows_to_insert)
        
#         # Update dataset's table count
#         dataset_trace.update_one(
#             {"dataset_name": datasetName},
#             {"$inc": {"total_tables": 1}}
#         )
        
#         return {
#             "message": "Table added successfully.",
#             "tableName": tableName,
#             "datasetName": datasetName
#         }
        
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error processing JSON: {str(e)}")

@router.post("/dataset/{datasetName}/table", status_code=status.HTTP_201_CREATED)
async def add_table(datasetName: str, table: TableItem, db: Database = Depends(get_db)):
    """
    Adds a new table to an existing dataset in the database.
    """
    try:
        # Check if dataset exists using GraphQL query
        dataset_info = graphql_query.get_dataset_info(dataset_name=datasetName)
        if not dataset_info:
            raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
    except Exception:
        # Fallback to direct database check
        dataset_trace = db["dataset_trace"]
        dataset = dataset_trace.find_one({"dataset_name": datasetName})
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
    
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
async def list_tables(
    datasetName: str, 
    cursor: Optional[str] = None, 
    limit: int = 10, 
    db: Database = Depends(get_db)
):
    """
    Retrieve a list of tables in dataset with pagination.
    Using GraphQL query for pagination.
    """
    try:
        # Check if dataset exists using GraphQL query
        dataset_info = graphql_query.get_dataset_info(dataset_name=datasetName)
        if not dataset_info:
            raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
        
        # Use GraphQL query to get tables in dataset
        result = graphql_query.get_tables_in_dataset(
            dataset_name=datasetName,
            cursor=cursor,
            page_size=limit,
            direction="next"
        )
        
        # Format the response to match the REST API structure
        tables_list = []
        for table in result.tables:
            tables_list.append({
                "tableName": table.table_name,
                "status": table.status
            })
        
        return {
            "tables": tables_list,
            "pagination": {
                "nextCursor": result.next_cursor,
                "previousCursor": result.previous_cursor
            }
        }
    except Exception as e:
        # Fallback to direct database query
        dataset_trace = db["dataset_trace"]
        dataset = dataset_trace.find_one({"dataset_name": datasetName})
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
        
        table_trace = db["table_trace"]
        query = {"dataset_name": datasetName}
        if cursor: 
            try:
                query["_id"] = {"$gt": ObjectId(cursor)}
            except:
                raise HTTPException(status_code=400, detail="Invalid cursor format")
        
        tables_cursor = table_trace.find(query).limit(limit + 1)
        tables = list(tables_cursor)

        has_next_page = len(tables) > limit
        if has_next_page:
            tables = tables[:limit]
        
        tables_list = []
        for table in tables:
            tables_list.append({
                "tableName": table["table_name"],
                "status": table["status"]
            })
        
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
async def get_table_results(
    datasetName: str, 
    tableName: str, 
    cursor: Optional[int] = None, 
    limit: int = 10, 
    db: Database = Depends(get_db)
):
    """
    Retrieve the processing results for a specific table, including entity linking.
    Using GraphQL query for pagination.
    """
    try:
        # First check if table exists by trying the GraphQL query
        table_info = graphql_query.get_table_info(table_name=tableName)
        if not table_info or table_info.dataset_name != datasetName:
            raise HTTPException(
                status_code=404, 
                detail=f"Table '{tableName}' not found in dataset '{datasetName}'"
            )
        
        # Get table data with pagination using GraphQL
        cursor_str = str(cursor) if cursor is not None else None
        result = graphql_query.get_table_data(
            dataset_name=datasetName,
            table_name=tableName,
            cursor=cursor_str,
            page_size=limit,
            direction="next"
        )
        
        # Get the table header (not always included in GraphQL response)
        table_trace = db["table_trace"]
        table = table_trace.find_one({
            "dataset_name": datasetName,
            "table_name": tableName
        })
        
        # Format the response to match the REST API structure
        formatted_rows = []
        for row in result.rows:
            # Build linked entities list (entity linking info might be in GraphQL response)
            # This is simplified and may need adjustment based on actual GraphQL response structure
            linked_entities = []
            
            # For simplicity, we'll query the database for el_results
            input_data = db["input_data"]
            db_row = input_data.find_one({
                "dataset_name": datasetName, 
                "table_name": tableName,
                "row_id": row.row_id
            })
            
            if db_row:
                el_results = db_row.get("el_results", {})
                for col_idx in range(len(row.row_data)):
                    col_key = str(col_idx)
                    if col_key in el_results and el_results[col_key]:
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
                        linked_entities.append({
                            "idColumn": col_idx,
                            "entity": None
                        })
            
            formatted_rows.append({
                "idRow": row.row_id,
                "data": row.row_data,
                "linked_entities": linked_entities
            })
        
        return {
            "data": {
                "datasetName": datasetName,
                "tableName": tableName,
                "header": table["header"] if table else [],
                "rows": formatted_rows
            },
            "pagination": {
                "nextCursor": result.next_cursor,
                "previousCursor": result.previous_cursor
            }
        }
    except Exception as e:
        # Fallback to direct database query
        dataset_trace = db["dataset_trace"]
        dataset = dataset_trace.find_one({"dataset_name": datasetName})
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{datasetName}' not found")
        
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
        
        input_data = db["input_data"]
        query = {
            "dataset_name": datasetName,
            "table_name": tableName
        }
        
        if cursor is not None:
            query["row_id"] = {"$gt": cursor}
        
        rows_cursor = input_data.find(query).sort("row_id", 1).limit(limit + 1)
        rows = list(rows_cursor)
        
        has_next_page = len(rows) > limit
        if has_next_page:
            rows = rows[:limit]
        
        formatted_rows = []
        for row in rows:
            linked_entities = []
            el_results = row.get("el_results", {})
            
            for col_idx in range(len(row["data"])):
                col_key = str(col_idx)
                if col_key in el_results and el_results[col_key]:
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
                    linked_entities.append({
                        "idColumn": col_idx,
                        "entity": None
                    })
            
            formatted_rows.append({
                "idRow": row["row_id"],
                "data": row["data"],
                "linked_entities": linked_entities
            })
        
        next_cursor = None
        previous_cursor = None
        
        if has_next_page and rows:
            next_cursor = str(rows[-1]["row_id"])
        
        if cursor is not None:
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
async def delete_table(
    datasetName: str, 
    tableName: str, 
    db: Database = Depends(get_db)
):
    """
    Delete a specific table from a dataset.
    """
    try:
        # Check if table exists using GraphQL
        table_info = graphql_query.get_table_info(table_name=tableName)
        if not table_info or table_info.dataset_name != datasetName:
            raise HTTPException(
                status_code=404, 
                detail=f"Table '{tableName}' not found in dataset '{datasetName}'"
            )
    except Exception:
        # Fallback to direct database check
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
    table_trace = db["table_trace"]
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
    dataset_trace = db["dataset_trace"]
    dataset_trace.update_one(
        {"dataset_name": datasetName},
        {"$inc": {"total_tables": -1}}
    )
    
    return {
        "message": f"Table '{tableName}' deleted from dataset '{datasetName}'."
    }
