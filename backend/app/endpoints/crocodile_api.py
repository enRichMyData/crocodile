import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Union
import base64

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
from pydantic import BaseModel
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError
from services.data_service import DataService
from services.result_sync import ResultSyncService
from services.utils import sanitize_for_json
from services.pagination_service import PaginationService
from services.query_service import QueryService
from services.table_service import TableService
from services.annotation_service import AnnotationService

router = APIRouter()

class TableUpload(BaseModel):
    table_name: str
    header: List[str]
    total_rows: int
    classified_columns: Optional[Dict[str, Dict[str, str]]] = {}
    data: List[dict]

class DatasetCreate(BaseModel):
    dataset_name: str

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

class AnnotationUpdate(BaseModel):
    """Request model for updating an annotation."""
    entity_id: str
    match: bool = True
    score: Optional[float] = 1.0
    notes: Optional[str] = None
    candidate_info: Optional[EntityCandidate] = None

@router.get("/datasets")
def get_datasets(
    limit: int = Query(10),
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    token_payload: Dict = Depends(verify_token),
    db: Database = Depends(get_db),
):
    user_id = token_payload.get("email")
    query_filter = {"user_id": user_id}
    try:
        query_filter, sort_direction = PaginationService.get_mongo_pagination(
            collection=db.datasets,
            query_filter=query_filter,
            next_cursor=next_cursor,
            prev_cursor=prev_cursor,
            limit=limit
        )
        results = db.datasets.find(query_filter).sort("_id", sort_direction).limit(limit + 1)
        datasets = list(results)
        datasets, next_cursor, prev_cursor = PaginationService.process_mongo_results(
            results=datasets,
            limit=limit,
            collection=db.datasets,
            query_filter=query_filter,
            sort_direction=sort_direction
        )
        for dataset in datasets:
            dataset["_id"] = str(dataset["_id"])
            if "created_at" in dataset:
                dataset["created_at"] = dataset["created_at"].isoformat()
        return {
            "data": datasets,
            "pagination": {"next_cursor": next_cursor, "prev_cursor": prev_cursor},
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/datasets", status_code=status.HTTP_201_CREATED)
def create_dataset(
    dataset_data: DatasetCreate = Body(...),
    token_payload: Dict = Depends(verify_token),
    db: Database = Depends(get_db),
):
    user_id = token_payload.get("email")
    existing = db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_data.dataset_name}
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset with dataset_name {dataset_data.dataset_name} already exists"
        )
    dataset_dict = dataset_data.dict()
    dataset_dict["created_at"] = datetime.now()
    dataset_dict["total_tables"] = 0
    dataset_dict["total_rows"] = 0
    dataset_dict["user_id"] = user_id
    try:
        result = db.datasets.insert_one(dataset_dict)
        dataset_dict["_id"] = str(result.inserted_id)
        return {"message": "Dataset created successfully", "dataset": dataset_dict}
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Dataset already exists")

@router.delete("/datasets/{dataset_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_name: str,
    token_payload: Dict = Depends(verify_token),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    user_id = token_payload.get("email")
    existing = db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_name}
    )
    if not existing:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    db.tables.delete_many({"user_id": user_id, "dataset_name": dataset_name})
    db.datasets.delete_one({"user_id": user_id, "dataset_name": dataset_name})
    crocodile_db.input_data.delete_many({"user_id": user_id, "dataset_name": dataset_name})
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
            refresh=True
        )
    except Exception as e:
        print(f"Error deleting dataset from Elasticsearch: {str(e)}")
    return None

@router.get("/datasets/{dataset_name}/tables")
def get_tables(
    dataset_name: str,
    limit: int = Query(10),
    next_cursor: Optional[str] = Query(None),
    prev_cursor: Optional[str] = Query(None),
    token_payload: Dict = Depends(verify_token),
    db: Database = Depends(get_db),
):
    user_id = token_payload.get("email")
    if not db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    query_filter = {"user_id": user_id, "dataset_name": dataset_name}
    try:
        query_filter, sort_direction = PaginationService.get_mongo_pagination(
            collection=db.tables,
            query_filter=query_filter,
            next_cursor=next_cursor,
            prev_cursor=prev_cursor,
            limit=limit
        )
        results = db.tables.find(query_filter).sort("_id", sort_direction).limit(limit + 1)
        tables = list(results)
        tables, next_cursor, prev_cursor = PaginationService.process_mongo_results(
            results=tables,
            limit=limit,
            collection=db.tables,
            query_filter=query_filter,
            sort_direction=sort_direction
        )
        for table in tables:
            table["_id"] = str(table["_id"])
            if "created_at" in table:
                table["created_at"] = table["created_at"].isoformat()
            if "completed_at" in table and table["completed_at"]:
                table["completed_at"] = table["completed_at"].isoformat()
        return {
            "dataset": dataset_name,
            "data": tables,
            "pagination": {"next_cursor": next_cursor, "prev_cursor": prev_cursor},
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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
    token_payload: Dict = Depends(verify_token),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    user_id = token_payload.get("email")
    if not db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    table = db.tables.find_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    )
    if not table:
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )
    header = table.get("header", [])
    table_types = table.get("column_types", {})
    classified_columns = table.get("classified_columns", {})
    if search is not None or include_types or exclude_types or sort_by is not None:
        try:
            search_after = None
            search_before = None
            if next_cursor and prev_cursor:
                raise HTTPException(status_code=400, detail="Only one of next_cursor or prev_cursor should be provided")
            
            # Check if sorting by confidence - in this case, we only support forward pagination
            is_confidence_sort = sort_by in ["confidence", "confidence_avg"]
            
            # For confidence sorting, don't allow backward pagination
            if is_confidence_sort and prev_cursor:
                raise HTTPException(
                    status_code=400, 
                    detail="Backward pagination is not supported when sorting by confidence"
                )
            
            if next_cursor:
                try:
                    search_after = PaginationService.parse_es_cursor(next_cursor)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
            elif prev_cursor and not is_confidence_sort:
                # Only process prev_cursor if NOT sorting by confidence
                try:
                    search_before = PaginationService.parse_es_cursor(prev_cursor)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
                    
            query_body = QueryService.build_es_table_query(
                user_id=user_id,
                dataset_name=dataset_name,
                table_name=table_name,
                search=search,
                search_columns=search_columns,
                column=column,
                include_types=include_types,
                exclude_types=exclude_types,
                sort_by=sort_by,
                sort_direction=sort_direction,
                search_after=search_after,
                search_before=search_before,
                limit=limit,
                es_index=ES_INDEX
            )
            res = es.search(index=ES_INDEX, body=query_body)
            hits = res["hits"]["hits"]
            total_hits = res["hits"]["total"]["value"]
            has_more = len(hits) > limit
            if has_more:
                hits = hits[:limit]
            if search_before:
                hits.reverse()
            row_ids = [hit["_source"]["row_id"] for hit in hits]
            new_next_cursor = new_prev_cursor = None
            if len(hits) > 0:
                if has_more:
                    last_hit = hits[-1]
                    new_next_cursor = PaginationService.create_es_cursor(last_hit["sort"])
                elif search_before:
                    first_hit = hits[0]
                    forward_query = {k: v for k, v in query_body.items() if k != "sort"}
                    forward_query["sort"] = [{k: {
                        "order": "desc" if v["order"] == "asc" else "asc"
                    } if isinstance(v, dict) and "order" in v else v 
                      for k, v in item.items()} 
                     for item in query_body["sort"]]
                    forward_query["search_after"] = first_hit["sort"]
                    forward_query["size"] = 1
                    forward_query["_source"] = False
                    check_res = es.search(index=ES_INDEX, body=forward_query)
                    if len(check_res["hits"]["hits"]) > 0:
                        new_next_cursor = PaginationService.create_es_cursor(first_hit["sort"])
                
                # Only compute previous cursor if NOT sorting by confidence
                if not is_confidence_sort:
                    first_hit = hits[0]
                    backward_query = {k: v for k, v in query_body.items() if k != "sort"}
                    backward_query["sort"] = [{k: {
                        "order": "desc" if v["order"] == "asc" else "asc"
                    } if isinstance(v, dict) and "order" in v else v 
                      for k, v in item.items()} 
                     for item in query_body["sort"]]
                    backward_query["search_after"] = first_hit["sort"]
                    backward_query["size"] = 1
                    backward_query["_source"] = False
                    try:
                        check_res = es.search(index=ES_INDEX, body=backward_query)
                        if len(check_res["hits"]["hits"]) > 0:
                            new_prev_cursor = PaginationService.create_es_cursor(first_hit["sort"])
                    except Exception as e:
                        print(f"Error checking for previous page: {str(e)}")
                    
                    if next_cursor:
                        new_prev_cursor = PaginationService.create_es_cursor(first_hit["sort"])
            
            rows_formatted = TableService.fetch_and_format_table_rows(
                db=db,
                crocodile_db=crocodile_db,
                user_id=user_id,
                dataset_name=dataset_name,
                table_name=table_name,
                row_ids=row_ids,
                header=header
            )
            status = TableService.get_table_status(
                db=db,
                crocodile_db=crocodile_db,
                user_id=user_id,
                dataset_name=dataset_name,
                table_name=table_name
            )
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
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error querying data: {str(e)}")
    else:
        query_filter = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
        }
        try:
            query_filter, sort_direction = PaginationService.get_mongo_pagination(
                collection=db.input_data,
                query_filter=query_filter,
                cursor_field="_id",
                next_cursor=next_cursor,
                prev_cursor=prev_cursor,
                limit=limit
            )
            results = db.input_data.find(query_filter).sort("row_id", sort_direction).limit(limit + 1)
            raw_rows = list(results)
            if not raw_rows:
                results = crocodile_db.input_data.find(query_filter).sort("row_id", sort_direction).limit(limit + 1)
                raw_rows = list(results)
            collections = [db.input_data, crocodile_db.input_data]
            for collection in collections:
                if raw_rows:
                    raw_rows, next_cursor, prev_cursor = PaginationService.process_mongo_results(
                        results=raw_rows,
                        limit=limit,
                        collection=collection,
                        query_filter=query_filter,
                        sort_direction=sort_direction
                    )
                    break
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
            status = TableService.get_table_status(
                db=db,
                crocodile_db=crocodile_db,
                user_id=user_id,
                dataset_name=dataset_name,
                table_name=table_name
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
            return sanitize_for_json(response_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

def parse_json_column_classification(column_classification: str = Form("")) -> Optional[dict]:
    if not column_classification:
        return None
    return json.loads(column_classification)

@router.post("/datasets/{datasetName}/tables/json", status_code=status.HTTP_201_CREATED)
def add_table(
    datasetName: str,
    table_upload: TableUpload = Body(..., example=IMDB_EXAMPLE),
    background_tasks: BackgroundTasks = None,
    token_payload: Dict = Depends(verify_token),
    db: Database = Depends(get_db),
):
    user_id = token_payload.get("email")
    try:
        df = pd.DataFrame(table_upload.data)
        classification = DataService.get_or_create_column_classification(
            data=df,
            header=table_upload.header,
            provided_classification=table_upload.classified_columns,
        )
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
        def run_crocodile_task():
            TableService.process_table_data_with_crocodile(
                df=df,
                user_id=user_id,
                dataset_name=datasetName,
                table_name=table_upload.table_name,
                classification=classification
            )
        def sync_results_task():
            time.sleep(5)
            mongo_uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
            sync_service = ResultSyncService(mongo_uri=mongo_uri)
            sync_service.sync_results(
                user_id=user_id, dataset_name=datasetName, table_name=table_upload.table_name
            )
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

@router.post("/datasets/{datasetName}/tables/csv", status_code=status.HTTP_201_CREATED)
def add_table_csv(
    datasetName: str,
    table_name: str = Form(...),
    file: UploadFile = File(...),
    column_classification: Optional[dict] = Depends(parse_json_column_classification),
    background_tasks: BackgroundTasks = None,
    token_payload: Dict = Depends(verify_token),
    db: Database = Depends(get_db),
):
    user_id = token_payload.get("email")
    try:
        df = pd.read_csv(file.file)
        df = df.replace({np.nan: None})
        header = df.columns.tolist()
        total_rows = len(df)
        classification = DataService.get_or_create_column_classification(
            data=df, header=header, provided_classification=column_classification
        )
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
        def run_crocodile_task():
            TableService.process_table_data_with_crocodile(
                df=df,
                user_id=user_id,
                dataset_name=datasetName,
                table_name=table_name,
                classification=classification
            )
        def sync_results_task():
            mongo_uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
            sync_service = ResultSyncService(mongo_uri=mongo_uri)
            sync_service.sync_results(
                user_id=user_id, dataset_name=datasetName, table_name=table_name
            )
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

@router.delete("/datasets/{dataset_name}/tables/{table_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_table(
    dataset_name: str,
    table_name: str,
    token_payload: Dict = Depends(verify_token),
    db: Database = Depends(get_db),
    crocodile_db: Database = Depends(get_crocodile_db),
):
    user_id = token_payload.get("email")
    dataset = db.datasets.find_one(
        {"user_id": user_id, "dataset_name": dataset_name}
    )
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
    db.tables.delete_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    )
    db.datasets.update_one(
        {"user_id": user_id, "dataset_name": dataset_name},
        {"$inc": {"total_tables": -1, "total_rows": -row_count}},
    )
    crocodile_db.input_data.delete_many(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    )
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
            refresh=True
        )
    except Exception as e:
        print(f"Error deleting table from Elasticsearch: {str(e)}")
    return None

@router.put("/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}")
def update_annotation(
    dataset_name: str,
    table_name: str,
    row_id: int,
    column_id: int,
    annotation: AnnotationUpdate,
    token_payload: Dict = Depends(verify_token),
    db: Database = Depends(get_db),
):
    user_id = token_payload.get("email")
    if not db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    if not db.tables.find_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    ):
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )
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
    el_results = row.get("el_results", {})
    column_candidates = el_results.get(str(column_id), [])
    if not column_candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No entity linking candidates found for column {column_id} in row {row_id}",
        )
    try:
        updated_candidates, matched_candidate = AnnotationService.update_candidate_list(
            candidates=column_candidates,
            annotation=annotation
        )
        el_results[str(column_id)] = updated_candidates
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}/candidates/{entity_id}")
def delete_candidate(
    dataset_name: str,
    table_name: str,
    row_id: int,
    column_id: int,
    entity_id: str,
    token_payload: Dict = Depends(verify_token),
    db: Database = Depends(get_db),
):
    user_id = token_payload.get("email")
    if not db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name}):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    if not db.tables.find_one(
        {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
    ):
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )
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
    el_results = row.get("el_results", {})
    column_candidates = el_results.get(str(column_id), [])
    if not column_candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No entity linking candidates found for column {column_id} in row {row_id}",
        )
    try:
        updated_candidates = AnnotationService.delete_candidate(
            candidates=column_candidates,
            entity_id=entity_id
        )
        el_results[str(column_id)] = updated_candidates
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
