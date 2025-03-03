from pymongo import MongoClient
from bson import ObjectId
from typing import Optional, Any, Dict, List
import logging
import datetime


class DatabaseManager:
    """Handles all database operations for datasets and tables."""

    def __init__(self, mongo_uri: str, db_name: str) -> None:
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.dataset_collection = self.db["dataset_trace"]
        self.table_collection = self.db["table_trace"]
        self.input_data_collection = self.db["input_data"]
        self.status_collection = self.db["dataset_status"]  # ✅ New collection to track status

    def create_dataset(self, dataset_name: str, tables: List[str]) -> Dict[str, Any]:
        """Creates a new dataset with tables."""
        if self.dataset_collection.find_one({"dataset_name": dataset_name}):
            return {"error": "Dataset already exists"}

        dataset_id = self.dataset_collection.insert_one({
            "dataset_name": dataset_name,
            "tables": tables
        }).inserted_id

        return {"message": "Dataset created", "id": str(dataset_id)}

    def get_datasets(
        self, 
        cursor: Optional[str] = None, 
        page_size: int = 10,
        dataset_name: Optional[str] = None  # New parameter
    ) -> Dict[str, Any]:
        """Retrieves a paginated list of datasets with optional dataset name filter."""
        try:
            # Base query
            query = {}
            if cursor:
                query["_id"] = {"$gt": ObjectId(cursor)}
            if dataset_name:
                query["dataset_name"] = dataset_name
            
            # Get datasets from dataset_collection
            datasets = list(self.dataset_collection.find(query).sort("_id", 1).limit(page_size + 1))
            
            # Process each dataset to include tables and status
            processed_datasets = []
            for dataset in datasets[:page_size]:
                dataset_name = dataset.get("dataset_name")
                
                # Get tables for this dataset
                tables = list(self.table_collection.find(
                    {"dataset_name": dataset_name},
                    {"table_name": 1, "_id": 0}
                ))
                table_names = [table["table_name"] for table in tables]
                
                # Get current status
                current_status = self.check_and_update_dataset_status(dataset_name)
                
                processed_datasets.append({
                    "dataset_name": dataset_name,
                    "tables": table_names,
                    "status": current_status
                })
            
            # Determine if there are more datasets
            has_more = len(datasets) > page_size
            next_cursor = str(datasets[page_size]["_id"]) if has_more else None
            
            return {
                "datasets": processed_datasets,
                "next_cursor": next_cursor
            }
            
        except Exception as e:
            logging.error(f"Error getting datasets: {e}")
            return {
                "datasets": [],
                "next_cursor": None
            }

    def delete_dataset(self, dataset_name: str) -> Dict[str, str]:
        """Deletes a dataset and its associated tables."""
        if not self.dataset_collection.find_one({"dataset_name": dataset_name}):
            return {"error": "Dataset not found"}

        self.dataset_collection.delete_one({"dataset_name": dataset_name})
        self.table_collection.delete_many({"dataset_name": dataset_name})
        self.input_data_collection.delete_many({"dataset_name": dataset_name})

        return {"message": "Dataset and its tables deleted"}

    def add_table(self, dataset_name: str, table_name: str) -> Dict[str, str]:
        """Adds a table to an existing dataset."""
        if not self.dataset_collection.find_one({"dataset_name": dataset_name}):
            return {"error": "Dataset not found"}

        self.table_collection.insert_one({"dataset_name": dataset_name, "table_name": table_name})
        return {"message": "Table added to dataset"}

    def get_tables(self, dataset_name: str, cursor: Optional[str] = None, page_size: int = 5) -> Dict[str, Any]:
        """Retrieves a paginated list of tables in a dataset."""
        query = {"dataset_name": dataset_name}
        if cursor:
            query["_id"] = {"$gt": ObjectId(cursor)}

        tables = list(self.table_collection.find(query).sort("_id", 1).limit(page_size + 1))

        return {
            "tables": tables[:page_size],
            "next_cursor": str(tables[-1]["_id"]) if len(tables) == page_size else None
        }

    def get_table_results(self, dataset_name: str, table_name: str, cursor: Optional[str] = None, page_size: int = 5) -> Dict[str, Any]:
        """Retrieves paginated table data."""
        query = {"dataset_name": dataset_name, "table_name": table_name}
        if cursor:
            query["_id"] = {"$gt": ObjectId(cursor)}

        results = list(self.input_data_collection.find(query).sort("_id", 1).limit(page_size + 1))

        return {
            "results": results[:page_size],
            "next_cursor": str(results[-1]["_id"]) if len(results) == page_size else None
        }

    def delete_table(self, dataset_name: str, table_name: str) -> Dict[str, str]:
        """Deletes a table from a dataset."""
        if not self.table_collection.find_one({"dataset_name": dataset_name, "table_name": table_name}):
            return {"error": "Table not found"}

        self.table_collection.delete_one({"dataset_name": dataset_name, "table_name": table_name})
        self.input_data_collection.delete_many({"dataset_name": dataset_name, "table_name": table_name})

        return {"message": "Table deleted"}

    # newly added

    def get_dataset_metadata(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific dataset.
        Ensures classified_columns is always returned as a list.
        """
        try:
            # First try to get from table_collection which contains more metadata
            metadata = self.table_collection.find_one(
                {"dataset_name": dataset_name},
                {
                    "dataset_name": 1, 
                    "table_name": 1, 
                    "header": 1,
                    "classified_columns": 1,
                    "total_rows": 1,
                    "start_time": 1,
                    "_id": 0
                }
            )
            
            if not metadata:
                # Fallback to dataset_collection
                metadata = self.dataset_collection.find_one(
                    {"dataset_name": dataset_name},
                    {
                        "dataset_name": 1,
                        "total_rows": 1,
                        "start_time": 1,
                        "_id": 0
                    }
                )
            
            if metadata:
                # Ensure classified_columns is always a list
                if "classified_columns" in metadata:
                    if not isinstance(metadata["classified_columns"], list):
                        # If it's a string (comma-separated), convert to list
                        if isinstance(metadata["classified_columns"], str):
                            metadata["classified_columns"] = [col.strip() for col in metadata["classified_columns"].split(',')]
                        # If it's some other type, wrap in a list
                        else:
                            metadata["classified_columns"] = [str(metadata["classified_columns"])]
                else:
                    # Set default empty list if missing
                    metadata["classified_columns"] = []
                    
            return metadata
        except Exception as e:
            logging.error(f"Error getting dataset metadata: {e}")
            return None

    def get_table_results_with_annotations(
        self, 
        dataset_name: str, 
        table_name: str, 
        cursor: Optional[str] = None, 
        page_size: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieves paginated table data with semantic annotations.
        """
        try:
            query = {"dataset_name": dataset_name, "table_name": table_name}
            if cursor:
                query["_id"] = {"$gt": ObjectId(cursor)}

            # Get basic results
            results = list(self.input_data_collection.find(query).sort("_id", 1).limit(page_size + 1))
            
            # If no results, return early
            if not results:
                return {"results": [], "next_cursor": None}
                
            # Get the results we'll return (all but the last one if we have more than page_size)
            return_results = results[:page_size]
            
            # Add semantic annotations from crocodile_results collection
            for result in return_results:
                row_id = result.get("row_id")
                
                el_results = result.get("el_results", {})
                semantic_annotations = []
                # Iterate over each column key in el_results
                for column, annotations in el_results.items():
                    for annotation in annotations:
                        types = annotation.get("types", [])
                        # Join all type names with a comma
                        entity_type = ", ".join([t.get("name", "") for t in types])
                        semantic_annotations.append({
                            "entity_id": annotation.get("id", ""),
                            "entity_type": entity_type,  
                            "entity_name": annotation.get("name", ""),
                            "confidence_score": annotation.get("score", 0.0),
                            "source_column": column,  
                            "row_index": row_id
                        })
                result["semantic_annotations"] = semantic_annotations
            
            # Determine if there are more results
            has_more = len(results) > page_size
            next_cursor = str(results[page_size]["_id"]) if has_more else None
            
            return {
                "results": return_results,
                "next_cursor": next_cursor
            }
        except Exception as e:
            logging.error(f"Error getting table results with annotations: {e}")
            return {"results": [], "next_cursor": None}    

    def set_todo_status(self, dataset_name: str, table_name: str):
        """
        Updates MongoDB to mark all records of the specified dataset as 'TODO'.
        """
        logging.info(f"🔍 Setting TODO status for dataset: {dataset_name}, table: {table_name}")
        
        # First, set all other datasets to SKIPPED
        self.input_data_collection.update_many(
            {"dataset_name": {"$ne": dataset_name}},
            {"$set": {"status": "SKIPPED"}}
        )
        
        # Then set all records of this dataset to TODO
        result = self.input_data_collection.update_many(
            {"dataset_name": dataset_name},
            {"$set": {
                "status": "TODO",
                "table_name": table_name,
            }}
        )
        
        logging.info(f"✅ Updated {result.modified_count} records to TODO status")
        
        # Verify the update
        todo_count = self.input_data_collection.count_documents({
            "dataset_name": dataset_name,
            "status": "TODO"
        })
        logging.info(f"✅ Verified {todo_count} records are now in TODO status")
        
        return result


    def insert_dataset_status(self, dataset_name, status):
        """Insert or update dataset status."""
        self.status_collection.update_one(
            {"dataset_name": dataset_name},
            {"$set": {"status": status}},
            upsert=True
        )

    def check_and_update_dataset_status(self, dataset_name: str) -> str:
        """
        Check the actual processing status based on input_data records and update status accordingly.
        """
        logging.info(f"Checking actual status for dataset: {dataset_name}")
        
        # Get counts for different statuses
        todo_count = self.input_data_collection.count_documents({
            "dataset_name": dataset_name,
            "status": "TODO"
        })
        
        done_count = self.input_data_collection.count_documents({
            "dataset_name": dataset_name,
            "status": "DONE"
        })
        
        total_count = self.input_data_collection.count_documents({
            "dataset_name": dataset_name
        })
        
        logging.info(f"Status counts - TODO: {todo_count}, DONE: {done_count}, Total: {total_count}")
        
        # Determine status based on counts
        if total_count == 0:
            new_status = "unknown"
        elif done_count == total_count:
            new_status = "completed"
        elif todo_count == 0 and done_count < total_count:
            new_status = "partially_completed"
        elif todo_count > 0:
            new_status = "processing"
        else:
            new_status = "processing"
            
        logging.info(f"Determined status: {new_status}")
        
        # Update status in both collections
        self.status_collection.update_one(
            {"dataset_name": dataset_name},
            {
                "$set": {
                    "status": new_status,
                    "last_updated": datetime.datetime.utcnow(),
                    "stats": {
                        "total": total_count,
                        "done": done_count,
                        "todo": todo_count
                    }
                }
            },
            upsert=True
        )
        
        self.dataset_collection.update_one(
            {"dataset_name": dataset_name},
            {
                "$set": {
                    "status": new_status,
                    "last_updated": datetime.datetime.utcnow()
                }
            },
            upsert=True
        )
        
        return new_status

    def get_status(self, dataset_name: str) -> str:
        """
        Enhanced get_status that checks actual processing status.
        """
        # First check actual status
        current_status = self.check_and_update_dataset_status(dataset_name)
        return current_status

    def update_status(self, dataset_name, status):
        """Update status in both collections to maintain consistency."""
        # Update in status collection
        self.status_collection.update_one(
            {"dataset_name": dataset_name},
            {"$set": {"status": status}},
            upsert=True
        )
        
        # Also update in dataset collection
        self.dataset_collection.update_one(
            {"dataset_name": dataset_name},
            {"$set": {"status": status}},
            upsert=True
        )
        
        logging.info(f"Status updated to {status} for dataset {dataset_name}")         

# curl -X POST http://localhost:8006/upload-dataset/ \
#   -F "csv_file=@/Users/ashishr/Desktop/enRichmyData/GetMethod/imdb_top_1000.csv" \
#   -F "metadata_file=@/Users/ashishr/Desktop/enRichmyData/GetMethod/imdb_metadata.json"
