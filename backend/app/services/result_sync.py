import time
from datetime import datetime
from typing import Any, Dict, List
from collections import defaultdict, Counter

from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from services.utils import log_error, log_info

from crocodile import CrocodileResultFetcher


class ResultSyncService:
    """
    Service for synchronizing entity linking results from Crocodile to the backend database.

    This service handles:
    - Retrieving entity linking results from the Crocodile processing engine
    - Updating the backend database with the latest results
    - Tracking completion status and reporting progress
    """

    def __init__(
        self,
        mongo_uri: str = "mongodb://mongodb:27017",
        backend_db_name: str = "crocodile_backend_db",
        batch_size: int = 50,  # Reduced from 100 to prevent memory issues
    ):
        """
        Initialize the ResultSyncService.

        Args:
            mongo_uri: MongoDB connection URI
            backend_db_name: Name of the backend database
            batch_size: Number of rows to process in each batch
        """
        self.mongo_uri = mongo_uri
        self.backend_db_name = backend_db_name
        self.batch_size = batch_size

    def _get_client(self) -> MongoClient:
        """Get a MongoDB client connection with timeout settings."""
        return MongoClient(
            self.mongo_uri,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=5000,
            socketTimeoutMS=30000,  # 30 second socket timeout
            maxPoolSize=10,  # Limit connection pool size
        )

    def _update_table_status(
        self,
        db,
        user_id: str,
        dataset_name: str,
        table_name: str,
        completed_count: int,
        total_count: int,
        error: Exception = None,
    ) -> Dict[str, Any]:
        """
        Update the table status based on completion percentage.

        Args:
            db: MongoDB database connection
            user_id: User ID
            dataset_name: Dataset name
            table_name: Table name
            completed_count: Number of completed rows
            total_count: Total number of rows
            error: Optional exception if an error occurred

        Returns:
            Dictionary with status information
        """
        try:
            completion_percentage = completed_count / total_count if total_count > 0 else 0

            # Determine status based on completion and errors
            if error:
                table_status = "sync_error"
                status_message = f"Error: {str(error)}"
            else:
                table_status = (
                    "DONE" if completion_percentage >= 0.95 else "DOING"
                )
                status_message = f"Sync {table_status}"

            # Update the table record
            db.tables.update_one(
                {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name},
                {
                    "$set": {
                        "status": table_status,
                        "completion_percentage": round(completion_percentage * 100, 2),
                        "last_synced": datetime.now(),
                        **({"error": str(error)} if error else {}),
                    }
                },
            )

            return {
                "status": "error" if error else "success",
                "message": status_message,
                "table_status": table_status,
                "completion_percentage": round(completion_percentage * 100, 2),
                **({"error": str(error)} if error else {}),
            }

        except Exception as e:
            log_error("Failed to update table status", e)
            return {
                "status": "error",
                "message": f"Failed to update table status: {str(e)}",
                "error": str(e),
            }

    def sync_results(self, user_id: str, dataset_name: str, table_name: str) -> Dict[str, Any]:
        """
        Sync entity linking results from Crocodile to the backend database.
        Uses a simpler batch-based approach to process rows in manageable chunks.

        Args:
            user_id: User ID for multi-tenant isolation
            dataset_name: Name of the dataset
            table_name: Name of the table

        Returns:
            Dict with sync results summary
        """
        log_info(f"Starting result sync for {user_id}/{dataset_name}/{table_name}")

        # Delay to wait a bit before starting the sync
        time.sleep(10)

        # Create a MongoDB connection
        client = self._get_client()
        db = client[self.backend_db_name]

        try:
            # Verify the table exists
            table = db.tables.find_one(
                {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
            )

            if not table:
                raise ValueError(f"Table {dataset_name}/{table_name} not found")

            # Get table header for cell processing
            header = table.get("header", [])
            
            # Count rows in the table for status tracking
            total_count = db.input_data.count_documents(
                {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
            )

            if total_count == 0:
                log_info(f"No rows found for {dataset_name}/{table_name}")
                return {
                    "status": "success",
                    "message": "No rows to sync",
                    "completion_percentage": 0,
                }

            log_info(f"Found {total_count} rows for {dataset_name}/{table_name}")

            # Create a CrocodileResultFetcher instance
            result_fetcher = CrocodileResultFetcher(
                client_id=user_id,
                dataset_name=dataset_name,
                table_name=table_name,
                mongo_uri=self.mongo_uri,
            )

            # Track progress
            completed_count = 0
            batch_size = self.batch_size
           
            # Dictionary to collect type frequencies by column - using type_id as key
            column_type_frequencies = defaultdict(lambda: defaultdict(int))
            column_type_mapping = {}  # Maps type_id to {id, name} for later reference

            # Process all rows in batches, focusing on incomplete ones
            input_collection = db.input_data

            last_remaining_count = None
            last_update_time = time.time()
            update_timeout = 300  # Reduced from 600 to 5 minutes

            while True:
                cursor = input_collection.find(
                    {
                        "user_id": user_id,
                        "dataset_name": dataset_name,
                        "table_name": table_name,
                        "$or": [
                            {"status": {"$ne": "DONE"}},
                            {"ml_status": {"$ne": "DONE"}},
                        ]
                    },
                    {"row_id": 1},
                    no_cursor_timeout=True
                ).batch_size(self.batch_size)  # Add batch size to cursor

                batch_ids = []
                for doc in cursor:
                    batch_ids.append(doc["row_id"])
                    if len(batch_ids) >= batch_size:
                        self._process_batch(
                            batch_ids, result_fetcher, db, user_id, dataset_name, table_name,
                            column_type_frequencies, column_type_mapping, header
                        )
                        last_update_time = time.time()
                        batch_ids = []

                if batch_ids:
                    self._process_batch(
                        batch_ids, result_fetcher, db, user_id, dataset_name, table_name,
                        column_type_frequencies, column_type_mapping, header
                    )
                    last_update_time = time.time()

                # After processing all batches in this loop iteration, update progress
                final_completed_count = db.input_data.count_documents(
                    {
                        "user_id": user_id,
                        "dataset_name": dataset_name,
                        "table_name": table_name,
                        "status": "DONE",
                        "ml_status": "DONE",
                    }
                )
                log_info(
                    f"Processed batch, completed count: {final_completed_count} out of {total_count}"
                )
                self._update_table_status(
                    db, user_id, dataset_name, table_name, final_completed_count, total_count
                )

                # Check if there are still incomplete rows
                remaining = input_collection.count_documents({
                    "user_id": user_id,
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "$or": [
                        {"status": {"$ne": "DONE"}},
                        {"ml_status": {"$ne": "DONE"}},
                    ]
                })

                # Timeout logic based on unchanged remaining count
                if remaining == last_remaining_count:
                    if time.time() - last_update_time > update_timeout:
                        log_info("Remaining count hasn't changed in 5 minutes, exiting sync loop")
                        break
                else:
                    last_remaining_count = remaining
                    last_update_time = time.time()

                if remaining == 0:
                    break
                    
                # Add small delay to prevent overwhelming the database
                time.sleep(0.1)

            # After processing all rows, update the table with type frequencies
            if column_type_frequencies:
                column_type_summary = {}
                for col_idx, type_counter in column_type_frequencies.items():
                    # Convert Counter to a sorted list with normalized frequencies and type information
                    type_info = []
                    for type_id, count in type_counter.items():
                        # Calculate normalized frequency by dividing by total_count and clamp into [0,1]
                        raw_freq = count / total_count if total_count > 0 else 0.0
                        frequency = max(0.0, min(raw_freq, 1.0))
                        
                        # Filter out types with very low frequency
                        if frequency < 0.01:
                            continue
                        
                        # Get the full type info from our mapping
                        type_data = column_type_mapping.get(type_id, {"name": "unknown"})
                        
                        # Create the type entry with id, name and frequency
                        type_entry = {
                            "frequency": frequency
                        }
                        
                        # Add ID and name if available
                        if "id" in type_data:
                            type_entry["id"] = type_data["id"]
                        if "name" in type_data:
                            type_entry["name"] = type_data["name"]
                        
                        type_info.append(type_entry)
                    
                    # Sort types by frequency in descending order
                    type_info.sort(key=lambda x: x["frequency"], reverse=True)
                    
                    column_type_summary[str(col_idx)] = {
                        "types": type_info,
                        "total_count": total_count,
                    }
                
                # Update the table document with type frequencies
                db.tables.update_one(
                    {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name},
                    {"$set": {"column_types": column_type_summary}}
                )
                log_info(f"Updated table {dataset_name}/{table_name} with type frequencies including IDs and names")

            # Do a final count of completed rows to get accurate completion percentage
            final_completed_count = db.input_data.count_documents(
                {
                    "user_id": user_id,
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "status": "DONE",
                    "ml_status": "DONE",
                }
            )

        except Exception as e:
            log_error("Sync process terminated with error", e)

            # Try to get current completion status for error reporting
            try:
                completed_count = db.input_data.count_documents(
                    {
                        "user_id": user_id,
                        "dataset_name": dataset_name,
                        "table_name": table_name,
                        "status": "DONE",
                        "ml_status": "DONE",
                    }
                )

                total_count = db.input_data.count_documents(
                    {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name}
                )
            except Exception:
                completed_count = 0
                total_count = 1  # Avoid division by zero

            # Update table with error information
            result = self._update_table_status(
                db, user_id, dataset_name, table_name, completed_count, total_count, e
            )
            return result

        finally:
            # Close MongoDB connection
            client.close()
            log_info(f"Sync process finished for {dataset_name}/{table_name}")

    def _process_batch(
        self,
        batch_ids: List[Any],
        result_fetcher: Any,
        db: Any,
        user_id: str,
        dataset_name: str,
        table_name: str,
        column_type_frequencies: Any,
        column_type_mapping: Any,
        header: List[str],
    ):
        """
        Process a batch of row_ids: fetch results, update backend db and populate cell_data collection.
        """
        try:
            results = result_fetcher.get_results(batch_ids)
            if not results:
                log_info(f"No results found for batch with {len(batch_ids)} row IDs")
                return

            # Prepare bulk operations for both collections
            input_data_updates = []
            cell_data_operations = []

            for result in results:
                row_id = result.get("row_id")
                status = result.get("status")
                ml_status = result.get("ml_status")
                el_results = result.get("el_results", {})

                confidence_scores = []
                
                # Get the original row data for cell processing
                row_doc = db.input_data.find_one({
                    "user_id": user_id,
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "row_id": row_id,
                })
                
                if not row_doc:
                    continue
                    
                row_data = row_doc.get("data", [])

                for col_idx, candidates in el_results.items():
                    col_index = int(col_idx)
                    
                    # Get cell text value
                    cell_text = ""
                    if col_index < len(row_data) and row_data[col_index] is not None:
                        cell_text = str(row_data[col_index])
                    
                    # Process candidates for input_data update
                    if candidates and len(candidates) > 0:
                        top_candidate = candidates[0]
                        confidence = top_candidate.get("score", 0.0)
                        if confidence is not None:
                            confidence_scores.append(confidence)
                        
                        # Extract types for the column
                        types = []
                        if "types" in top_candidate:
                            for type_obj in top_candidate["types"]:
                                if isinstance(type_obj, dict) and "name" in type_obj and "id" in type_obj:
                                    type_id = type_obj["id"]
                                    type_name = type_obj["name"]
                                    types.append(type_id)
                                    column_type_frequencies[col_index][type_id] += 1
                                    column_type_mapping[type_id] = {"id": type_id, "name": type_name}

                        # Create cell_data document for this cell
                        cell_doc = {
                            "user_id": user_id,
                            "dataset_name": dataset_name,
                            "table_name": table_name,
                            "row_id": row_id,
                            "col_id": col_index,
                            "cell_text": cell_text,
                            "confidence": confidence,
                            "types": types,
                            "last_updated": datetime.now(),
                        }
                        
                        # Use upsert to replace existing cell data
                        cell_data_operations.append(
                            UpdateOne(
                                {
                                    "user_id": user_id,
                                    "dataset_name": dataset_name,
                                    "table_name": table_name,
                                    "row_id": row_id,
                                    "col_id": col_index,
                                },
                                {"$set": cell_doc},
                                upsert=True
                            )
                        )

                row_avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

                # Prepare simplified input_data update - remove flattened fields
                update_dict = {
                    "status": status,
                    "ml_status": ml_status,
                    "el_results": el_results,
                    "last_updated": datetime.now(),
                    "avg_confidence": row_avg_confidence,  # Keep this for row-level sorting
                }

                input_data_updates.append(
                    UpdateOne(
                        {
                            "user_id": user_id,
                            "dataset_name": dataset_name,
                            "table_name": table_name,
                            "row_id": row_id,
                        },
                        {"$set": update_dict},
                        upsert=False,
                    )
                )

            # Execute bulk operations
            if input_data_updates:
                db.input_data.bulk_write(input_data_updates)
                
            if cell_data_operations:
                db.cell_data.bulk_write(cell_data_operations)
                log_info(f"Updated {len(cell_data_operations)} cells in cell_data collection")

        except Exception as e:
            log_error("Error syncing results for batch", e)
