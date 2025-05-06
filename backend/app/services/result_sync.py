import time
from datetime import datetime
from typing import Any, Dict, List
from collections import defaultdict, Counter

from pymongo import MongoClient
from pymongo.collection import Collection
from services.utils import log_error, log_info
from dependencies import es, ES_INDEX

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
        batch_size: int = 100,
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
        """Get a MongoDB client connection."""
        return MongoClient(self.mongo_uri)

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
                    "completed" if completion_percentage >= 0.95 else "partially_completed"
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
            last_progress_time = time.time()
            consecutive_failures = 0
            max_failures = 5  # Safety limit for consecutive failures

            # Dictionary to collect type frequencies by column
            column_type_frequencies = defaultdict(Counter)

            # Process all rows in batches, focusing on incomplete ones
            input_collection = db.input_data

            # First, try to get rows that are not yet complete
            while True:
                # Get a batch of rows that are not completed
                incomplete_rows = self._get_incomplete_batch(
                    input_collection, user_id, dataset_name, table_name, batch_size
                )

                # If no more incomplete rows, we're done
                if not incomplete_rows:
                    log_info(f"All rows have been synced for {dataset_name}/{table_name}")
                    break

                # Extract the row IDs from the batch
                batch_ids = [row["row_id"] for row in incomplete_rows]

                try:
                    # Fetch the latest results from Crocodile
                    results = result_fetcher.get_results(batch_ids)

                    if not results:
                        # No results for this batch, move to the next one
                        log_info(f"No results found for batch with {len(batch_ids)} row IDs")
                        time.sleep(5)  # Wait a bit before trying another batch
                        continue

                    # Prepare bulk operations for Elasticsearch updates
                    es_operations = []

                    # Process each result
                    for result in results:
                        row_id = result.get("row_id")
                        status = result.get("status")
                        ml_status = result.get("ml_status")
                        el_results = result.get("el_results", {})

                        # Skip if no results yet
                        if not el_results:
                            continue

                        # Calculate row-level average confidence score
                        confidence_scores = []
                        # Prepare data updates with types from each column's top candidate
                        data_updates = []
                        column_confidence_scores = {}  # Track confidence by column

                        for col_idx, candidates in el_results.items():
                            if candidates and len(candidates) > 0:
                                # Get the top candidate and its confidence score
                                top_candidate = candidates[0]
                                confidence = top_candidate.get("score", 0.0)
                                
                                # Track score if not None for row average
                                if confidence is not None:
                                    confidence_scores.append(confidence)
                                    column_confidence_scores[col_idx] = confidence
                                
                                # Extract all types from the top candidate
                                entity_types = []
                                if "types" in top_candidate:
                                    for type_obj in top_candidate["types"]:
                                        if isinstance(type_obj, dict) and "name" in type_obj:
                                            type_name = type_obj["name"]
                                            entity_types.append(type_name)
                                            # Count for type frequencies
                                            column_type_frequencies[col_idx][type_name] += 1

                                # Prepare update with types and confidence score
                                if entity_types or confidence is not None:
                                    update = {
                                        "col_index": int(col_idx),
                                    }
                                    if entity_types:
                                        update["types"] = entity_types
                                    if confidence is not None:
                                        update["confidence"] = confidence
                                    data_updates.append(update)

                        # Calculate row average confidence
                        row_avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

                        # Update the backend database with confidence scores
                        db.input_data.update_one(
                            {
                                "user_id": user_id,
                                "dataset_name": dataset_name,
                                "table_name": table_name,
                                "row_id": row_id,
                            },
                            {
                                "$set": {
                                    "status": status,
                                    "ml_status": ml_status,
                                    "el_results": el_results,
                                    "last_updated": datetime.now(),
                                    "confidence_scores": column_confidence_scores,
                                    "avg_confidence": row_avg_confidence
                                }
                            },
                            upsert=False,
                        )

                        # Update type information in ES and collect type frequencies
                        doc_id = f"{user_id}_{dataset_name}_{table_name}_{row_id}"

                        # Create an update script if we have any type or confidence data
                        if data_updates:
                            # Use a script to update types and confidence scores
                            update_script = {
                                "script": {
                                    "source": """
                                    for (def update : params.updates) {
                                        for (int i = 0; i < ctx._source.data.length; i++) {
                                            if (ctx._source.data[i].col_index == update.col_index) {
                                                if (update.containsKey('types')) {
                                                    ctx._source.data[i].types = update.types;
                                                }
                                                if (update.containsKey('confidence')) {
                                                    ctx._source.data[i].confidence = update.confidence;
                                                }
                                            }
                                        }
                                    }
                                    // Add the row's average confidence
                                    if (!ctx._source.containsKey('avg_confidence')) {
                                        ctx._source.avg_confidence = params.avg_confidence;
                                    } else {
                                        ctx._source.avg_confidence = params.avg_confidence;
                                    }
                                    """,
                                    "params": {
                                        "updates": data_updates,
                                        "avg_confidence": row_avg_confidence
                                    }
                                }
                            }

                            es_operations.append({"update": {"_index": ES_INDEX, "_id": doc_id}})
                            es_operations.append(update_script)

                        # Count as completed if done
                        if status == "DONE" and ml_status == "DONE":
                            completed_count += 1

                    # Execute bulk ES updates if we have operations
                    if es_operations:
                        try:
                            resp = es.bulk(body=es_operations, refresh=True)
                            if resp.get("errors"):
                                log_error(f"Errors in bulk update: {resp.get('items')}")
                            else:
                                log_info(f"Updated {len(es_operations)//2} documents with type information")
                        except Exception as e:
                            log_error(f"Failed to update ES: {str(e)}", e)

                    # We made progress
                    consecutive_failures = 0
                    last_progress_time = time.time()
                    log_info(
                        f"Processed batch with {len(results)} results, completed {completed_count} rows so far"
                    )

                except Exception as e:
                    log_error("Error syncing results for batch", e)
                    consecutive_failures += 1

                    # Break out if we've had too many consecutive failures
                    if consecutive_failures >= max_failures:
                        log_error(f"Too many consecutive failures ({max_failures}), stopping sync")
                        raise

                # Check if we haven't made progress in a long time (5 minutes)
                if time.time() - last_progress_time > 300:
                    log_info("No progress for 5 minutes, completing sync process")
                    break

                # Brief pause between batches to avoid hammering the database
                time.sleep(2)

            # After processing all rows, update the table with type frequencies
            if column_type_frequencies:
                column_type_summary = {}
                for col_idx, type_counter in column_type_frequencies.items():
                    # Get the total count of types for this column for normalization
                    total_count = sum(type_counter.values())
                    
                    # Convert Counter to a sorted list with normalized frequencies only
                    type_info = []
                    for type_name, count in type_counter.most_common():
                        # Calculate normalized frequency between 0 and 1
                        frequency = count / total_count if total_count > 0 else 0
                        
                        type_info.append({
                            "name": type_name,
                            "frequency": frequency  # Only keep frequency
                        })
                    
                    column_type_summary[str(col_idx)] = {
                        "types": type_info
                    }
                
                # Update the table document with type frequencies
                db.tables.update_one(
                    {"user_id": user_id, "dataset_name": dataset_name, "table_name": table_name},
                    {"$set": {"column_types": column_type_summary}}
                )
                log_info(f"Updated table {dataset_name}/{table_name} with type frequencies")

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

            # Update table status based on completion
            result = self._update_table_status(
                db, user_id, dataset_name, table_name, final_completed_count, total_count
            )

            completion_percentage = final_completed_count / total_count if total_count > 0 else 0
            log_info(
                f"""Marked table {dataset_name}/{table_name} as
                    {result['table_status']} ({completion_percentage:.1%} complete)"""
            )

            return result

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

    def _get_incomplete_batch(
        self,
        collection: Collection,
        user_id: str,
        dataset_name: str,
        table_name: str,
        batch_size: int,
    ) -> List[Dict]:
        """
        Get a batch of rows that are not completed.

        Args:
            collection: MongoDB collection
            user_id: User ID
            dataset_name: Dataset name
            table_name: Table name
            batch_size: Size of the batch to retrieve

        Returns:
            List of row documents that need processing
        """
        # Find rows that are not completed yet
        cursor = collection.find(
            {
                "user_id": user_id,
                "dataset_name": dataset_name,
                "table_name": table_name,
                "$or": [
                    {"status": {"$ne": "DONE"}},
                    {"ml_status": {"$ne": "DONE"}},
                    {"el_results": {"$eq": {}}},
                ],
            },
            {"row_id": 1},
        ).limit(batch_size)

        return list(cursor)
