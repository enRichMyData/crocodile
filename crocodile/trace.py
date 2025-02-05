import multiprocessing as mp
import time
from datetime import datetime

from crocodile.mongo import MongoConnectionManager


class TraceThread(mp.Process):
    def __init__(
        self,
        mongo_uri,
        db_name,
        input_collection_name,
        dataset_trace_collection_name,
        table_trace_collection_name,
        timing_collection_name,
    ):
        """
        Thread responsible for tracking dataset/table progress:
          1) Marks the first PENDING dataset as IN_PROGRESS.
          2) Continuously checks row counts (TODO, DOING, DONE).
          3) Updates both dataset_trace_collection and table_trace_collection with progress.
          4) Marks dataset/table as DONE/COMPLETED when no more TODO or DOING rows remain.

        :param input_collection: MongoDB collection that stores individual rows
                                 with fields like: { "dataset_name", "table_name", "status" }
        :param dataset_trace_collection: MongoDB collection storing dataset-level info
        :param table_trace_collection: MongoDB collection storing table-level info
        :param timing_collection: MongoDB collection for logging operation timings
        """
        super().__init__()
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.input_collection_name = input_collection_name
        self.dataset_trace_collection_name = dataset_trace_collection_name
        self.table_trace_collection_name = table_trace_collection_name
        self.timing_collection_name = timing_collection_name

        # Fetch the first dataset to process (status=PENDING)
        self.dataset_name = self.get_next_dataset()

    def get_db(self):
        """Get MongoDB database connection for current process"""
        client = MongoConnectionManager.get_client(self.mongo_uri)
        return client[self.db_name]

    def time_mongo_operation(self, operation_name, query_function, *args, **kwargs):
        """
        Wraps a MongoDB operation with timing-logging functionality.
        Logs the start/end times and duration to timing_collection.
        """
        start_time = time.time()
        db = self.get_db()
        timing_collection = db[self.timing_collection_name]

        try:
            result = query_function(*args, **kwargs)
        except Exception as e:
            end_time = time.time()
            timing_collection.insert_one(
                {
                    "operation_name": operation_name,
                    "start_time": datetime.fromtimestamp(start_time),
                    "end_time": datetime.fromtimestamp(end_time),
                    "duration_seconds": round(end_time - start_time, 4),
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "error": str(e),
                    "status": "FAILED",
                }
            )
            raise

        end_time = time.time()
        timing_collection.insert_one(
            {
                "operation_name": operation_name,
                "start_time": datetime.fromtimestamp(start_time),
                "end_time": datetime.fromtimestamp(end_time),
                "duration_seconds": round(end_time - start_time, 4),
                "args": str(args),
                "kwargs": str(kwargs),
                "status": "SUCCESS",
            }
        )
        return result

    def get_next_dataset(self):
        """
        Fetches the next dataset with status=PENDING and marks it as IN_PROGRESS.
        Returns the dataset_name or None if no dataset is found.
        """
        db = self.get_db()
        dataset_trace_collection = db[self.dataset_trace_collection_name]
        doc_dataset = self.time_mongo_operation(
            "find_next_dataset", dataset_trace_collection.find_one, {"status": "PENDING"}
        )
        if not doc_dataset:
            return None

        dataset_name = doc_dataset.get("dataset_name")
        if not dataset_name:
            return None

        # Mark the dataset as IN_PROGRESS and set start_time
        self.time_mongo_operation(
            "update_dataset_status_to_in_progress",
            dataset_trace_collection.update_one,
            {"dataset_name": dataset_name},
            {"$set": {"status": "IN_PROGRESS", "start_time": datetime.now()}},
        )
        return dataset_name

    def run(self):
        """
        Main loop of the thread: processes the current dataset, then fetches the next one.
        """
        while self.dataset_name:
            # Process the current dataset until it's DONE
            self.process_current_dataset()

            # Attempt to fetch the next dataset
            next_dataset = self.get_next_dataset()
            if next_dataset:
                self.dataset_name = next_dataset
            else:
                # No more datasets found
                break

    def process_current_dataset(self):
        """
        Monitors and updates BOTH dataset-level and table-level progress until all rows are DONE.
        Once no more TODO/DOING rows remain, marks the dataset as DONE.
        """
        db = self.get_db()
        input_collection = db[self.input_collection_name]
        dataset_trace_collection = db[self.dataset_trace_collection_name]
        table_trace_collection = db[self.table_trace_collection_name]
        while True:
            # -----------------------------------------------------------
            # 1) DATASET-LEVEL AGGREGATION
            # -----------------------------------------------------------
            dataset_counts_pipeline = [
                {"$match": {"dataset_name": self.dataset_name}},
                {"$group": {"_id": "$status", "count": {"$sum": 1}}},
            ]

            dataset_counts = self.time_mongo_operation(
                "aggregate_dataset_counts", input_collection.aggregate, dataset_counts_pipeline
            )
            dataset_counts_dict = {doc["_id"]: doc["count"] for doc in dataset_counts}
            total_todo_dataset = dataset_counts_dict.get("TODO", 0)
            total_doing_dataset = dataset_counts_dict.get("DOING", 0)
            total_done_dataset = dataset_counts_dict.get("DONE", 0)

            # Fetch dataset trace
            dataset_trace = self.time_mongo_operation(
                "find_dataset_trace",
                dataset_trace_collection.find_one,
                {"dataset_name": self.dataset_name},
            )
            if not dataset_trace:
                # Dataset record no longer exists; break out
                break

            # If total_rows is not set, count them once and store
            total_rows_dataset = dataset_trace.get("total_rows", None)
            if total_rows_dataset is None:
                total_rows_dataset = self.time_mongo_operation(
                    "count_total_rows_dataset",
                    input_collection.count_documents,
                    {"dataset_name": self.dataset_name},
                )
                self.time_mongo_operation(
                    "update_total_rows_dataset",
                    dataset_trace_collection.update_one,
                    {"dataset_name": self.dataset_name},
                    {"$set": {"total_rows": total_rows_dataset}},
                )

            # Compute dataset-level progress stats
            dataset_start_time = dataset_trace.get("start_time", datetime.now())
            dataset_time_passed = (datetime.now() - dataset_start_time).total_seconds()
            dataset_processed_rows = total_done_dataset
            dataset_rows_per_second = (
                dataset_processed_rows / dataset_time_passed if dataset_time_passed > 0 else 0.0
            )
            dataset_completion_percentage = (
                (dataset_processed_rows / total_rows_dataset) * 100
                if total_rows_dataset > 0
                else 0.0
            )

            # Update dataset trace with current progress
            self.time_mongo_operation(
                "update_dataset_trace_progress",
                dataset_trace_collection.update_one,
                {"dataset_name": self.dataset_name},
                {
                    "$set": {
                        "status_counts": {
                            "TODO": total_todo_dataset,
                            "DOING": total_doing_dataset,
                            "DONE": total_done_dataset,
                        },
                        "processed_rows": dataset_processed_rows,
                        "time_passed_seconds": round(dataset_time_passed, 2),
                        "rows_per_second": round(dataset_rows_per_second, 2),
                        "completion_percentage": round(dataset_completion_percentage, 2),
                    }
                },
            )

            # -----------------------------------------------------------
            # 2) TABLE-LEVEL: Only process tables not yet COMPLETED
            # -----------------------------------------------------------
            # First fetch all table_traces in the current dataset that are not "COMPLETED"
            non_completed_tables = self.time_mongo_operation(
                "find_non_completed_tables",
                table_trace_collection.find,
                {"dataset_name": self.dataset_name, "status": {"$ne": "COMPLETED"}},
            )

            # For each table that's not COMPLETED, get row counts by status
            for table_doc in non_completed_tables:
                table_name = table_doc["table_name"]
                tbl_status = table_doc.get("status", "PENDING")
                tbl_start_time = table_doc.get("start_time")

                # Aggregate row counts just for this table
                table_counts_pipeline = [
                    {"$match": {"dataset_name": self.dataset_name, "table_name": table_name}},
                    {"$group": {"_id": "$status", "count": {"$sum": 1}}},
                ]
                result = self.time_mongo_operation(
                    "aggregate_table_counts", input_collection.aggregate, table_counts_pipeline
                )

                # Convert to dict
                counts_dict = {doc["_id"]: doc["count"] for doc in result}
                total_todo = counts_dict.get("TODO", 0)
                total_doing = counts_dict.get("DOING", 0)
                total_done = counts_dict.get("DONE", 0)

                # If 'total_rows' not set in the table trace, fetch and store it
                table_total_rows = table_doc.get("total_rows")
                if table_total_rows is None:
                    table_total_rows = total_todo + total_doing + total_done
                    # Alternatively, you could do a count_documents if you want absolute accuracy
                    # table_total_rows = self.time_mongo_operation(
                    #     "count_rows_for_table",
                    #     self.input_collection.count_documents,
                    #     {"dataset_name": self.dataset_name, "table_name": table_name}
                    # )
                    self.time_mongo_operation(
                        "update_table_total_rows",
                        table_trace_collection.update_one,
                        {"dataset_name": self.dataset_name, "table_name": table_name},
                        {"$set": {"total_rows": table_total_rows}},
                    )
                else:
                    # Use the doc's existing total_rows
                    pass

                # If table is PENDING but we have DOING or DONE, mark it IN_PROGRESS
                if tbl_status == "PENDING" and (total_doing > 0 or total_done > 0):
                    tbl_status = "IN_PROGRESS"
                    tbl_start_time = datetime.now()
                    self.time_mongo_operation(
                        "update_table_status_in_progress",
                        table_trace_collection.update_one,
                        {"dataset_name": self.dataset_name, "table_name": table_name},
                        {"$set": {"status": "IN_PROGRESS", "start_time": tbl_start_time}},
                    )

                # Calculate table-based stats
                if tbl_start_time:
                    table_time_passed = (datetime.now() - tbl_start_time).total_seconds()
                else:
                    table_time_passed = 0.0

                processed_rows = total_done
                rows_per_second = (
                    processed_rows / table_time_passed if table_time_passed > 0 else 0.0
                )
                completion_percentage = (
                    (processed_rows / table_total_rows) * 100
                    if table_total_rows and table_total_rows > 0
                    else 0.0
                )

                # Update table trace
                self.time_mongo_operation(
                    "update_table_trace_progress",
                    table_trace_collection.update_one,
                    {"dataset_name": self.dataset_name, "table_name": table_name},
                    {
                        "$set": {
                            "status_counts": {
                                "TODO": total_todo,
                                "DOING": total_doing,
                                "DONE": total_done,
                            },
                            "processed_rows": processed_rows,
                            "time_passed_seconds": round(table_time_passed, 2),
                            "rows_per_second": round(rows_per_second, 2),
                            "completion_percentage": round(completion_percentage, 2),
                        }
                    },
                )

                # If this table has no TODO/DOING, it's completed
                if total_todo + total_doing == 0:
                    self.time_mongo_operation(
                        "update_table_status_completed",
                        table_trace_collection.update_one,
                        {"dataset_name": self.dataset_name, "table_name": table_name},
                        {"$set": {"status": "COMPLETED", "end_time": datetime.now()}},
                    )

            # -----------------------------------------------------------
            # 3) CHECK IF DATASET IS FULLY DONE
            # -----------------------------------------------------------
            if total_todo_dataset + total_doing_dataset == 0:
                # Mark dataset as DONE
                self.time_mongo_operation(
                    "update_dataset_status_done",
                    dataset_trace_collection.update_one,
                    {"dataset_name": self.dataset_name},
                    {"$set": {"status": "DONE", "end_time": datetime.now()}},
                )
                break
