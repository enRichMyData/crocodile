import time
import asyncio
import aiohttp
from pymongo import MongoClient
import multiprocessing as mp
import traceback
from datetime import datetime
from urllib.parse import quote
import nltk
import warnings
import absl.logging
import pandas as pd
import hashlib
from collections import defaultdict, Counter
import numpy as np
from typing import Dict
from threading import Lock
import os

MY_TIMEOUT = aiohttp.ClientTimeout(
    total=30,        # Total time for the request
    connect=5,       # Time to connect to the server
    sock_connect=5,  # Time to wait for a free socket
    sock_read=25     # Time to read response
)

# Suppress certain Keras/TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Do not pass an `input_shape`.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Compiled the loaded model, but the compiled metrics.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Error in loading the saved optimizer state.*")

# Set logging levels
#tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)

# NLTK setup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

class MongoCache:
    """MongoDB-based cache for storing key-value pairs."""
    def __init__(self, db, collection_name):
        self.collection = db[collection_name]
        self.collection.create_index('key', unique=True)

    def get(self, key):
        result = self.collection.find_one({'key': key})
        if result:
            return result['value']
        return None

    def put(self, key, value):
        self.collection.update_one({'key': key}, {'$set': {'value': value}}, upsert=True)


import time
from datetime import datetime


class TraceThread(mp.Process):
    def __init__(self, mongo_uri, db_name, input_collection_name, dataset_trace_collection_name, table_trace_collection_name, timing_collection_name):
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
            timing_collection.insert_one({
                "operation_name": operation_name,
                "start_time": datetime.fromtimestamp(start_time),
                "end_time": datetime.fromtimestamp(end_time),
                "duration_seconds": round(end_time - start_time, 4),
                "args": str(args),
                "kwargs": str(kwargs),
                "error": str(e),
                "status": "FAILED",
            })
            raise
     
        end_time = time.time()
        timing_collection.insert_one({
            "operation_name": operation_name,
            "start_time": datetime.fromtimestamp(start_time),
            "end_time": datetime.fromtimestamp(end_time),
            "duration_seconds": round(end_time - start_time, 4),
            "args": str(args),
            "kwargs": str(kwargs),
            "status": "SUCCESS",
        })
        return result

    def get_next_dataset(self):
        """
        Fetches the next dataset with status=PENDING and marks it as IN_PROGRESS.
        Returns the dataset_name or None if no dataset is found.
        """
        db = self.get_db()
        dataset_trace_collection = db[self.dataset_trace_collection_name]
        doc_dataset = self.time_mongo_operation(
            "find_next_dataset",
            dataset_trace_collection.find_one,
            {"status": "PENDING"}
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
            {
                "$set": {
                    "status": "IN_PROGRESS",
                    "start_time": datetime.now()
                }
            }
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
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            
            dataset_counts = self.time_mongo_operation(
                "aggregate_dataset_counts",
                input_collection.aggregate,
                dataset_counts_pipeline
            )
            dataset_counts_dict = {doc["_id"]: doc["count"] for doc in dataset_counts}
            total_todo_dataset = dataset_counts_dict.get("TODO", 0)
            total_doing_dataset = dataset_counts_dict.get("DOING", 0)
            total_done_dataset = dataset_counts_dict.get("DONE", 0)

            # Fetch dataset trace
            dataset_trace = self.time_mongo_operation(
                "find_dataset_trace",
                dataset_trace_collection.find_one,
                {"dataset_name": self.dataset_name}
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
                    {"dataset_name": self.dataset_name}
                )
                self.time_mongo_operation(
                    "update_total_rows_dataset",
                    dataset_trace_collection.update_one,
                    {"dataset_name": self.dataset_name},
                    {"$set": {"total_rows": total_rows_dataset}}
                )

            # Compute dataset-level progress stats
            dataset_start_time = dataset_trace.get("start_time", datetime.now())
            dataset_time_passed = (datetime.now() - dataset_start_time).total_seconds()
            dataset_processed_rows = total_done_dataset
            dataset_rows_per_second = (
                dataset_processed_rows / dataset_time_passed if dataset_time_passed > 0 else 0.0
            )
            dataset_completion_percentage = (
                (dataset_processed_rows / total_rows_dataset) * 100 if total_rows_dataset > 0 else 0.0
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
                            "DONE": total_done_dataset
                        },
                        "processed_rows": dataset_processed_rows,
                        "time_passed_seconds": round(dataset_time_passed, 2),
                        "rows_per_second": round(dataset_rows_per_second, 2),
                        "completion_percentage": round(dataset_completion_percentage, 2)
                    }
                }
            )

            # -----------------------------------------------------------
            # 2) TABLE-LEVEL: Only process tables not yet COMPLETED
            # -----------------------------------------------------------
            # First fetch all table_traces in the current dataset that are not "COMPLETED"
            non_completed_tables = self.time_mongo_operation(
                "find_non_completed_tables",
                table_trace_collection.find,
                {
                    "dataset_name": self.dataset_name,
                    "status": {"$ne": "COMPLETED"}
                }
            )

            # For each table that's not COMPLETED, get row counts by status
            for table_doc in non_completed_tables:
                table_name = table_doc["table_name"]
                tbl_status = table_doc.get("status", "PENDING")
                tbl_start_time = table_doc.get("start_time")

                # Aggregate row counts just for this table
                table_counts_pipeline = [
                    {
                        "$match": {
                            "dataset_name": self.dataset_name,
                            "table_name": table_name
                        }
                    },
                    {
                        "$group": {
                            "_id": "$status",
                            "count": {"$sum": 1}
                        }
                    }
                ]
                result = self.time_mongo_operation(
                    "aggregate_table_counts",
                    input_collection.aggregate,
                    table_counts_pipeline
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
                        {
                            "dataset_name": self.dataset_name,
                            "table_name": table_name
                        },
                        {"$set": {"total_rows": table_total_rows}}
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
                        {
                            "dataset_name": self.dataset_name,
                            "table_name": table_name
                        },
                        {
                            "$set": {
                                "status": "IN_PROGRESS",
                                "start_time": tbl_start_time
                            }
                        }
                    )

                # Calculate table-based stats
                if tbl_start_time:
                    table_time_passed = (datetime.now() - tbl_start_time).total_seconds()
                else:
                    table_time_passed = 0.0

                processed_rows = total_done
                rows_per_second = processed_rows / table_time_passed if table_time_passed > 0 else 0.0
                completion_percentage = (
                    (processed_rows / table_total_rows) * 100
                    if table_total_rows and table_total_rows > 0 else 0.0
                )

                # Update table trace
                self.time_mongo_operation(
                    "update_table_trace_progress",
                    table_trace_collection.update_one,
                    {
                        "dataset_name": self.dataset_name,
                        "table_name": table_name
                    },
                    {
                        "$set": {
                            "status_counts": {
                                "TODO": total_todo,
                                "DOING": total_doing,
                                "DONE": total_done
                            },
                            "processed_rows": processed_rows,
                            "time_passed_seconds": round(table_time_passed, 2),
                            "rows_per_second": round(rows_per_second, 2),
                            "completion_percentage": round(completion_percentage, 2)
                        }
                    }
                )

                # If this table has no TODO/DOING, it's completed
                if total_todo + total_doing == 0:
                    self.time_mongo_operation(
                        "update_table_status_completed",
                        table_trace_collection.update_one,
                        {
                            "dataset_name": self.dataset_name,
                            "table_name": table_name
                        },
                        {
                            "$set": {
                                "status": "COMPLETED",
                                "end_time": datetime.now()
                            }
                        }
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
                    {
                        "$set": {
                            "status": "DONE",
                            "end_time": datetime.now()
                        }
                    }
                )
                break

    
class MongoConnectionManager:
    _instances: Dict[int, MongoClient] = {}
    _lock = Lock()
    
    @classmethod
    def get_client(cls, mongo_uri):
        pid = os.getpid()
        
        with cls._lock:
            if pid not in cls._instances:
                client = MongoClient(
                    mongo_uri,
                    maxPoolSize=100,
                    minPoolSize=8,
                    waitQueueTimeoutMS=30000,
                    retryWrites=True,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=30000,
                    socketTimeoutMS=30000
                )
                cls._instances[pid] = client
            
            return cls._instances[pid]

    @classmethod
    def close_connection(cls, pid=None):
        if pid is None:
            pid = os.getpid()
            
        with cls._lock:
            if pid in cls._instances:
                cls._instances[pid].close()
                del cls._instances[pid]
    
    @classmethod
    def close_all_connections(cls):
        with cls._lock:
            for client in cls._instances.values():
                client.close()
            cls._instances.clear()

class Crocodile:
    def __init__(self, mongo_uri="mongodb://localhost:27017/",
                 db_name="crocodile_db",
                 table_trace_collection_name="table_trace",
                 dataset_trace_collection_name="dataset_trace",
                 input_collection="input_data",
                 training_collection_name="training_data",
                 error_log_collection_name="error_logs",
                 timing_collection_name="timing_trace",
                 cache_collection_name="candidate_cache",
                 bow_cache_collection_name="bow_cache",
                 max_workers=None, max_candidates=5, max_training_candidates=10,
                 entity_retrieval_endpoint=None, entity_bow_endpoint=None, entity_retrieval_token=None,
                 selected_features=None, candidate_retrieval_limit=100,
                 model_path=None,
                 batch_size=1024,
                 ml_ranking_workers=2,
                 top_n_for_type_freq=3,
                 max_bow_batch_size=100):

        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.input_collection = input_collection
        self.table_trace_collection_name = table_trace_collection_name
        self.dataset_trace_collection_name = dataset_trace_collection_name
        self.training_collection_name = training_collection_name
        self.error_log_collection_name = error_log_collection_name
        self.timing_collection_name = timing_collection_name
        self.cache_collection_name = cache_collection_name
        self.bow_cache_collection_name = bow_cache_collection_name
        self.max_workers = max_workers or mp.cpu_count()
        self.max_candidates = max_candidates
        self.max_training_candidates = max_training_candidates
        self.entity_retrieval_endpoint = entity_retrieval_endpoint
        self.entity_retrieval_token = entity_retrieval_token
        self.candidate_retrieval_limit = candidate_retrieval_limit
        self.selected_features = selected_features or [
            "ntoken_mention", "ntoken_entity", "length_mention", "length_entity",
            "popularity", "ed_score", "jaccard_score", "jaccardNgram_score", "desc", "descNgram",
            "bow_similarity", "kind", "NERtype", "column_NERtype",
            "typeFreq1", "typeFreq2", "typeFreq3", "typeFreq4", "typeFreq5"
        ]
        self.model_path = model_path
        self.entity_bow_endpoint = entity_bow_endpoint
        self.batch_size = batch_size
        self.ml_ranking_workers = ml_ranking_workers
        self.top_n_for_type_freq = top_n_for_type_freq
        self.MAX_BOW_BATCH_SIZE = max_bow_batch_size

    def get_db(self):
        """Get MongoDB database connection for current process"""
        client = MongoConnectionManager.get_client(self.mongo_uri)
        return client[self.db_name]

    def __del__(self):
        """Cleanup when instance is destroyed"""
        try:
            MongoConnectionManager.close_connection()
        except:
            pass

    def get_candidate_cache(self):
        db = self.get_db()
        return MongoCache(db, self.cache_collection_name)

    def get_bow_cache(self):
        db = self.get_db()
        return MongoCache(db, self.bow_cache_collection_name)

    def time_mongo_operation(self, operation_name, query_function, *args, **kwargs):
        start_time = time.time()
        db = self.get_db()
        timing_trace_collection = db[self.timing_collection_name]
        try:
            result = query_function(*args, **kwargs)
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            timing_trace_collection.insert_one({
                "operation_name": operation_name,
                "start_time": datetime.fromtimestamp(start_time),
                "end_time": datetime.fromtimestamp(end_time),
                "duration_seconds": duration,
                "args": str(args),
                "kwargs": str(kwargs),
                "error": str(e),
                "status": "FAILED",
            })
            raise
        else:
            end_time = time.time()
            duration = end_time - start_time
            timing_trace_collection.insert_one({
                "operation_name": operation_name,
                "start_time": datetime.fromtimestamp(start_time),
                "end_time": datetime.fromtimestamp(end_time),
                "duration_seconds": duration,
                "args": str(args),
                "kwargs": str(kwargs),
                "status": "SUCCESS",
            })
            return result

    def update_document(self, collection, query, update, upsert=False):
        operation_name = f"update_document:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.update_one, query, update, upsert=upsert)

    def update_documents(self, collection, query, update, upsert=False):
        operation_name = f"update_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.update_many, query, update, upsert=upsert)

    def find_documents(self, collection, query, projection=None, limit=None):
        operation_name = f"find_documents:{collection.name}"
        def query_function(query, projection=None):
            cursor = collection.find(query, projection)
            if limit is not None:
                cursor = cursor.limit(limit)
            return list(cursor)
        return self.time_mongo_operation(operation_name, query_function, query, projection)

    def count_documents(self, collection, query):
        operation_name = f"count_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.count_documents, query)

    def find_one_document(self, collection, query, projection=None):
        operation_name = f"find_one_document:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.find_one, query, projection=projection)

    def find_one_and_update(self, collection, query, update, return_document=False):
        operation_name = f"find_one_and_update:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.find_one_and_update, query, update, return_document=return_document)

    def insert_one_document(self, collection, document):
        operation_name = f"insert_one_document:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.insert_one, document)

    def insert_many_documents(self, collection, documents):
        operation_name = f"insert_many_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.insert_many, documents)

    def delete_documents(self, collection, query):
        operation_name = f"delete_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.delete_many, query)

    def log_time(self, operation_name, dataset_name, table_name, start_time, end_time, details=None):
        db = self.get_db()
        timing_collection = db[self.timing_collection_name]
        duration = end_time - start_time
        log_entry = {
            "operation_name": operation_name,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "start_time": datetime.fromtimestamp(start_time),
            "end_time": datetime.fromtimestamp(end_time),
            "duration_seconds": duration,
        }
        if details:
            log_entry["details"] = details
        timing_collection.insert_one(log_entry)

    def log_to_db(self, level, message, trace=None, attempt=None):
        db = self.get_db()
        log_collection = db[self.error_log_collection_name]
        log_entry = {
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "traceback": trace
        }
        if attempt is not None:
            log_entry["attempt"] = attempt
        log_collection.insert_one(log_entry)

    async def _fetch_candidates(self, entity_name, row_text, fuzzy, qid, session):
        db = self.get_db()
        timing_trace_collection = db[self.timing_collection_name]
        
        # Encode the entity_name to handle special characters
        encoded_entity_name = quote(entity_name)
        url = f"{self.entity_retrieval_endpoint}?name={encoded_entity_name}&limit={self.candidate_retrieval_limit}&fuzzy={fuzzy}&token={self.entity_retrieval_token}"
        
        if qid:
            url += f"&ids={qid}"
        backoff = 1

        # We'll attempt up to 5 times
        for attempts in range(5):
            start_time = time.time()
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    candidates = await response.json()
                    row_tokens = set(self.tokenize_text(row_text))
                    fetched_candidates = self._process_candidates(candidates, entity_name, row_tokens)

                    # Ensure all QIDs are included by adding placeholders for missing ones
                    required_qids = qid.split() if qid else []
                    existing_qids = {c['id'] for c in fetched_candidates if c.get('id')}
                    missing_qids = set(required_qids) - existing_qids

                    for missing_qid in missing_qids:
                        fetched_candidates.append({
                            'id': missing_qid,  # Placeholder for missing QID
                            'name': None,
                            'description': None,
                            'features': None, # Explicitly set features to None
                            'is_placeholder': True  # Explicitly mark as placeholder
                        })

                    # Merge with existing cache if present
                    cache = self.get_candidate_cache()
                    cache_key = f"{entity_name}_{fuzzy}"
                    cached_result = cache.get(cache_key)

                    if cached_result:
                        # Use a dict keyed by QID to ensure uniqueness
                        all_candidates = {c['id']: c for c in cached_result if 'id' in c}
                        for c in fetched_candidates:
                            if c.get('id'):
                                all_candidates[c['id']] = c
                        merged_candidates = list(all_candidates.values())
                    else:
                        merged_candidates = fetched_candidates

                    # Update cache with merged results
                    cache.put(cache_key, merged_candidates)

                    # Log success
                    end_time = time.time()
                    timing_trace_collection.insert_one({
                        "operation_name": "_fetch_candidate",
                        "url": url,
                        "start_time": datetime.fromtimestamp(start_time),
                        "end_time": datetime.fromtimestamp(end_time),
                        "duration_seconds": end_time - start_time,
                        "status": "SUCCESS",
                        "attempt": attempts + 1,
                    })

                    return entity_name, merged_candidates
            except Exception as e:
                    end_time = time.time()
                    if attempts == 4:
                        # Log the error if all attempts failed
                        self.log_to_db("FETCH_CANDIDATES_ERROR", 
                                       f"Error fetching candidates for {entity_name}", 
                                       traceback.format_exc(), attempt=attempts + 1)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 16)

        # If all attempts fail, return empty
        return entity_name, []

    async def fetch_candidates_batch_async(self, entities, row_texts, fuzzies, qids):
        results = {}
        cache = self.get_candidate_cache()
        to_fetch = []

        # Decide which entities need to be fetched
        for entity_name, fuzzy, row_text, qid_str in zip(entities, fuzzies, row_texts, qids):
            cache_key = f"{entity_name}_{fuzzy}"
            cached_result = cache.get(cache_key)
            forced_qids = qid_str.split() if qid_str else []

            if cached_result is not None:
                if forced_qids:
                    # Check if all forced QIDs are already present
                    cached_qids = {c['id'] for c in cached_result if 'id' in c}
                    if all(q in cached_qids for q in forced_qids):
                        # All required QIDs are present, no fetch needed
                        results[entity_name] = cached_result
                    else:
                        # Forced QIDs not all present, must fetch
                        to_fetch.append((entity_name, fuzzy, row_text, qid_str))
                else:
                    # No forced QIDs, just use cached
                    results[entity_name] = cached_result
            else:
                # No cache entry, must fetch
                to_fetch.append((entity_name, fuzzy, row_text, qid_str))
        
        # If nothing to fetch, return what we have
        if not to_fetch:
            return self._remove_placeholders(results)

        # Fetch missing data
        async with aiohttp.ClientSession(timeout=MY_TIMEOUT, connector=aiohttp.TCPConnector(ssl=False, limit=10)) as session:
            tasks = []
            for (entity_name, fuzzy, row_text, qid_str) in to_fetch:
                tasks.append(self._fetch_candidates(entity_name, row_text, fuzzy, qid_str, session))
            done = await asyncio.gather(*tasks, return_exceptions=False)
            for entity_name, candidates in done:
                results[entity_name] = candidates

        return self._remove_placeholders(results)

    def _remove_placeholders(self, results):
        """Removes placeholder candidates from the results based on `is_placeholder` attribute."""
        for entity_name, candidates in results.items():
            results[entity_name] = [
                c for c in candidates if not c.get('is_placeholder', False)
            ]
        return results
    
    def fetch_candidates_batch(self, entities, row_texts, fuzzies, qids):
        return asyncio.run(self.fetch_candidates_batch_async(entities, row_texts, fuzzies, qids))

    async def _fetch_bow_for_multiple_qids(self, row_hash, row_text, qids, session):
        """
        Entry point for fetching BoW data for multiple QIDs in one row.
        This function:
          1) Splits QIDs into smaller batches;
          2) Fetches each batch sequentially (or you could parallelize);
          3) Merges results into a single dict;
          4) Returns bow_results (qid -> bow_info).
        """
        db = self.get_db()
        timing_trace_collection = db[self.timing_collection_name]
        bow_cache = self.get_bow_cache()

        # 1) Check which QIDs we actually need to fetch
        to_fetch = []
        bow_results = {}

        for qid in qids:
            cache_key = f"{row_hash}_{qid}"
            cached_result = bow_cache.get(cache_key)
            if cached_result is not None:
                bow_results[qid] = cached_result
            else:
                to_fetch.append(qid)

        # If everything is cached, no need to query
        if len(to_fetch) == 0:
            return bow_results

        # 2) Break the `to_fetch` QIDs into small batches
        #    We define chunk size = MAX_BOW_BATCH_SIZE (e.g. 50).
        chunked_qids = [to_fetch[i:i + self.MAX_BOW_BATCH_SIZE] 
                        for i in range(0, len(to_fetch), self.MAX_BOW_BATCH_SIZE)]

        # 3) Fetch each chunk (serially here, but could use asyncio.gather for concurrency)
        for chunk in chunked_qids:
            # We define a helper method that tries to fetch BoW data for a single chunk
            chunk_results = await self._fetch_bow_for_chunk(row_hash, row_text, chunk, session)
            # Merge the chunk results into bow_results
            for qid, data in chunk_results.items():
                bow_results[qid] = data

        return bow_results

    async def _fetch_bow_for_chunk(self, row_hash, row_text, chunk_qids, session):
        """
        Fetch BoW data for a *subset* (chunk) of QIDs.
        Includes the same backoff/retry logic as before, but only
        for these chunk_qids.
        """
        db = self.get_db()
        timing_trace_collection = db[self.timing_collection_name]
        bow_cache = self.get_bow_cache()
        
        # Prepare the results dictionary
        chunk_bow_results = {}
        
        # If empty chunk somehow, just return
        if not chunk_qids:
            return chunk_bow_results

        url = f"{self.entity_bow_endpoint}?token={self.entity_retrieval_token}"
        # The payload includes only the chunk of QIDs
        payload = {
            "json": {
                "text": row_text,
                "qids": chunk_qids
            }
        }

        backoff = 1
        for attempts in range(5):
            start_time = time.time()
            try:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    bow_data = await response.json()

                    # Cache the results and populate
                    for qid in chunk_qids:
                        qid_data = bow_data.get(qid, {"similarity_score": 0.0, "matched_words": []})
                        cache_key = f"{row_hash}_{qid}"
                        bow_cache.put(cache_key, qid_data)
                        chunk_bow_results[qid] = qid_data

                    # Log success
                    end_time = time.time()
                    timing_trace_collection.insert_one({
                        "operation_name": "_fetch_bow_for_multiple_qids:CHUNK",
                        "url": url,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_seconds": end_time - start_time,
                        "status": "SUCCESS",
                        "attempt": attempts + 1,
                    })

                    return chunk_bow_results

            except Exception as e:
                end_time = time.time()
                if attempts == 4:
                    # Log the error if all attempts failed
                    self.log_to_db("FETCH_BOW_ERROR", 
                                f"Error fetching BoW for row_hash={row_hash}, chunk_qids={chunk_qids}",
                                traceback.format_exc(),
                                attempt=attempts + 1)
                # Exponential backoff
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 16)

        # If all attempts fail, return partial or empty
        return chunk_bow_results

    def fetch_bow_vectors_batch(self, row_hash, row_text, qids):
        """
        Public method that logs the request, calls the async method,
        and returns the final BoW results.
        """
        db = self.get_db()

        async def runner():
            async with aiohttp.ClientSession(
                timeout=MY_TIMEOUT,
                connector=aiohttp.TCPConnector(ssl=False, limit=10)
            ) as session:
                return await self._fetch_bow_for_multiple_qids(row_hash, row_text, qids, session)

        return asyncio.run(runner())

    def process_rows_batch(self, docs, dataset_name, table_name):
        db = self.get_db()
        try:
            # Step 1: Collect all entities from all rows (batch) for candidate fetch
            all_entities_to_process = []
            all_row_texts = []
            all_fuzzies = []
            all_qids = []
            all_row_indices = []
            all_col_indices = []
            all_ner_types = []
            
            # This list will hold info for each row so we can process them individually later
            row_data_list = []

            # ---------------------------------------------------------------------
            # Gather row data and NE columns for candidate fetching
            # ---------------------------------------------------------------------
            for doc in docs:
                row = doc['data']
                ne_columns = doc['classified_columns']['NE']
                context_columns = doc.get('context_columns', [])
                correct_qids = doc.get('correct_qids', {})
                row_index = doc.get("row_id", None)

                # Build row_text from the context columns
                raw_context_text = ' '.join(
                    str(row[int(c)]) for c in sorted(context_columns, key=lambda col: str(row[int(col)]))
                )
                # Normalize row text: lowercase and remove extra spaces
                normalized_row_text = raw_context_text.lower()
                normalized_row_text = " ".join(normalized_row_text.split())

                # Hash the normalized text for caching
                row_hash = hashlib.sha256(normalized_row_text.encode()).hexdigest()

                # Save row-level info for later
                row_data_list.append(
                    (
                        doc['_id'],
                        row,
                        ne_columns,
                        context_columns,
                        correct_qids,
                        row_index,
                        raw_context_text,
                        row_hash
                    )
                )

                # For each NE column, prepare entity lookups
                for c, ner_type in ne_columns.items():
                    c = str(c)
                    if int(c) < len(row):
                        ne_value = row[int(c)]
                        if ne_value and pd.notna(ne_value):
                            # Normalize entity value for consistent lookups
                            ne_value = str(ne_value).strip().replace("_", " ").lower()
                            correct_qid = correct_qids.get(f"{row_index}-{c}", None)

                            all_entities_to_process.append(ne_value)
                            all_row_texts.append(raw_context_text)
                            all_fuzzies.append(False)
                            all_qids.append(correct_qid)
                            all_row_indices.append(row_index)
                            all_col_indices.append(c)
                            all_ner_types.append(ner_type)

            # ---------------------------------------------------------------------
            # Fetch candidates (batch) for all entities
            # ---------------------------------------------------------------------
            # 1. Initial fetch
            candidates_results = self.fetch_candidates_batch(
                all_entities_to_process,
                all_row_texts,
                all_fuzzies,
                all_qids
            )

            # ---------------------------------------------------------------------
            # 2. Fuzzy retry for items that returned exactly 1 candidate
            # ---------------------------------------------------------------------
            entities_to_retry = []
            row_texts_retry = []
            fuzzies_retry = []
            qids_retry = []
            row_indices_retry = []
            col_indices_retry = []
            ner_types_retry = []

            for ne_value, r_index, c_index, n_type in zip(
                all_entities_to_process,
                all_row_indices,
                all_col_indices,
                all_ner_types
            ):
                candidates = candidates_results.get(ne_value, [])
                # If there's exactly 1 candidate, let's attempt a fuzzy retry
                if len(candidates) == 1:
                    entities_to_retry.append(ne_value)
                    idx = all_entities_to_process.index(ne_value)
                    row_texts_retry.append(all_row_texts[idx])
                    fuzzies_retry.append(True)
                    correct_qid = all_qids[idx]
                    qids_retry.append(correct_qid)
                    row_indices_retry.append(r_index)
                    col_indices_retry.append(c_index)
                    ner_types_retry.append(n_type)
                else:
                    # Keep the existing candidates
                    candidates_results[ne_value] = candidates

            if entities_to_retry:
                retry_results = self.fetch_candidates_batch(
                    entities_to_retry,
                    row_texts_retry,
                    fuzzies_retry,
                    qids_retry
                )
                for ne_value in entities_to_retry:
                    candidates_results[ne_value] = retry_results.get(ne_value, [])

            # ---------------------------------------------------------------------
            # Process each row individually (including BoW retrieval for that row)
            # ---------------------------------------------------------------------
            for (
                doc_id,
                row,
                ne_columns,
                context_columns,
                correct_qids,
                row_index,
                raw_context_text,
                row_hash
            ) in row_data_list:

                # -------------------------------------------------------------
                # 1. Gather the QIDs relevant to this row
                # -------------------------------------------------------------
                row_qids = []
                for c, ner_type in ne_columns.items():
                    if int(c) < len(row):
                        ne_value = row[int(c)]
                        if ne_value and pd.notna(ne_value):
                            ne_value = str(ne_value).strip().replace("_", " ").lower()
                            candidates = candidates_results.get(ne_value, [])
                            for cand in candidates:
                                if cand['id']:
                                    row_qids.append(cand['id'])
                row_qids = list(set(q for q in row_qids if q))
                
                # -------------------------------------------------------------
                # 2. Fetch BoW vectors for this row’s QIDs
                # -------------------------------------------------------------
                if row_qids and self.entity_bow_endpoint and self.entity_retrieval_token:
                    bow_data = self.fetch_bow_vectors_batch(
                        row_hash,
                        raw_context_text,
                        row_qids
                    )
                else:
                    bow_data = {}

                # -------------------------------------------------------------
                # 3. Build final linked_entities + training_candidates
                # -------------------------------------------------------------
                linked_entities = {}
                training_candidates_by_ne_column = {}

                for c, ner_type in ne_columns.items():
                    c = str(c)
                    if int(c) < len(row):
                        ne_value = row[int(c)]
                        if ne_value and pd.notna(ne_value):
                            ne_value = str(ne_value).strip().replace("_", " ").lower()
                            candidates = candidates_results.get(ne_value, [])

                            # Assign the BoW score + numeric NER type
                            for cand in candidates:
                                qid = cand['id']
                                cand['features']['bow_similarity'] = (
                                    bow_data.get(qid, {}).get('similarity_score', 0.0)
                                )
                                cand['features']['column_NERtype'] = self.map_nertype_to_numeric(ner_type)

                            # 4. Rank candidates by feature scoring
                            ranked_candidates = self.rank_with_feature_scoring(candidates)

                            # 4.1 Ensure they are sorted by 'score' descending
                            ranked_candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)

                            # If there's a correct_qid, ensure it appears in the top training slice
                            correct_qid = correct_qids.get(f"{row_index}-{c}", None)
                            if (
                                correct_qid
                                and correct_qid not in [rc['id'] for rc in ranked_candidates[: self.max_training_candidates]]
                            ):
                                correct_candidate = next(
                                    (x for x in ranked_candidates if x['id'] == correct_qid),
                                    None
                                )
                                if correct_candidate:
                                    top_slice = ranked_candidates[: self.max_training_candidates - 1]
                                    top_slice.append(correct_candidate)
                                    ranked_candidates = top_slice
                                    # Re-sort after adding the correct candidate
                                    ranked_candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)

                            el_results_candidates = ranked_candidates[: self.max_candidates]
                            linked_entities[c] = el_results_candidates
                            training_candidates_by_ne_column[c] = ranked_candidates[: self.max_training_candidates]

                # -------------------------------------------------------------
                # 5. Save to DB: training candidates + final EL results
                # -------------------------------------------------------------
                self.save_candidates_for_training(
                    training_candidates_by_ne_column,
                    dataset_name,
                    table_name,
                    row_index
                )
                db[self.input_collection].update_one(
                    {'_id': doc_id},
                    {'$set': {'el_results': linked_entities, 'status': 'DONE'}}
                )

            # Optionally track speed after full batch
            self.log_processing_speed(dataset_name, table_name)

        except Exception as e:
            self.log_to_db("ERROR", "Error processing batch of rows", traceback.format_exc())

    def map_kind_to_numeric(self, kind):
        mapping = {
            'entity': 1,
            'type': 2,
            'disambiguation': 3,
            'predicate': 4
        }
        return mapping.get(kind, 1)

    def map_nertype_to_numeric(self, nertype):
        mapping = {
            'LOCATION': 1,
            'LOC': 1,
            'ORGANIZATION': 2,
            'ORG': 2,
            'PERSON': 3,
            'PERS': 3,
            'OTHER': 4,
            'OTHERS': 4
        }
        return mapping.get(nertype, 4)

    def calculate_token_overlap(self, tokens_a, tokens_b):
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0

    def calculate_ngram_similarity(self, a, b, n=3):
        a_ngrams = self.ngrams(a, n)
        b_ngrams = self.ngrams(b, n)
        intersection = len(set(a_ngrams) & set(b_ngrams))
        union = len(set(a_ngrams) | set(b_ngrams))
        return intersection / union if union > 0 else 0

    def ngrams(self, string, n=3):
        tokens = [string[i:i+n] for i in range(len(string)-n+1)]
        return tokens

    def tokenize_text(self, text):
        tokens = word_tokenize(text.lower())
        return set(t for t in tokens if t not in stop_words)

    def _process_candidates(self, candidates, entity_name, row_tokens):
        processed_candidates = []
        for candidate in candidates:
            candidate_name = candidate.get('name', '')
            candidate_description = candidate.get('description', '') or ""
            kind_numeric = self.map_kind_to_numeric(candidate.get('kind', 'entity'))
            nertype_numeric = self.map_nertype_to_numeric(candidate.get('NERtype', 'OTHERS'))

            features = {
                'ntoken_mention': round(candidate.get('ntoken_mention', len(entity_name.split())), 4),
                'ntoken_entity': round(candidate.get('ntoken_entity', len(candidate_name.split())), 4),
                'length_mention': round(len(entity_name), 4),
                'length_entity': round(len(candidate_name), 4),
                'popularity': round(candidate.get('popularity', 0.0), 4),
                'ed_score': round(candidate.get('ed_score', 0.0), 4),
                'jaccard_score': round(candidate.get('jaccard_score', 0.0), 4),
                'jaccardNgram_score': round(candidate.get('jaccardNgram_score', 0.0), 4),
                'desc': round(self.calculate_token_overlap(row_tokens, set(self.tokenize_text(candidate_description))), 4),
                'descNgram': round(self.calculate_ngram_similarity(entity_name, candidate_description), 4),
                'bow_similarity': 0.0,
                'kind': kind_numeric,
                'NERtype': nertype_numeric,
                'column_NERtype': None
            }

            processed_candidates.append({
                'id': candidate.get('id'),
                'name': candidate_name,
                'description': candidate_description,
                'types': candidate.get('types'),
                'features': features
            })
        return processed_candidates

    def score_candidate(self, candidate):
        ed_score = candidate['features'].get('ed_score', 0.0)
        desc_score = candidate['features'].get('desc', 0.0)
        desc_ngram_score = candidate['features'].get('descNgram', 0.0)
        feature_sum = ed_score + desc_score + desc_ngram_score
        total_score = (feature_sum / 3) if feature_sum > 0 else 0.0
        candidate['score'] = round(total_score, 2)
        return candidate

    def rank_with_feature_scoring(self, candidates):
        scored_candidates = [self.score_candidate(c) for c in candidates]
        return sorted(scored_candidates, key=lambda x: x['score'], reverse=True)

    def save_candidates_for_training(self, candidates_by_ne_column, dataset_name, table_name, row_index):
        db = self.get_db()
        training_collection = db[self.training_collection_name]
        training_document = {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_index,
            "candidates": candidates_by_ne_column,
            "ml_ranked": False
        }
        training_collection.insert_one(training_document)

    def log_processing_speed(self, dataset_name, table_name):
        db = self.get_db()
        table_trace_collection = db[self.table_trace_collection_name]

        trace = table_trace_collection.find_one({"dataset_name": dataset_name, "table_name": table_name})
        if not trace:
            return
        processed_rows = trace.get("processed_rows", 1)
        start_time = trace.get("start_time")
        elapsed_time = (datetime.now() - start_time).total_seconds() if start_time else 0
        rows_per_second = processed_rows / elapsed_time if elapsed_time > 0 else 0
        table_trace_collection.update_one(
            {"dataset_name": dataset_name, "table_name": table_name},
            {"$set": {"rows_per_second": rows_per_second}}
        )

    def claim_todo_batch(self, input_collection, batch_size=10):
        """
        Atomically claims a batch of TODO documents by setting them to DOING,
        and returns the full documents so we don't have to fetch them again.
        """
        docs = []
        for _ in range(batch_size):
            doc = input_collection.find_one_and_update(
                {"status": "TODO"},
                {"$set": {"status": "DOING"}}
            )
            if doc is None:
                # No more TODO docs
                break
            docs.append(doc)
        return docs

    def worker(self):
        db = self.get_db()
        input_collection = db[self.input_collection]

        while True:
            # Atomically claim a batch of documents (full docs, not partial)
            todo_docs = self.claim_todo_batch(input_collection)
            if not todo_docs:
                print("No more tasks to process.")
                break

            # Group the claimed documents by (dataset_name, table_name)
            tasks_by_table = {}
            for doc in todo_docs:
                dataset_name = doc["dataset_name"]
                table_name = doc["table_name"]
                # Accumulate full docs in a list
                tasks_by_table.setdefault((dataset_name, table_name), []).append(doc)

            # Process each group as a batch
            for (dataset_name, table_name), docs in tasks_by_table.items():
                self.process_rows_batch(docs, dataset_name, table_name)

    def ml_ranking_worker(self):
        model = self.load_ml_model()
        db = self.get_db()
        table_trace_collection = db[self.table_trace_collection_name]
        while True:
            todo_table = self.find_one_document(table_trace_collection, {"ml_ranking_status": None})
            if todo_table is None: # no more tasks to process 
                break
            table_trace_obj = self.find_one_and_update(table_trace_collection,  {"$and": [{"status": "COMPLETED"}, {"ml_ranking_status": None}]}, {"$set": {"ml_ranking_status": "PENDING"}})
            if table_trace_obj:
                dataset_name = table_trace_obj.get("dataset_name")
                table_name = table_trace_obj.get("table_name")
                self.apply_ml_ranking(dataset_name, table_name, model)
            
    def run(self):
        #mp.set_start_method("spawn", force=True)

        db = self.get_db()
        input_collection = db[self.input_collection]
        dataset_trace_collection = db[self.dataset_trace_collection_name]
        table_trace_collection = db[self.table_trace_collection_name]
        timing_collection = db[self.timing_collection_name]

        total_rows = self.count_documents(input_collection, {"status": "TODO"})
        if total_rows == 0:
            print("No more tasks to process.")
        else:    
            print(f"Found {total_rows} tasks to process.")

        processes = []
        for _ in range(self.max_workers):
            p = mp.Process(target=self.worker)
            p.start()
            processes.append(p)
        
        for _ in range(self.ml_ranking_workers):
             p = mp.Process(target=self.ml_ranking_worker)
             p.start()
             processes.append(p)
        
        trace_thread = TraceThread(self.mongo_uri, self.db_name, self.input_collection, self.dataset_trace_collection_name, 
                                   self.table_trace_collection_name, self.timing_collection_name)
        trace_thread.start()
        processes.append(trace_thread)

        for p in processes:
            p.join()
        
        self.__del__()

        print("All tasks have been processed.")

    def apply_ml_ranking(self, dataset_name, table_name, model):
        db = self.get_db()
        training_collection = db[self.training_collection_name]
        input_collection = db[self.input_collection]
        table_trace_collection = db[self.table_trace_collection_name]

        processed_count = 0
        total_count = self.count_documents(training_collection, {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "ml_ranked": False
        })
        print(f"Total unprocessed documents: {total_count}")

        # We'll restrict type-freq counting to the top N candidates in each row/col
        top_n_for_type_freq = self.top_n_for_type_freq  # e.g. 3

        while processed_count < total_count:
            batch_docs = list(training_collection.find(
                {"dataset_name": dataset_name, "table_name": table_name, "ml_ranked": False},
                limit=self.batch_size
            ))
            if not batch_docs:
                break

            # Map documents by _id so we can update them later
            doc_map = {doc["_id"]: doc for doc in batch_docs}

            # For each column, we'll store a Counter of type_id -> how many rows had that type
            type_freq_by_column = defaultdict(Counter)

            # We also track how many rows in the batch actually had that column
            # so we can normalize frequencies to 0..1
            rows_count_by_column = Counter()

            #--------------------------------------------------------------------------
            # 1) Collect "top N" type IDs per column, ignoring duplicates in same row
            #--------------------------------------------------------------------------
            for doc in batch_docs:
                candidates_by_column = doc["candidates"]
                for col_index, candidates in candidates_by_column.items():
                    top_candidates_for_freq = candidates[:top_n_for_type_freq]

                    # Collect distinct type IDs from those top candidates
                    row_qids = set()
                    for cand in top_candidates_for_freq:
                        for t_dict in cand.get("types", []):
                            qid = t_dict.get("id")
                            if qid:
                                row_qids.add(qid)

                    # Increase counts for each distinct type in this row
                    for qid in row_qids:
                        type_freq_by_column[col_index][qid] += 1

                    # Mark that this row *had* that column (so we can normalize)
                    rows_count_by_column[col_index] += 1

            #--------------------------------------------------------------------------
            # 2) Convert raw counts to frequencies in [0..1].
            #    We'll keep frequencies for ALL types, not just the top 5.
            #--------------------------------------------------------------------------
            for col_index, freq_counter in type_freq_by_column.items():
                row_count = rows_count_by_column[col_index]
                if row_count == 0:
                    continue

                # Convert each type's raw count => ratio in [0..1]
                for qid in freq_counter:
                    freq_counter[qid] = freq_counter[qid] / row_count

            #--------------------------------------------------------------------------
            # 3) Assign new features (typeFreq1..typeFreq5) for each candidate
            #    by looking up *all* its types' frequencies, sorting desc, and
            #    storing the top 5.
            #--------------------------------------------------------------------------
            for doc in batch_docs:
                candidates_by_column = doc["candidates"]
                for col_index, candidates in candidates_by_column.items():
                    # If we never built a freq for this column, skip
                    if col_index not in type_freq_by_column:
                        continue

                    freq_counter = type_freq_by_column[col_index]

                    for cand in candidates:
                        # Ensure we have a features dict
                        if "features" not in cand:
                            cand["features"] = {}

                        # Gather candidate's type IDs
                        cand_qids = [t_obj.get("id") for t_obj in cand.get("types", []) if t_obj.get("id")]

                        # Find the frequency for each type
                        cand_type_freqs = []
                        for qid in cand_qids:
                            cand_type_freqs.append(freq_counter.get(qid, 0.0))

                        # Sort descending
                        cand_type_freqs.sort(reverse=True)

                        # Assign typeFreq1..typeFreq5
                        for i in range(1, 6):
                            # If the candidate has fewer than i types, default to 0
                            freq_val = cand_type_freqs[i-1] if (i-1) < len(cand_type_freqs) else 0.0
                            cand["features"][f"typeFreq{i}"] = freq_val

            #--------------------------------------------------------------------------
            # 4) Build final feature matrix & do ML predictions
            #--------------------------------------------------------------------------
            all_candidates = []
            doc_info = []
            for doc in batch_docs:
                row_index = doc["row_id"]
                candidates_by_column = doc["candidates"]
                for col_index, candidates in candidates_by_column.items():
                    features_list = [self.extract_features(c) for c in candidates]
                    all_candidates.extend(features_list)

                    # doc_info: track the location of each candidate
                    doc_info.extend([
                        (doc["_id"], row_index, col_index, idx)
                        for idx in range(len(candidates))
                    ])

            if len(all_candidates) == 0:
                print(f"No candidates to predict for dataset={dataset_name}, table={table_name}. Skipping...")
                return

            candidate_features = np.array(all_candidates)
            print(f"Predicting scores for {len(candidate_features)} candidates...")
            ml_scores = model.predict(candidate_features, batch_size=128)[:, 1]
            print("Scores predicted.")

            # Assign new ML scores back to candidates
            for i, (doc_id, row_index, col_index, cand_idx) in enumerate(doc_info):
                candidate = doc_map[doc_id]["candidates"][col_index][cand_idx]
                candidate["score"] = float(ml_scores[i])

            #--------------------------------------------------------------------------
            # 5) Sort by final 'score' and trim to max_candidates; update DB
            #--------------------------------------------------------------------------
            for doc_id, doc in doc_map.items():
                row_index = doc["row_id"]
                updated_candidates_by_column = {}
                for col_index, candidates in doc["candidates"].items():
                    ranked_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
                    updated_candidates_by_column[col_index] = ranked_candidates[: self.max_candidates]

                # Mark doc as ML-ranked
                training_collection.update_one(
                    {"_id": doc_id},
                    {"$set": {"candidates": doc["candidates"], "ml_ranked": True}}
                )

                # Update final results in input_collection
                input_collection.update_one(
                    {"dataset_name": dataset_name, "table_name": table_name, "row_id": row_index},
                    {"$set": {"el_results": updated_candidates_by_column}}
                )

            processed_count += len(batch_docs)
            progress = min((processed_count / total_count) * 100, 100.0)
            print(f"ML ranking progress: {progress:.2f}% completed")

            # Update progress in table_trace
            table_trace_collection.update_one(
                {"dataset_name": dataset_name, "table_name": table_name},
                {"$set": {"ml_ranking_progress": progress}},
                upsert=True
            )

        # Mark the table as COMPLETED
        table_trace_collection.update_one(
            {"dataset_name": dataset_name, "table_name": table_name},
            {"$set": {"ml_ranking_status": "COMPLETED"}}
        )
        print("ML ranking completed.")
    
    def extract_features(self, candidate):
        numerical_features = [
            'ntoken_mention', 'length_mention', 'ntoken_entity', 'length_entity',
            'popularity', 'ed_score', 'desc', 'descNgram',
            'bow_similarity',
            'kind', 'NERtype', 'column_NERtype'
        ]
        return [candidate['features'].get(feature, 0.0) for feature in numerical_features]
    
    def load_ml_model(self):
        from tensorflow.keras.models import load_model
        return load_model(self.model_path)
