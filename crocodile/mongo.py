import os
import time
from datetime import datetime
from threading import Lock
from typing import Dict

from pymongo import MongoClient


class MongoCache:
    """MongoDB-based cache for storing key-value pairs."""

    def __init__(self, db, collection_name):
        self.collection = db[collection_name]
        self.collection.create_index("key", unique=True)

    def get(self, key):
        result = self.collection.find_one({"key": key})
        if result:
            return result["value"]
        return None

    def put(self, key, value):
        self.collection.update_one({"key": key}, {"$set": {"value": value}}, upsert=True)


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
                    socketTimeoutMS=30000,
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


class MongoWrapper:
    def __init__(
        self,
        mongo_uri,
        db_name,
        timing_collection_name="timing_trace",
        error_log_collection_name="error_log",
    ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.timing_collection_name = timing_collection_name
        self.error_log_collection_name = error_log_collection_name

    def get_db(self):
        client = MongoConnectionManager.get_client(self.mongo_uri)
        return client[self.db_name]

    def time_mongo_operation(self, operation_name, query_function, *args, **kwargs):
        start_time = time.perf_counter()
        db = self.get_db()
        timing_trace_collection = db[self.timing_collection_name]
        try:
            result = query_function(*args, **kwargs)
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            timing_trace_collection.insert_one(
                {
                    "operation_name": operation_name,
                    "start_time": datetime.fromtimestamp(start_time),
                    "end_time": datetime.fromtimestamp(end_time),
                    "duration_seconds": duration,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "error": str(e),
                    "status": "FAILED",
                }
            )
            raise
        else:
            end_time = time.perf_counter()
            duration = end_time - start_time
            timing_trace_collection.insert_one(
                {
                    "operation_name": operation_name,
                    "start_time": datetime.fromtimestamp(start_time),
                    "end_time": datetime.fromtimestamp(end_time),
                    "duration_seconds": duration,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "status": "SUCCESS",
                }
            )
            return result

    def update_document(self, collection, query, update, upsert=False):
        operation_name = f"update_document:{collection.name}"
        return self.time_mongo_operation(
            operation_name, collection.update_one, query, update, upsert=upsert
        )

    def update_documents(self, collection, query, update, upsert=False):
        operation_name = f"update_documents:{collection.name}"
        return self.time_mongo_operation(
            operation_name, collection.update_many, query, update, upsert=upsert
        )

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
        return self.time_mongo_operation(
            operation_name, collection.find_one, query, projection=projection
        )

    def find_one_and_update(self, collection, query, update, return_document=False):
        operation_name = f"find_one_and_update:{collection.name}"
        return self.time_mongo_operation(
            operation_name,
            collection.find_one_and_update,
            query,
            update,
            return_document=return_document,
        )

    def insert_one_document(self, collection, document):
        operation_name = f"insert_one_document:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.insert_one, document)

    def insert_many_documents(self, collection, documents):
        operation_name = f"insert_many_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.insert_many, documents)

    def delete_documents(self, collection, query):
        operation_name = f"delete_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.delete_many, query)

    def log_time(
        self, operation_name, dataset_name, table_name, start_time, end_time, details=None
    ):
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
            "traceback": trace,
        }
        if attempt is not None:
            log_entry["attempt"] = attempt
        log_collection.insert_one(log_entry)
