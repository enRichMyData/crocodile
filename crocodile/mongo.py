import os
import time
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, TypeVar

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.results import DeleteResult, InsertManyResult, InsertOneResult, UpdateResult

T = TypeVar("T")


class MongoCache:
    """MongoDB-based cache for storing key-value pairs."""

    def __init__(self, db: Database, collection_name: str) -> None:
        self.collection: Collection = db[collection_name]
        self.collection.create_index("key", unique=True)

    def get(self, key: str) -> Optional[Any]:
        result: Optional[Dict[str, Any]] = self.collection.find_one({"key": key})
        if result:
            return result["value"]
        return None

    def put(self, key: str, value: Any) -> None:
        self.collection.update_one({"key": key}, {"$set": {"value": value}}, upsert=True)


class MongoConnectionManager:
    _instances: Dict[int, MongoClient] = {}
    _lock: Lock = Lock()

    @classmethod
    def get_client(cls, mongo_uri: str) -> MongoClient:
        pid: int = os.getpid()
        with cls._lock:
            if pid not in cls._instances:
                client: MongoClient = MongoClient(
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
    def close_connection(cls, pid: Optional[int] = None) -> None:
        if pid is None:
            pid = os.getpid()
        with cls._lock:
            if pid in cls._instances:
                cls._instances[pid].close()
                del cls._instances[pid]

    @classmethod
    def close_all_connections(cls) -> None:
        with cls._lock:
            for client in cls._instances.values():
                client.close()
            cls._instances.clear()


class MongoWrapper:
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        timing_collection_name: str = "timing_trace",
        error_log_collection_name: str = "error_log",
        table_trace_collection_name: str = "table_trace"
    ) -> None:
        self.mongo_uri: str = mongo_uri
        self.db_name: str = db_name
        self.timing_collection_name: str = timing_collection_name
        self.error_log_collection_name: str = error_log_collection_name
        self.table_trace_collection_name = table_trace_collection_name
      
    def get_db(self) -> Database:
        client: MongoClient = MongoConnectionManager.get_client(self.mongo_uri)
        return client[self.db_name]

    def time_mongo_operation(
        self, operation_name: str, query_function: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        start_time: float = time.perf_counter()
        db: Database = self.get_db()
        timing_trace_collection: Collection = db[self.timing_collection_name]
        try:
            result: T = query_function(*args, **kwargs)
        except Exception as e:
            end_time: float = time.perf_counter()
            duration: float = end_time - start_time
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
            end_time: float = time.perf_counter()
            duration: float = end_time - start_time
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

    def update_document(
        self,
        collection: Collection,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> UpdateResult:
        operation_name: str = f"update_document:{collection.name}"
        return self.time_mongo_operation(
            operation_name, collection.update_one, query, update, upsert=upsert
        )

    def update_documents(
        self,
        collection: Collection,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> UpdateResult:
        operation_name: str = f"update_documents:{collection.name}"
        return self.time_mongo_operation(
            operation_name, collection.update_many, query, update, upsert=upsert
        )

    def find_documents(
        self,
        collection: Collection,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        operation_name: str = f"find_documents:{collection.name}"

        def query_function(
            query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None
        ) -> List[Dict[str, Any]]:
            cursor = collection.find(query, projection)
            if limit is not None:
                cursor = cursor.limit(limit)
            return list(cursor)

        return self.time_mongo_operation(operation_name, query_function, query, projection)

    def count_documents(self, collection: Collection, query: Dict[str, Any]) -> int:
        operation_name: str = f"count_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.count_documents, query)

    def find_one_document(
        self,
        collection: Collection,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        operation_name: str = f"find_one_document:{collection.name}"
        return self.time_mongo_operation(
            operation_name, collection.find_one, query, projection=projection
        )

    def find_one_and_update(
        self,
        collection: Collection,
        query: Dict[str, Any],
        update: Dict[str, Any],
        return_document: bool = False,
    ) -> Optional[Dict[str, Any]]:
        operation_name: str = f"find_one_and_update:{collection.name}"
        return self.time_mongo_operation(
            operation_name,
            collection.find_one_and_update,
            query,
            update,
            return_document=return_document,
        )

    def insert_one_document(
        self, collection: Collection, document: Dict[str, Any]
    ) -> InsertOneResult:
        operation_name: str = f"insert_one_document:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.insert_one, document)

    def insert_many_documents(
        self, collection: Collection, documents: List[Dict[str, Any]]
    ) -> InsertManyResult:
        operation_name: str = f"insert_many_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.insert_many, documents)

    def delete_documents(self, collection: Collection, query: Dict[str, Any]) -> DeleteResult:
        operation_name: str = f"delete_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.delete_many, query)

    def log_time(
        self,
        operation_name: str,
        dataset_name: str,
        table_name: str,
        start_time: float,
        end_time: float,
        details: Optional[Any] = None,
    ) -> None:
        db: Database = self.get_db()
        timing_collection: Collection = db[self.timing_collection_name]
        duration: float = end_time - start_time
        log_entry: Dict[str, Any] = {
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

    def log_to_db(
        self, level: str, message: str, trace: Optional[str] = None, attempt: Optional[int] = None
    ) -> None:
        db: Database = self.get_db()
        log_collection: Collection = db[self.error_log_collection_name]
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "traceback": trace,
        }
        if attempt is not None:
            log_entry["attempt"] = attempt
        log_collection.insert_one(log_entry)

    def log_processing_speed(self, dataset_name: str, table_name: str) -> None:
        db: Database = self.get_db()
        table_trace_collection: Collection = db[self.table_trace_collection_name]  
        trace: Optional[Dict[str, Any]] = table_trace_collection.find_one(
            {"dataset_name": dataset_name, "table_name": table_name}
        )
        if not trace:
            return
        processed_rows: Any = trace.get("processed_rows", 1)
        start_time_value: Any = trace.get("start_time")
        elapsed_time: float = (
            (datetime.now() - start_time_value).total_seconds() if start_time_value else 0
        )
        rows_per_second: float = processed_rows / elapsed_time if elapsed_time > 0 else 0
        table_trace_collection.update_one(
            {"dataset_name": dataset_name, "table_name": table_name},
            {"$set": {"rows_per_second": rows_per_second}},
        )
