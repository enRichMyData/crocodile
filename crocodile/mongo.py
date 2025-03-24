import os
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, TypeVar

from pymongo import ASCENDING, MongoClient
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
        error_log_collection_name: str = "error_logs",
    ) -> None:
        self.mongo_uri: str = mongo_uri
        self.db_name: str = db_name
        self.error_log_collection_name: str = error_log_collection_name

    def get_db(self) -> Database:
        client: MongoClient = MongoConnectionManager.get_client(self.mongo_uri)
        return client[self.db_name]

    def update_document(
        self,
        collection: Collection,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> UpdateResult:
        return collection.update_one(query, update, upsert=upsert)

    def update_documents(
        self,
        collection: Collection,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> UpdateResult:
        return collection.update_many(query, update, upsert=upsert)

    def find_documents(
        self,
        collection: Collection,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        def query_function(
            query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None
        ) -> List[Dict[str, Any]]:
            cursor = collection.find(query, projection)
            if limit is not None:
                cursor = cursor.limit(limit)
            return list(cursor)

        return query_function(query, projection)

    def count_documents(self, collection: Collection, query: Dict[str, Any]) -> int:
        return collection.count_documents(query)

    def find_one_document(
        self,
        collection: Collection,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return collection.find_one(query, projection=projection)

    def find_one_and_update(
        self,
        collection: Collection,
        query: Dict[str, Any],
        update: Dict[str, Any],
        return_document: bool = False,
    ) -> Optional[Dict[str, Any]]:
        return collection.find_one_and_update(query, update, return_document=return_document)

    def insert_one_document(
        self, collection: Collection, document: Dict[str, Any]
    ) -> InsertOneResult:
        return collection.insert_one(document)

    def insert_many_documents(
        self, collection: Collection, documents: List[Dict[str, Any]]
    ) -> InsertManyResult:
        return collection.insert_many(documents)

    def delete_documents(self, collection: Collection, query: Dict[str, Any]) -> DeleteResult:
        return collection.delete_many(query)

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

    # Ensure indexes for uniqueness and performance
    def create_indexes(self):
        db: Database = self.get_db()
        input_collection: Collection = db["input_data"]

        input_collection.create_index(
            [("dataset_name", ASCENDING), ("table_name", ASCENDING)]
        )  # Ensure fast retrieval of items by dataset and table
        input_collection.create_index(
            [("dataset_name", ASCENDING), ("table_name", ASCENDING), ("row_id", ASCENDING)],
            unique=True,
        )
        input_collection.create_index(
            [("dataset_name", ASCENDING), ("table_name", ASCENDING), ("status", ASCENDING)]
        )  # Ensure fast retrieval of items by status
        input_collection.create_index(
            [("status", ASCENDING)]
        )  # Ensure fast retrieval of items by status
        input_collection.create_index(
            [
                ("dataset_name", ASCENDING),
                ("table_name", ASCENDING),
                ("status", ASCENDING),
                ("candidates", ASCENDING),
            ]
        )
