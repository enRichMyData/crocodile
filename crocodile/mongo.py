import os
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
