from __future__ import annotations

import time
from typing import Callable, TypeVar

from pymongo import ASCENDING, MongoClient
from pymongo.errors import AutoReconnect, ConnectionFailure, NetworkTimeout, NotPrimaryError

from config import settings


T = TypeVar("T")

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGO_URI, retryReads=True, retryWrites=True)
    return _client


def get_db():
    return get_client()[settings.MONGO_DB]


def init_indexes(db) -> None:
    db.jobs.create_index("job_id", unique=True)
    db.jobs.create_index("idempotency_key", unique=True, sparse=True)
    db.jobs.create_index([("status", ASCENDING), ("lease_expires_at", ASCENDING)])

    db.job_parts.create_index([("job_id", ASCENDING), ("part_number", ASCENDING)], unique=True)

    db.job_results.create_index([("job_id", ASCENDING), ("segment_idx", ASCENDING)], unique=True)
    db.job_results.create_index(
        [
            ("job_id", ASCENDING),
            ("sort_key.part", ASCENDING),
            ("sort_key.row", ASCENDING),
            ("sort_key.col", ASCENDING),
        ]
    )

    db.job_events.create_index([("job_id", ASCENDING), ("created_at", ASCENDING)])


def with_retry(fn: Callable[[], T], *, max_attempts: int = 3, base_delay: float = 0.1) -> T:
    for attempt in range(max_attempts):
        try:
            return fn()
        except (AutoReconnect, NetworkTimeout, ConnectionFailure, NotPrimaryError):
            if attempt >= max_attempts - 1:
                raise
            time.sleep(base_delay * (2**attempt))
    return fn()
