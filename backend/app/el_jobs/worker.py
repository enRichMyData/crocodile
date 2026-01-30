from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from config import settings
from crocodile import Crocodile, CrocodileResultFetcher
from el_jobs.db import get_db, init_indexes, with_retry
from el_jobs.models import JobStatus
from el_jobs.utils import build_sort_key, log_json, utcnow

logger = logging.getLogger("crocodile.el_worker")
logging.basicConfig(level=logging.INFO)


def claim_job(db, owner_id: str):
    now = utcnow()
    lease_until = now + timedelta(seconds=settings.EL_WORKER_LEASE_SECONDS)
    job = db.jobs.find_one_and_update(
        {
            "status": {"$in": [JobStatus.queued.value, JobStatus.running.value]},
            "$or": [
                {"lease_expires_at": {"$exists": False}},
                {"lease_expires_at": {"$lte": now}},
            ],
        },
        {
            "$set": {
                "status": JobStatus.running.value,
                "lease_owner": owner_id,
                "lease_expires_at": lease_until,
                "updated_at": now,
            }
        },
        sort=[("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )
    if job and not job.get("started_at"):
        db.jobs.update_one({"job_id": job["job_id"]}, {"$set": {"started_at": now}})
    return job


def renew_lease(db, job_id: str, owner_id: str) -> None:
    now = utcnow()
    lease_until = now + timedelta(seconds=settings.EL_WORKER_LEASE_SECONDS)
    db.jobs.update_one(
        {"job_id": job_id, "lease_owner": owner_id},
        {"$set": {"lease_expires_at": lease_until, "updated_at": now}},
    )


def get_resume_cursor(db, job: Dict[str, Any]) -> Dict[str, int]:
    progress = job.get("progress") or {"part_number": 0, "row_index": 0}
    last_segment = db.job_results.find_one(
        {"job_id": job["job_id"]}, sort=[("segment_idx", -1)]
    )
    if last_segment and last_segment.get("cursor_after"):
        return last_segment["cursor_after"]
    return progress


def iter_rows(db, job_id: str, start_cursor: Dict[str, int]):
    start_part = start_cursor.get("part_number", 0)
    start_row = start_cursor.get("row_index", 0)

    parts_cursor = db.job_parts.find({"job_id": job_id}).sort("part_number", 1)
    for part in parts_cursor:
        part_number = part["part_number"]
        if part_number < start_part:
            continue
        rows = part.get("rows", [])
        row_start = start_row if part_number == start_part else 0
        if row_start >= len(rows):
            continue
        for row_index in range(row_start, len(rows)):
            yield part_number, row_index, rows[row_index]


def should_cancel(db, job_id: str) -> bool:
    job = db.jobs.find_one({"job_id": job_id}, {"status": 1, "cancel_requested": 1})
    if not job:
        return True
    if job.get("status") == JobStatus.cancelled.value:
        return True
    if job.get("cancel_requested"):
        return True
    return False


def insert_segment(
    db,
    job_id: str,
    segment_idx: int,
    results: List[Dict[str, Any]],
    cursor_after: Dict[str, int],
    row_count: int,
    cell_count: int,
) -> None:
    now = utcnow()
    segment_doc = {
        "job_id": job_id,
        "segment_idx": segment_idx,
        "results": results,
        "count": len(results),
        "sort_key_first": results[0]["sort_key"] if results else None,
        "sort_key_last": results[-1]["sort_key"] if results else None,
        "sort_key": results[-1]["sort_key"] if results else None,
        "cursor_after": cursor_after,
        "created_at": now,
    }

    try:
        with_retry(lambda: db.job_results.insert_one(segment_doc))
    except DuplicateKeyError:
        return

    with_retry(
        lambda: db.jobs.update_one(
            {"job_id": job_id},
            {
                "$inc": {
                    "results.segments": 1,
                    "results.cells": cell_count,
                    "processed_rows": row_count,
                },
                "$set": {"progress": cursor_after, "updated_at": now},
            },
        )
    )


def build_columns_type(header: List[str], link_columns: List[str]) -> Dict[str, Any]:
    ne_cols = {}
    ignored = []
    for idx, col_name in enumerate(header):
        if col_name in link_columns:
            ne_cols[str(idx)] = "OTHER"
        else:
            ignored.append(str(idx))
    return {"NE": ne_cols, "LIT": {}, "IGNORED": ignored}


def process_job(db, job: Dict[str, Any], owner_id: str) -> None:
    job_id = job["job_id"]
    header = job["header"]
    link_columns = job["link_columns"]
    top_k = job.get("top_k", 5)
    config = job.get("config", {})

    col_index = {name: idx for idx, name in enumerate(header)}

    # Restart-safe: clear existing segments if present for this job.
    db.job_results.delete_many({"job_id": job_id})
    db.jobs.update_one(
        {"job_id": job_id},
        {"$set": {"results": {"segments": 0, "cells": 0}, "progress": {"part_number": 0, "row_index": 0}}},
    )

    if should_cancel(db, job_id):
        db.jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": JobStatus.cancelled.value,
                    "updated_at": utcnow(),
                    "cancelled_at": utcnow(),
                }
            },
        )
        return

    row_order: List[Dict[str, Any]] = []
    rows_data: List[List[Any]] = []
    for part_number, row_index, row in iter_rows(db, job_id, {"part_number": 0, "row_index": 0}):
        row_order.append(
            {
                "part_number": part_number,
                "row_index": row_index,
                "row_id": row["row_id"],
                "cells": row["cells"],
            }
        )
        rows_data.append(row["cells"])

    if not rows_data:
        db.jobs.update_one(
            {"job_id": job_id},
            {"$set": {"status": JobStatus.done.value, "updated_at": utcnow(), "completed_at": utcnow()}},
        )
        return

    df = pd.DataFrame(rows_data, columns=header)
    columns_type = build_columns_type(header, link_columns)

    croco = Crocodile(
        input_csv=df,
        client_id=job_id,
        dataset_name=job_id,
        table_name="job",
        columns_type=columns_type,
        max_candidates_in_result=top_k,
        entity_retrieval_endpoint=(
            config.get("entity_retrieval_endpoint") if isinstance(config, dict) else None
        )
        or os.getenv("ENTITY_RETRIEVAL_ENDPOINT"),
        entity_retrieval_token=(
            config.get("entity_retrieval_token") if isinstance(config, dict) else None
        )
        or os.getenv("ENTITY_RETRIEVAL_TOKEN"),
        entity_bow_endpoint=(
            config.get("entity_bow_endpoint") if isinstance(config, dict) else None
        )
        or os.getenv("ENTITY_BOW_ENDPOINT"),
        candidate_retrieval_limit=(
            int(config.get("candidate_retrieval_limit", 16)) if isinstance(config, dict) else 16
        ),
        max_workers=int(config.get("max_workers", 4)) if isinstance(config, dict) else 4,
        ml_ranking_workers=int(config.get("ml_ranking_workers", 2)) if isinstance(config, dict) else 2,
        model_path=config.get("model_path") if isinstance(config, dict) else None,
        save_output_to_csv=False,
        return_dataframe=False,
        mongo_uri=settings.MONGO_URI,
        db_name="crocodile",
    )
    croco.run()

    if should_cancel(db, job_id):
        db.jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": JobStatus.cancelled.value,
                    "updated_at": utcnow(),
                    "cancelled_at": utcnow(),
                }
            },
        )
        return

    fetcher = CrocodileResultFetcher(
        client_id=job_id,
        dataset_name=job_id,
        table_name="job",
        mongo_uri=settings.MONGO_URI,
        db_name="crocodile",
    )
    croco_db = fetcher.get_db()
    input_collection = croco_db["input_data"]

    current_segment: List[Dict[str, Any]] = []
    segment_row_count = 0
    segment_cell_count = 0
    segment_idx = 0
    last_row_position: Tuple[int, int] | None = None

    def flush_segment():
        nonlocal segment_idx, current_segment, segment_row_count, segment_cell_count, last_row_position
        if not current_segment or last_row_position is None:
            return
        cursor_after = {"part_number": last_row_position[0], "row_index": last_row_position[1] + 1}
        insert_segment(
            db,
            job_id,
            segment_idx,
            current_segment,
            cursor_after,
            segment_row_count,
            segment_cell_count,
        )
        segment_idx += 1
        current_segment = []
        segment_row_count = 0
        segment_cell_count = 0
        last_row_position = None

    cursor = input_collection.find(
        {"client_id": job_id, "dataset_name": job_id, "table_name": "job"},
        projection={"row_id": 1, "el_results": 1},
    ).sort("row_id", 1)

    for doc in cursor:
        croco_row_idx = doc.get("row_id")
        if croco_row_idx is None or croco_row_idx >= len(row_order):
            continue
        row_meta = row_order[croco_row_idx]
        part_number = row_meta["part_number"]
        row_index = row_meta["row_index"]
        original_row_id = row_meta["row_id"]
        cells = row_meta["cells"]

        el_results = doc.get("el_results", {})
        row_results: List[Dict[str, Any]] = []
        for col_name in link_columns:
            col_idx = col_index[col_name]
            mention = str(cells[col_idx]) if cells[col_idx] is not None else ""
            candidates_raw = el_results.get(str(col_idx), [])
            candidates: List[Dict[str, Any]] = []
            for cand in candidates_raw[:top_k]:
                metadata = {k: v for k, v in cand.items() if k not in {"id", "name", "score"}}
                candidates.append(
                    {
                        "entity_id": cand.get("id", ""),
                        "label": cand.get("name", ""),
                        "score": cand.get("score", 0.0),
                        "metadata": metadata or None,
                    }
                )

            sort_key = build_sort_key(part_number, row_index, col_idx)
            row_results.append(
                {
                    "row_id": original_row_id,
                    "col_id": col_name,
                    "mention": mention,
                    "candidates": candidates,
                    "sort_key": sort_key,
                }
            )

        row_results.sort(key=lambda r: r["sort_key"]["col"])

        if current_segment and (segment_cell_count + len(row_results) > settings.EL_RESULT_SEGMENT_SIZE):
            flush_segment()

        current_segment.extend(row_results)
        segment_row_count += 1
        segment_cell_count += len(row_results)
        last_row_position = (part_number, row_index)

    flush_segment()

    if should_cancel(db, job_id):
        db.jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": JobStatus.cancelled.value,
                    "updated_at": utcnow(),
                    "cancelled_at": utcnow(),
                }
            },
        )
        return

    db.jobs.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": JobStatus.done.value,
                "updated_at": utcnow(),
                "completed_at": utcnow(),
            }
        },
    )


def worker_loop():
    db = get_db()
    init_indexes(db)
    owner_id = str(uuid.uuid4())

    log_json(logger, "worker_started", owner_id=owner_id)

    while True:
        job = claim_job(db, owner_id)
        if not job:
            time.sleep(settings.EL_WORKER_POLL_INTERVAL_S)
            continue

        job_id = job["job_id"]
        try:
            log_json(logger, "job_claimed", job_id=job_id)
            process_job(db, job, owner_id)
            log_json(logger, "job_done", job_id=job_id)
        except Exception as exc:
            log_json(logger, "job_failed", job_id=job_id, error=str(exc))
            db.jobs.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": JobStatus.failed.value,
                        "updated_at": utcnow(),
                        "error": {"message": str(exc), "code": "worker_error"},
                    }
                },
            )


if __name__ == "__main__":
    worker_loop()
