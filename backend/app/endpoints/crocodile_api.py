from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response, status
from fastapi.security import APIKeyHeader
from pymongo.errors import DuplicateKeyError

from config import settings
from el_jobs.db import get_db, with_retry
from el_jobs.models import (
    CellResult,
    HealthResponse,
    JobCancelResponse,
    JobCreateRequest,
    JobCreateResponse,
    JobFinalizeRequest,
    JobFinalizeResponse,
    JobPartUploadRequest,
    JobPartUploadResponse,
    JobStatus,
    JobStatusResponse,
    ResultsPage,
    UploadInstructions,
)
from el_jobs.utils import (
    decode_cursor,
    encode_cursor,
    hash_request,
    log_json,
    metrics_incr,
    normalize_rows,
    utcnow,
)

logger = logging.getLogger("crocodile.el_api")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Depends(api_key_header)) -> str:
    if settings.CROCODILE_API_KEY:
        if not api_key or api_key != settings.CROCODILE_API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    else:
        if not api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    return api_key


def ensure_request_size(request: Request) -> None:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.EL_MAX_REQUEST_BYTES:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Payload too large")


router = APIRouter(dependencies=[Depends(verify_api_key)])
health_router = APIRouter()


def normalize_status(value: str) -> JobStatus:
    if value == "succeeded":
        return JobStatus.done
    return JobStatus(value)


@router.post(
    "/jobs",
    response_model=JobCreateResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_job(
    request: Request,
    response: Response,
    payload: JobCreateRequest = Body(...),
) -> JobCreateResponse:
    ensure_request_size(request)

    if not payload.header:
        raise HTTPException(status_code=422, detail="Header is required")
    if len(payload.header) > settings.EL_MAX_COLUMNS:
        raise HTTPException(status_code=422, detail="Too many columns")

    header = payload.header
    link_columns = payload.link_columns
    if not link_columns:
        raise HTTPException(status_code=422, detail="link_columns is required")

    missing_cols = [col for col in link_columns if col not in header]
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"link_columns not in header: {missing_cols}")

    idempotency_key = request.headers.get("Idempotency-Key")

    mode = payload.mode
    is_inline = mode == "inline"

    if is_inline:
        if payload.rows is None:
            raise HTTPException(status_code=422, detail="rows are required for inline mode")
        if len(payload.rows) > settings.EL_INLINE_MAX_ROWS:
            raise HTTPException(status_code=422, detail="Inline row limit exceeded; use multipart")

        normalized_rows = normalize_rows(payload.rows, header, part_number=0, inline=True)
        now = utcnow()

        config_hash = hash_request(
            {
                "config": payload.config,
                "top_k": payload.top_k,
                "link_columns": link_columns,
                "header": header,
            }
        )

        request_hash = hash_request(
            {
                "mode": mode,
                "header": header,
                "rows": normalized_rows,
                "link_columns": link_columns,
                "top_k": payload.top_k,
                "config": payload.config,
            }
        )

        db = get_db()
        jobs = db.jobs
        job_parts = db.job_parts

        if idempotency_key:
            existing = jobs.find_one({"idempotency_key": idempotency_key})
            if existing:
                if existing.get("request_hash") != request_hash:
                    raise HTTPException(status_code=409, detail="Idempotency key conflict")
                response.status_code = status.HTTP_202_ACCEPTED
                return JobCreateResponse(
                    job_id=existing["job_id"],
                    status=normalize_status(existing["status"]),
                    mode="inline",
                    message="Idempotent replay",
                )

        job_id = str(uuid.uuid4())
        job_doc = {
            "job_id": job_id,
            "mode": "inline",
            "status": JobStatus.queued.value,
            "header": header,
            "link_columns": link_columns,
            "top_k": payload.top_k,
            "config": payload.config,
            "config_hash": config_hash,
            "created_at": now,
            "updated_at": now,
            "ingest": {
                "expected_parts": 1,
                "expected_rows": len(normalized_rows),
                "received_parts": 1,
                "received_rows": len(normalized_rows),
                "completed_at": now,
            },
            "progress": {"part_number": 0, "row_index": 0},
            "results": {"segments": 0, "cells": 0},
            "request_hash": request_hash,
            "cancel_requested": False,
        }
        if idempotency_key:
            job_doc["idempotency_key"] = idempotency_key

        try:
            with_retry(lambda: jobs.insert_one(job_doc))
            with_retry(
                lambda: job_parts.insert_one(
                    {
                        "job_id": job_id,
                        "part_number": 0,
                        "rows": normalized_rows,
                        "row_count": len(normalized_rows),
                        "created_at": now,
                    }
                )
            )
        except DuplicateKeyError:
            existing = jobs.find_one({"idempotency_key": idempotency_key}) if idempotency_key else None
            if existing and existing.get("request_hash") == request_hash:
                response.status_code = status.HTTP_202_ACCEPTED
                return JobCreateResponse(
                    job_id=existing["job_id"],
                    status=normalize_status(existing["status"]),
                    mode="inline",
                    message="Idempotent replay",
                )
            raise HTTPException(status_code=409, detail="Job already exists")

        metrics_incr("jobs_created", tags={"mode": "inline"})
        response.status_code = status.HTTP_202_ACCEPTED
        return JobCreateResponse(
            job_id=job_id,
            status=JobStatus.queued,
            mode="inline",
            message="Job accepted",
        )

    if payload.rows is not None:
        raise HTTPException(status_code=422, detail="rows must be omitted for multipart mode")

    now = utcnow()
    config_hash = hash_request(
        {
            "config": payload.config,
            "top_k": payload.top_k,
            "link_columns": link_columns,
            "header": header,
        }
    )

    request_hash = hash_request(
        {
            "mode": mode,
            "header": header,
            "link_columns": link_columns,
            "top_k": payload.top_k,
            "config": payload.config,
            "total_parts": payload.total_parts,
            "total_rows": payload.total_rows,
        }
    )

    db = get_db()
    jobs = db.jobs

    if idempotency_key:
        existing = jobs.find_one({"idempotency_key": idempotency_key})
        if existing:
            if existing.get("request_hash") != request_hash:
                raise HTTPException(status_code=409, detail="Idempotency key conflict")
            response.status_code = status.HTTP_201_CREATED
            return JobCreateResponse(
                job_id=existing["job_id"],
                status=normalize_status(existing["status"]),
                mode="multipart",
                message="Idempotent replay",
                upload=UploadInstructions(
                    upload_parts_url=f"/jobs/{existing['job_id']}/parts",
                    finalize_url=f"/jobs/{existing['job_id']}/finalize",
                ),
            )

    job_id = str(uuid.uuid4())
    job_doc = {
        "job_id": job_id,
        "mode": "multipart",
        "status": JobStatus.ingesting.value,
        "header": header,
        "link_columns": link_columns,
        "top_k": payload.top_k,
        "config": payload.config,
        "config_hash": config_hash,
        "created_at": now,
        "updated_at": now,
        "ingest": {
            "expected_parts": payload.total_parts,
            "expected_rows": payload.total_rows,
            "received_parts": 0,
            "received_rows": 0,
            "completed_at": None,
        },
        "progress": {"part_number": 0, "row_index": 0},
        "results": {"segments": 0, "cells": 0},
        "request_hash": request_hash,
        "cancel_requested": False,
    }
    if idempotency_key:
        job_doc["idempotency_key"] = idempotency_key

    try:
        with_retry(lambda: jobs.insert_one(job_doc))
    except DuplicateKeyError:
        existing = jobs.find_one({"idempotency_key": idempotency_key}) if idempotency_key else None
        if existing and existing.get("request_hash") == request_hash:
            response.status_code = status.HTTP_201_CREATED
            return JobCreateResponse(
                job_id=existing["job_id"],
                status=normalize_status(existing["status"]),
                mode="multipart",
                message="Idempotent replay",
                upload=UploadInstructions(
                    upload_parts_url=f"/jobs/{existing['job_id']}/parts",
                    finalize_url=f"/jobs/{existing['job_id']}/finalize",
                ),
            )
        raise HTTPException(status_code=409, detail="Job already exists")

    metrics_incr("jobs_created", tags={"mode": "multipart"})
    response.status_code = status.HTTP_201_CREATED
    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.ingesting,
        mode="multipart",
        message="Multipart job created",
        upload=UploadInstructions(
            upload_parts_url=f"/jobs/{job_id}/parts",
            finalize_url=f"/jobs/{job_id}/finalize",
        ),
    )


@router.post("/jobs/{job_id}/parts", response_model=JobPartUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_part(
    request: Request, job_id: str, payload: JobPartUploadRequest = Body(...)
) -> JobPartUploadResponse:
    ensure_request_size(request)

    if len(payload.rows) > settings.EL_PART_MAX_ROWS:
        raise HTTPException(status_code=422, detail="Part row limit exceeded")

    db = get_db()
    jobs = db.jobs
    job_parts = db.job_parts

    job = jobs.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["mode"] != "multipart":
        raise HTTPException(status_code=409, detail="Job is not multipart")
    if job["status"] not in [JobStatus.ingesting.value, JobStatus.created.value]:
        raise HTTPException(status_code=409, detail="Job is not accepting parts")

    header = job["header"]
    normalized_rows = normalize_rows(payload.rows, header, part_number=payload.part_number, inline=False)
    now = utcnow()

    try:
        with_retry(
            lambda: job_parts.insert_one(
                {
                    "job_id": job_id,
                    "part_number": payload.part_number,
                    "rows": normalized_rows,
                    "row_count": len(normalized_rows),
                    "created_at": now,
                }
            )
        )
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Duplicate part_number")

    with_retry(
        lambda: jobs.update_one(
            {"job_id": job_id},
            {
                "$inc": {"ingest.received_parts": 1, "ingest.received_rows": len(normalized_rows)},
                "$set": {"updated_at": now},
            },
        )
    )

    metrics_incr("parts_uploaded")
    return JobPartUploadResponse(
        job_id=job_id,
        part_number=payload.part_number,
        received_rows=len(normalized_rows),
        status=JobStatus.ingesting,
    )


@router.post("/jobs/{job_id}/finalize", response_model=JobFinalizeResponse)
def finalize_job(job_id: str, payload: JobFinalizeRequest = Body(...)) -> JobFinalizeResponse:
    db = get_db()
    jobs = db.jobs
    job_parts = db.job_parts

    job = jobs.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["mode"] != "multipart":
        raise HTTPException(status_code=409, detail="Job is not multipart")
    if job["status"] not in [JobStatus.ingesting.value, JobStatus.created.value]:
        raise HTTPException(status_code=409, detail="Job cannot be finalized")

    expected_parts = payload.total_parts or job.get("ingest", {}).get("expected_parts")
    expected_rows = payload.total_rows or job.get("ingest", {}).get("expected_rows")

    actual_parts = job_parts.count_documents({"job_id": job_id})
    actual_rows = db.job_parts.aggregate(
        [
            {"$match": {"job_id": job_id}},
            {"$group": {"_id": None, "total": {"$sum": "$row_count"}}},
        ]
    )
    actual_rows_total = 0
    for doc in actual_rows:
        actual_rows_total = doc.get("total", 0)

    if expected_parts is not None and expected_parts != actual_parts:
        raise HTTPException(status_code=409, detail="Part count mismatch")
    if expected_rows is not None and expected_rows != actual_rows_total:
        raise HTTPException(status_code=409, detail="Row count mismatch")

    now = utcnow()
    with_retry(
        lambda: jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": JobStatus.queued.value,
                    "updated_at": now,
                    "ingest.expected_parts": expected_parts,
                    "ingest.expected_rows": expected_rows,
                    "ingest.received_parts": actual_parts,
                    "ingest.received_rows": actual_rows_total,
                    "ingest.completed_at": now,
                }
            },
        )
    )

    metrics_incr("jobs_finalized")
    return JobFinalizeResponse(job_id=job_id, status=JobStatus.queued, message="Job queued")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = get_db().jobs.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=normalize_status(job["status"]),
        mode=job["mode"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        config_hash=job.get("config_hash", ""),
        ingest=job.get("ingest", {}),
        progress=job.get("progress", {}),
        results=job.get("results", {}),
        error=job.get("error"),
    )


@router.post("/jobs/{job_id}:cancel", response_model=JobCancelResponse)
def cancel_job(job_id: str) -> JobCancelResponse:
    now = utcnow()
    jobs = get_db().jobs

    job = jobs.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] in [JobStatus.done.value, JobStatus.failed.value, JobStatus.cancelled.value, "succeeded"]:
        return JobCancelResponse(
            job_id=job_id,
            status=normalize_status(job["status"]),
            message="Job already terminal",
        )

    jobs.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": JobStatus.cancelled.value,
                "cancel_requested": True,
                "updated_at": now,
                "cancelled_at": now,
            }
        },
    )

    metrics_incr("jobs_cancelled")
    return JobCancelResponse(job_id=job_id, status=JobStatus.cancelled, message="Job cancelled")


@router.get("/jobs/{job_id}/results", response_model=ResultsPage)
def get_results(
    job_id: str,
    cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=settings.EL_DEFAULT_LIMIT, ge=1),
) -> ResultsPage:
    if limit > settings.EL_MAX_LIMIT:
        raise HTTPException(status_code=422, detail="limit exceeds maximum")

    db = get_db()
    job = db.jobs.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = normalize_status(job["status"])
    if job_status in [JobStatus.created, JobStatus.ingesting, JobStatus.queued]:
        raise HTTPException(status_code=409, detail="Results not available yet")

    start_segment = 0
    start_offset = 0
    if cursor:
        try:
            payload = decode_cursor(cursor)
            start_segment = payload["segment_idx"]
            start_offset = payload["offset"]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    results: List[CellResult] = []
    next_cursor = None
    has_more = False

    segments_cursor = db.job_results.find(
        {"job_id": job_id, "segment_idx": {"$gte": start_segment}}
    ).sort("segment_idx", 1)

    for segment in segments_cursor:
        segment_idx = segment["segment_idx"]
        seg_results: List[Dict[str, Any]] = segment.get("results", [])
        offset = start_offset if segment_idx == start_segment else 0
        while offset < len(seg_results) and len(results) < limit:
            item = seg_results[offset]
            results.append(
                CellResult(
                    row_id=item["row_id"],
                    col_id=item["col_id"],
                    mention=item.get("mention", ""),
                    candidates=item.get("candidates", []),
                )
            )
            last_sort_key = item.get("sort_key")
            offset += 1

        if len(results) >= limit:
            more_in_segment = offset < len(seg_results)
            more_segments = db.job_results.find_one(
                {"job_id": job_id, "segment_idx": {"$gt": segment_idx}}
            )
            has_more = more_in_segment or bool(more_segments)
            if has_more:
                next_cursor = encode_cursor(
                    segment_idx=segment_idx,
                    offset=offset,
                    sort_key=last_sort_key if "last_sort_key" in locals() else None,
                )
            break

        start_offset = 0

    return ResultsPage(results=results, next_cursor=next_cursor, has_more=has_more, job_status=job_status)


@health_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", time=utcnow())


@health_router.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    db = get_db()
    with_retry(lambda: db.command("ping"))
    return HealthResponse(status="ready", time=utcnow())
