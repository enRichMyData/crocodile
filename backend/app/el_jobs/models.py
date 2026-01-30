from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    created = "created"
    ingesting = "ingesting"
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class Candidate(BaseModel):
    entity_id: str
    label: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class CellResult(BaseModel):
    row_id: Union[int, str]
    col_id: Union[int, str]
    mention: str
    candidates: List[Candidate]


class RowCells(BaseModel):
    row_id: Optional[Union[int, str]] = None
    cells: List[Any]


RowInput = Union[RowCells, Dict[str, Any]]


class JobCreateRequest(BaseModel):
    mode: Literal["inline", "multipart"] = "inline"
    header: List[str]
    rows: Optional[List[RowInput]] = None
    link_columns: List[str]
    top_k: int = Field(default=5, ge=1, le=100)
    config: Dict[str, Any] = Field(default_factory=dict)
    total_parts: Optional[int] = Field(default=None, ge=1)
    total_rows: Optional[int] = Field(default=None, ge=0)


class UploadInstructions(BaseModel):
    upload_parts_url: str
    finalize_url: str


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    mode: Literal["inline", "multipart"]
    message: str
    upload: Optional[UploadInstructions] = None


class JobPartUploadRequest(BaseModel):
    part_number: int = Field(ge=0)
    rows: List[RowInput]


class JobPartUploadResponse(BaseModel):
    job_id: str
    part_number: int
    received_rows: int
    status: JobStatus


class JobFinalizeRequest(BaseModel):
    total_parts: Optional[int] = Field(default=None, ge=1)
    total_rows: Optional[int] = Field(default=None, ge=0)


class JobFinalizeResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobCancelResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobIngestInfo(BaseModel):
    expected_parts: Optional[int] = None
    expected_rows: Optional[int] = None
    received_parts: int = 0
    received_rows: int = 0
    completed_at: Optional[datetime] = None


class JobProgressInfo(BaseModel):
    part_number: int = 0
    row_index: int = 0


class JobResultsInfo(BaseModel):
    segments: int = 0
    cells: int = 0


class JobErrorInfo(BaseModel):
    code: Optional[str] = None
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    mode: Literal["inline", "multipart"]
    created_at: datetime
    updated_at: datetime
    config_hash: str
    ingest: JobIngestInfo
    progress: JobProgressInfo
    results: JobResultsInfo
    error: Optional[JobErrorInfo] = None


class ResultsPage(BaseModel):
    results: List[CellResult]
    next_cursor: Optional[str] = None
    has_more: bool
    job_status: JobStatus


class HealthResponse(BaseModel):
    status: str
    time: datetime
