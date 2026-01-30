from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def json_dumps_canonical(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str)


def hash_request(payload: Dict[str, Any]) -> str:
    blob = json_dumps_canonical(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def encode_cursor(*, segment_idx: int, offset: int, sort_key: Dict[str, int] | None) -> str:
    payload = {
        "v": 1,
        "segment_idx": segment_idx,
        "offset": offset,
        "sort_key": sort_key,
    }
    raw = json_dumps_canonical(payload).encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    return token


def decode_cursor(token: str) -> Dict[str, Any]:
    padding = "=" * (-len(token) % 4)
    raw = base64.urlsafe_b64decode(token + padding)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid cursor payload")
    if payload.get("v") != 1:
        raise ValueError("Unsupported cursor version")
    if not isinstance(payload.get("segment_idx"), int) or not isinstance(payload.get("offset"), int):
        raise ValueError("Invalid cursor fields")
    sort_key = payload.get("sort_key")
    if sort_key is not None:
        if not isinstance(sort_key, dict):
            raise ValueError("Invalid cursor sort_key")
    return payload


def normalize_rows(
    rows: Iterable[Any],
    header: List[str],
    *,
    part_number: int,
    inline: bool,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        row_id = None
        cells: List[Any] | None = None

        if hasattr(row, "cells"):
            row_id = getattr(row, "row_id", None)
            cells = list(getattr(row, "cells"))
        elif isinstance(row, dict) and "cells" in row and isinstance(row.get("cells"), list):
            row_id = row.get("row_id")
            cells = row.get("cells")
        elif isinstance(row, dict):
            cells = [row.get(col) for col in header]
        else:
            raise ValueError("Row must be an object with cells or a dict of column values")

        if cells is None:
            raise ValueError("Row cells are required")
        if len(cells) != len(header):
            raise ValueError("Row cells length must match header length")

        if inline:
            row_id = idx
        else:
            row_id = f"{part_number}:{idx}"

        normalized.append({"row_id": row_id, "cells": cells})
    return normalized


def log_json(logger, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str))


def metrics_incr(name: str, value: int = 1, tags: Dict[str, str] | None = None) -> None:
    _ = (name, value, tags)
    # Placeholder for metrics integration.


def build_sort_key(part_number: int, row_index: int, col_idx: int) -> Dict[str, int]:
    return {"part": part_number, "row": row_index, "col": col_idx}
