from __future__ import annotations

from typing import Any, Dict, List


class EntityLinker:
    """
    Stub engine. Replace with the actual LLM-based entity linker.
    """

    def link(
        self,
        table_header: List[str],
        rows: List[Dict[str, Any]],
        link_columns: List[str],
        top_k: int,
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        _ = (table_header, link_columns, top_k, config)
        results: List[Dict[str, Any]] = []
        for row in rows:
            row_id = row["row_id"]
            cells = row["cells"]
            for col_name in link_columns:
                col_idx = table_header.index(col_name)
                mention = str(cells[col_idx]) if cells[col_idx] is not None else ""
                results.append(
                    {
                        "row_id": row_id,
                        "col_id": col_name,
                        "mention": mention,
                        "candidates": [],
                    }
                )
        return results
