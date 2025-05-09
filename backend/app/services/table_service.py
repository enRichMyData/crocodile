import os
from typing import Dict, List, Optional, Any, Set
import pandas as pd
from pymongo.database import Database
from crocodile import Crocodile
from services.data_service import DataService

class TableService:
    """
    Service for table-related operations including processing and Crocodile integration.
    """
    
    @staticmethod
    def process_table_data_with_crocodile(
        df: pd.DataFrame,
        user_id: str,
        dataset_name: str,
        table_name: str,
        classification: Dict
    ):
        """
        Process table data with Crocodile in a standardized way.
        """
        croco = Crocodile(
            input_csv=df,
            client_id=user_id,
            dataset_name=dataset_name,
            table_name=table_name,
            max_candidates=3,
            entity_retrieval_endpoint=os.environ.get("ENTITY_RETRIEVAL_ENDPOINT"),
            entity_bow_endpoint=os.environ.get("ENTITY_BOW_ENDPOINT"),
            entity_retrieval_token=os.environ.get("ENTITY_RETRIEVAL_TOKEN"),
            max_workers=8,
            candidate_retrieval_limit=10,
            model_path="./crocodile/models/default.h5",
            save_output_to_csv=False,
            columns_type=classification,
        )
        croco.run()

    @staticmethod
    def fetch_and_format_table_rows(
        db: Database,
        crocodile_db: Database,
        user_id: str,
        dataset_name: str,
        table_name: str,
        row_ids: List[int],
        header: List[str]
    ) -> List[Dict]:
        """
        Fetch rows from MongoDB and format them for API response.
        """
        from services.utils import sanitize_for_json
        
        # Build query for MongoDB
        mongo_query = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": {"$in": row_ids}
        }
        
        # Try backend DB first, then crocodile DB if needed
        raw_rows = list(db.input_data.find(mongo_query))
        if not raw_rows:
            raw_rows = list(crocodile_db.input_data.find(mongo_query))
            
        # Sort rows to match the order from row_ids
        row_id_to_idx = {row_id: i for i, row_id in enumerate(row_ids)}
        raw_rows.sort(key=lambda row: row_id_to_idx.get(row.get("row_id"), float('inf')))

        # Format rows
        rows_formatted = []
        for row in raw_rows:
            linked_entities = []
            el_results = row.get("el_results", {})

            for col_index in range(len(header)):
                candidates = el_results.get(str(col_index), [])
                if candidates:
                    sanitized_candidates = sanitize_for_json(candidates)
                    linked_entities.append({"idColumn": col_index, "candidates": sanitized_candidates})

            rows_formatted.append(
                {
                    "idRow": row.get("row_id"),
                    "data": sanitize_for_json(row.get("data", [])),
                    "linked_entities": linked_entities,
                }
            )
            
        return rows_formatted

    @staticmethod
    def get_table_status(
        db: Database,
        crocodile_db: Database,
        user_id: str,
        dataset_name: str,
        table_name: str
    ) -> str:
        """
        Determine if a table's processing is complete or still in progress.
        """
        table_status_filter = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "$or": [
                {"status": {"$in": ["TODO", "DOING"]}},
                {"ml_status": {"$in": ["TODO", "DOING"]}},
            ],
        }

        pending_docs_count = db.input_data.count_documents(table_status_filter)
        if pending_docs_count == 0:
            pending_docs_count = crocodile_db.input_data.count_documents(table_status_filter)

        return "DONE" if pending_docs_count == 0 else "DOING"
