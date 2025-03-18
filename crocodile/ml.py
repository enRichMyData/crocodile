import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Tuple

import numpy as np
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import UpdateOne

from crocodile import PROJECT_ROOT
from crocodile.feature import DEFAULT_FEATURES
from crocodile.mongo import MongoConnectionManager, MongoWrapper

if TYPE_CHECKING:
    from tensorflow.keras.models import Model


class MLWorker:
    def __init__(
        self,
        worker_id: int,
        table_name: str,
        dataset_name: str,
        training_collection_name: str,
        error_log_collection_name: str,
        input_collection: str,
        model_path: str | None = None,
        batch_size: int = 100,
        max_candidates: int = 5,
        top_n_for_type_freq: int = 3,
        features: List[str] | None = None,
        **kwargs,
    ) -> None:
        super(MLWorker, self).__init__()
        self.table_name = table_name
        self.dataset_name = dataset_name
        self.training_collection_name: str = training_collection_name
        self.error_log_collection_name: str = error_log_collection_name
        self.input_collection: str = input_collection
        self.model_path: str = model_path or os.path.join(
            PROJECT_ROOT, "crocodile", "models", "default.h5"
        )
        self.batch_size: int = batch_size
        self.max_candidates: int = max_candidates
        self.top_n_for_type_freq: int = top_n_for_type_freq
        self.selected_features = features or DEFAULT_FEATURES
        self._db_name = kwargs.pop("db_name", "crocodile_db")
        self._mongo_uri = kwargs.pop("mongo_uri", "mongodb://mongodb:27017/")
        self.mongo_wrapper: MongoWrapper = MongoWrapper(
            self._mongo_uri,
            self._db_name,
            error_log_collection_name=self.error_log_collection_name,
        )

    def get_db(self) -> Database:
        client = MongoConnectionManager.get_client(self._mongo_uri)
        return client[self._db_name]

    def load_ml_model(self) -> "Model":
        from tensorflow.keras.models import load_model  # Local import as in original code

        return load_model(self.model_path)

    def run(self) -> None:
        db: Database = self.get_db()
        model: "Model" = self.load_ml_model()
        input_collection: Collection = db[self.input_collection]
        training_collection: Collection = db[self.training_collection_name]

        # Count total rows that need processing for this dataset/table
        total_docs = self.mongo_wrapper.count_documents(
            input_collection,
            {"dataset_name": self.dataset_name, "table_name": self.table_name, "status": "DONE"},
        )
        print(f"Total documents to process for ML ranking: {total_docs}")

        # Count how many rows have been ML-ranked already
        processed_count = self.mongo_wrapper.count_documents(
            training_collection,
            {
                "dataset_name": self.dataset_name,
                "table_name": self.table_name,
                "ml_ranked": True,
            },
        )
        while True:
            print(f"ML ranking progress: {processed_count}/{total_docs} documents")

            # Check if we've processed all documents
            if processed_count >= total_docs:
                print(f"ML ranking complete: {processed_count}/{total_docs} documents processed")
                return

            # Process a batch of documents
            docs_processed = self.apply_ml_ranking(self.dataset_name, self.table_name, model)

            # If no documents were processed in this batch, there might be none left
            if docs_processed == 0:
                # Double-check if there are any unranked documents left
                remaining = self.mongo_wrapper.count_documents(
                    training_collection,
                    {
                        "dataset_name": self.dataset_name,
                        "table_name": self.table_name,
                        "ml_ranked": False,
                    },
                )

                if remaining == 0:
                    print(f"No unranked documents left. Processed {processed_count}/{total_docs}.")
                    return

            processed_count += docs_processed

    def apply_ml_ranking(self, dataset_name: str, table_name: str, model: "Model") -> bool:
        """Apply ML ranking to a batch of documents. Returns True if documents were processed."""
        db: Database = self.get_db()
        training_collection: Collection = db[self.training_collection_name]
        input_collection: Collection = db[self.input_collection]

        # We'll restrict type-freq counting to the top N candidates in each row/col
        top_n_for_type_freq: int = self.top_n_for_type_freq  # e.g., 3

        batch_docs = []
        for _ in range(self.batch_size):
            doc = training_collection.find_one_and_update(
                {"dataset_name": dataset_name, "table_name": table_name, "ml_ranked": False},
                {"$set": {"ml_ranked": True}},
            )
            if doc is None:
                break
            batch_docs.append(doc)

        if not batch_docs:
            return False

        # Map documents by _id so we can update them later
        doc_map: Dict[Any, Dict[str, Any]] = {doc["_id"]: doc for doc in batch_docs}

        # For each column, we'll store a Counter of type_id -> how many rows had that type
        type_freq_by_column: DefaultDict[Any, Counter] = defaultdict(Counter)

        # Also track how many rows in the batch actually
        # had that column so we can normalize frequencies
        rows_count_by_column: Counter = Counter()

        # --------------------------------------------------------------------------
        # 1) Collect "top N" type IDs per column, ignoring duplicates in same row
        # --------------------------------------------------------------------------
        for doc in batch_docs:
            candidates_by_column: Dict[Any, List[Dict[str, Any]]] = doc["candidates"]
            for col_index, candidates in candidates_by_column.items():
                top_candidates_for_freq: List[Dict[str, Any]] = candidates[:top_n_for_type_freq]

                # Collect distinct type IDs from those top candidates
                row_qids: set = set()
                for cand in top_candidates_for_freq:
                    for t_dict in cand.get("types", []):
                        qid: Any = t_dict.get("id")
                        if qid:
                            row_qids.add(qid)

                # Increase counts for each distinct type in this row
                for qid in row_qids:
                    type_freq_by_column[col_index][qid] += 1

                # Mark that this row had that column (for normalization)
                rows_count_by_column[col_index] += 1

        # --------------------------------------------------------------------------
        # 2) Convert raw counts to frequencies in [0..1].
        # --------------------------------------------------------------------------
        for col_index, freq_counter in type_freq_by_column.items():
            row_count: int = rows_count_by_column[col_index]
            if row_count == 0:
                continue
            # Convert each type's raw count to a ratio in [0..1]
            for qid in freq_counter:
                freq_counter[qid] = freq_counter[qid] / row_count

        # --------------------------------------------------------------------------
        # 3) Assign new features (typeFreq1..typeFreq5) for each candidate
        # --------------------------------------------------------------------------
        for doc in batch_docs:
            candidates_by_column: Dict[Any, List[Dict[str, Any]]] = doc["candidates"]
            for col_index, candidates in candidates_by_column.items():
                # If no frequency data was built for this column, default to empty
                freq_counter: Dict[Any, float] = type_freq_by_column.get(col_index, {})

                for cand in candidates:
                    # Ensure the candidate has a features dict
                    if "features" not in cand:
                        cand["features"] = {}

                    # Gather candidate's type IDs
                    cand_qids: List[Any] = [
                        t_obj.get("id") for t_obj in cand.get("types", []) if t_obj.get("id")
                    ]

                    # Get the frequency for each type
                    cand_type_freqs: List[float] = [
                        freq_counter.get(qid, 0.0) for qid in cand_qids
                    ]

                    # Sort in descending order
                    cand_type_freqs.sort(reverse=True)

                    # Assign typeFreq1..typeFreq5
                    for i in range(1, 6):
                        # If the candidate has fewer than i types, default to 0
                        freq_val: float = (
                            cand_type_freqs[i - 1] if (i - 1) < len(cand_type_freqs) else 0.0
                        )
                        cand["features"][f"typeFreq{i}"] = round(freq_val, 3)

        # --------------------------------------------------------------------------
        # 4) Build final feature matrix & do ML predictions
        # --------------------------------------------------------------------------
        all_candidates: List[List[float]] = []
        doc_info: List[Tuple[Any, int, Any, int]] = []
        for doc in batch_docs:
            row_index: int = doc["row_id"]
            candidates_by_column: Dict[Any, List[Dict[str, Any]]] = doc["candidates"]
            for col_index, candidates in candidates_by_column.items():
                features_list: List[List[float]] = [self.extract_features(c) for c in candidates]
                all_candidates.extend(features_list)

                # Track the location of each candidate
                doc_info.extend(
                    [(doc["_id"], row_index, col_index, idx) for idx in range(len(candidates))]
                )

        if len(all_candidates) == 0:
            print(
                f"No candidates to predict for dataset={dataset_name}, table={table_name}. "
                "Skipping..."
            )
            return

        candidate_features: np.ndarray = np.array(all_candidates)
        print(f"Predicting scores for {len(candidate_features)} candidates...")
        ml_scores: np.ndarray = model.predict(candidate_features, batch_size=128)[:, 1]
        print("Scores predicted.")

        # Assign new ML scores back to candidates
        for i, (doc_id, row_index, col_index, cand_idx) in enumerate(doc_info):
            candidate: Dict[str, Any] = doc_map[doc_id]["candidates"][col_index][cand_idx]
            candidate["score"] = float(ml_scores[i])

        # --------------------------------------------------------------------------
        # 5) Batch update the final results
        # --------------------------------------------------------------------------

        bulk_updates = []
        for doc_id, doc in doc_map.items():
            row_index = doc["row_id"]
            updated_candidates_by_column = {}

            for col_index, candidates in doc["candidates"].items():
                ranked_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[
                    : self.max_candidates
                ]
                updated_candidates_by_column[col_index] = ranked_candidates

            # Create update operation
            bulk_updates.append(
                UpdateOne(
                    {"dataset_name": dataset_name, "table_name": table_name, "row_id": row_index},
                    {"$set": {"el_results": updated_candidates_by_column}},
                )
            )

        # Execute all updates in a single batch operation
        if bulk_updates:
            input_collection.bulk_write(bulk_updates)

        return len(batch_docs)  # Return count of processed documents

    def extract_features(self, candidate: Dict[str, Any]) -> List[float]:
        return [candidate["features"].get(feature, 0.0) for feature in self.selected_features]
