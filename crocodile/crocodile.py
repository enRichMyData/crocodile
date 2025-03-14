import asyncio
import multiprocessing as mp
import os
import uuid
from pathlib import Path
from typing import List, Optional

import pandas as pd
from column_classifier import ColumnClassifier

from crocodile.feature import Feature
from crocodile.fetchers import BowFetcher, CandidateFetcher
from crocodile.ml import MLWorker
from crocodile.mongo import MongoCache, MongoConnectionManager, MongoWrapper
from crocodile.processors import RowBatchProcessor
from crocodile.trace import TraceWorker
from crocodile.typing import ColType


class Crocodile:
    """
    Crocodile entity linking system with hidden MongoDB configuration.
    """

    _DEFAULT_MONGO_URI = "mongodb://mongodb:27017/"  # Change this to a class-level default
    _DB_NAME = "crocodile_db"
    _TABLE_TRACE_COLLECTION = "table_trace"
    _DATASET_TRACE_COLLECTION = "dataset_trace"
    _INPUT_COLLECTION = "input_data"
    _TRAINING_COLLECTION = "training_data"
    _ERROR_LOG_COLLECTION = "error_logs"
    _TIMING_COLLECTION = "timing_trace"
    _CACHE_COLLECTION = "candidate_cache"
    _BOW_CACHE_COLLECTION = "bow_cache"

    def __init__(
        self,
        input_csv: str | Path,
        output_csv: str | Path | None = None,
        dataset_name: str = None,
        table_name: str = None,
        columns_type: ColType | None = None,
        max_workers: Optional[int] = None,
        max_candidates: int = 5,
        max_training_candidates: int = 10,
        entity_retrieval_endpoint: Optional[str] = None,
        entity_retrieval_token: Optional[str] = None,
        selected_features: Optional[List[str]] = None,
        candidate_retrieval_limit: int = 16,
        model_path: Optional[str] = None,
        batch_size: int = 1024,
        ml_ranking_workers: int = 2,
        top_n_for_type_freq: int = 3,
        **kwargs,
    ) -> None:
        self.input_csv = input_csv
        self.output_csv = output_csv
        if self.output_csv is None:
            self.output_csv = os.path.splitext(input_csv)[0] + "_output.csv"
        if dataset_name is None:
            dataset_name = uuid.uuid4().hex
        if table_name is None:
            table_name = os.path.basename(self.input_csv).split(".")[0]
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.columns_type = columns_type
        # Use the provided mongo_uri or fallback to the default
        self.max_workers = max_workers or mp.cpu_count()
        self.max_candidates = max_candidates
        self.max_training_candidates = max_training_candidates
        self.entity_retrieval_endpoint = entity_retrieval_endpoint
        self.entity_retrieval_token = entity_retrieval_token
        self.candidate_retrieval_limit = candidate_retrieval_limit
        self.model_path = model_path
        self.batch_size = batch_size
        self.ml_ranking_workers = ml_ranking_workers
        self.top_n_for_type_freq = top_n_for_type_freq
        self._max_bow_batch_size = kwargs.pop("max_bow_batch_size", 128)
        self._entity_bow_endpoint = kwargs.pop("entity_bow_endpoint", None)
        self._mongo_uri = kwargs.pop("mongo_uri", None) or self._DEFAULT_mongo_uri
        self._save_output_to_csv = kwargs.pop("save_output_to_csv", True)
        self.mongo_wrapper = MongoWrapper(
            self._mongo_uri, self._DB_NAME, self._TIMING_COLLECTION, self._ERROR_LOG_COLLECTION
        )
        self.feature = Feature(selected_features)

        # Instantiate our helper objects
        self._candidate_fetcher = CandidateFetcher(self)
        self._bow_fetcher = BowFetcher(self)
        self._row_processor = RowBatchProcessor(self)

        # Create indexes
        self.mongo_wrapper.create_indexes()

    def get_db(self):
        """Get MongoDB database connection for current process"""
        client = MongoConnectionManager.get_client(self._mongo_uri)
        return client[self._DB_NAME]

    def close_mongo_connection(self):
        """Cleanup when instance is destroyed"""
        try:
            MongoConnectionManager.close_connection()
        except Exception:
            pass

    def get_candidate_cache(self):
        db = self.get_db()
        return MongoCache(db, self._CACHE_COLLECTION)

    def get_bow_cache(self):
        db = self.get_db()
        return MongoCache(db, self._BOW_CACHE_COLLECTION)

    def fetch_candidates_batch(self, entities, row_texts, fuzzies, qids):
        return asyncio.run(
            self._candidate_fetcher.fetch_candidates_batch_async(
                entities, row_texts, fuzzies, qids
            )
        )

    def fetch_bow_vectors_batch(self, row_hash, row_text, qids):
        async def runner():
            return await self._bow_fetcher.fetch_bow_vectors_batch_async(row_hash, row_text, qids)

        return asyncio.run(runner())

    def process_rows_batch(self, docs, dataset_name, table_name):
        self._row_processor.process_rows_batch(docs, dataset_name, table_name)

    def claim_todo_batch(self, input_collection, batch_size=16):
        docs = []
        for _ in range(batch_size):
            doc = input_collection.find_one_and_update(
                {"status": "TODO"}, {"$set": {"status": "DOING"}}
            )
            if doc is None:
                break
            docs.append(doc)
        return docs

    def worker(self):
        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]

        while True:
            todo_docs = self.claim_todo_batch(input_collection)
            if not todo_docs:
                print("No more tasks to process.")
                break

            tasks_by_table = {}
            for doc in todo_docs:
                dataset_name = doc["dataset_name"]
                table_name = doc["table_name"]
                tasks_by_table.setdefault((dataset_name, table_name), []).append(doc)

            for (dataset_name, table_name), docs in tasks_by_table.items():
                self.process_rows_batch(docs, dataset_name, table_name)

    def onboard_data(
        self,
        dataset_name: str = None,
        table_name: str = None,
        columns_type: ColType | None = None,
    ):
        df = pd.read_csv(self.input_csv)

        if columns_type is None:
            classifier = ColumnClassifier(model_type="fast")
            classification_results = classifier.classify_multiple_tables([df])
            table_classification = classification_results[0].get("table_1", {})

            # We'll create two dictionaries: one for NE (Named Entity) columns and
            # one for LIT (Literal) columns.
            ne_cols = {}
            lit_cols = {}
            ignored_cols = []

            # Define which classification types should be considered as NE.
            NE_classifications = {"PERSON", "OTHER", "ORGANIZATION", "LOCATION"}

            # Iterate over the DataFrame columns by order so that
            # we use the column's index (as a string)
            for idx, col in enumerate(df.columns):
                # Get the classification result for this column.
                # If the classifier didn't return a result for a column, we default to "UNKNOWN".
                col_result = table_classification.get(col, {})
                classification = col_result.get("classification", "UNKNOWN")

                if classification in NE_classifications:
                    ne_cols[str(idx)] = classification
                else:
                    lit_cols[str(idx)] = classification
        else:
            ne_cols = columns_type.get("NE", {})
            lit_cols = columns_type.get("LIT", {})
            ignored_cols = columns_type.get("IGNORED", [])
        all_recognized_cols = set(ne_cols.keys()) | set(lit_cols.keys())
        all_cols = set([str(i) for i in range(len(df.columns))])
        if len(all_recognized_cols) != len(all_cols):
            ignored_cols.extend(list(all_cols - all_recognized_cols))
        ignored_cols = list(set(ignored_cols))
        context_cols = list(set([str(i) for i in range(len(df.columns))]) - set(ignored_cols))

        db = self.get_db()
        input_collection = db["input_data"]
        table_trace_collection = db["table_trace"]
        dataset_trace_collection = db["dataset_trace"]

        table_trace_collection.insert_one(
            {
                "dataset_name": dataset_name,
                "table_name": table_name,
                "header": list(df.columns),  # Store the header (column names)
                "total_rows": len(df),
                "processed_rows": 0,
                "status": "PENDING",
                "classified_columns": {"NE": ne_cols, "LIT": lit_cols, "IGNORED": ignored_cols},
            }
        )

        # Onboard data (values only, no headers) along with the classification metadata
        for index, row in df.iterrows():
            document = {
                "dataset_name": dataset_name,
                "table_name": table_name,
                "row_id": index,
                "data": row.tolist(),  # Store row values as a list
                "classified_columns": {"NE": ne_cols, "LIT": lit_cols, "IGNORED": ignored_cols},
                "context_columns": context_cols,  # Context columns (by index)
                "correct_qids": {},  # Empty as ground truth is not available
                "status": "TODO",
            }
            input_collection.insert_one(document)

        # Initialize dataset-level trace (if not done earlier)
        dataset_trace_collection.update_one(
            {"dataset_name": dataset_name},
            {
                "$setOnInsert": {
                    "total_tables": 1,
                    "processed_tables": 0,
                    "total_rows": len(df),
                    "processed_rows": 0,
                    "status": "PENDING",
                }
            },
            upsert=True,
        )

        print(
            f"Data onboarded successfully for dataset '{dataset_name}' and table '{table_name}'."
        )

    def fetch_results(self):
        """Retrieves processed documents from MongoDB including `el_results`.

        Extracts the first candidate per NE column (if available).
        Uses a **streaming approach** to avoid memory overload on large tables.
        """
        db = self.get_db()
        input_collection = db["input_data"]
        cursor = input_collection.find(
            {"dataset_name": self.dataset_name, "table_name": self.table_name}
        )

        extracted_rows = []
        table_trace = input_collection.database["table_trace"].find_one(
            {"dataset_name": self.dataset_name, "table_name": self.table_name}
        )
        header = table_trace.get("header")

        for doc in cursor:
            row_data = dict(zip(header, doc["data"]))  # Original row data as dict
            el_results = doc.get("el_results", {})

            # Extract first candidate for each NE column
            for col_idx, col_type in doc["classified_columns"].get("NE", {}).items():
                try:
                    col_index = int(col_idx)
                    col_header = header[col_index]
                except (ValueError, IndexError):
                    col_header = f"col_{col_idx}"

                id_field = f"{col_header}_id"
                name_field = f"{col_header}_name"
                desc_field = f"{col_header}_desc"
                score_field = f"{col_header}_score"

                # Extract first candidate from el_results (if available)
                candidate = el_results.get(col_idx, [{}])[0]

                row_data[id_field] = candidate.get("id", "")
                row_data[name_field] = candidate.get("name", "")
                row_data[desc_field] = candidate.get("description", "")
                row_data[score_field] = candidate.get("score", "")

            extracted_rows.append(row_data)

        return extracted_rows

    def run(self):
        self.onboard_data(self.dataset_name, self.table_name, columns_type=self.columns_type)

        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]

        total_rows = self.mongo_wrapper.count_documents(input_collection, {"status": "TODO"})
        if total_rows == 0:
            print("No more tasks to process.")
        else:
            print(f"Found {total_rows} tasks to process.")

        processes = []
        for _ in range(self.max_workers):
            p = mp.Process(target=self.worker)
            p.start()
            processes.append(p)

        for _ in range(self.ml_ranking_workers):
            p = MLWorker(
                db_uri=self._mongo_uri,
                db_name=self._DB_NAME,
                table_trace_collection_name=self._TABLE_TRACE_COLLECTION,
                training_collection_name=self._TRAINING_COLLECTION,
                timing_collection_name=self._TIMING_COLLECTION,
                error_log_collection_name=self._ERROR_LOG_COLLECTION,
                input_collection=self._INPUT_COLLECTION,
                model_path=self.model_path,
                batch_size=self.batch_size,
                max_candidates=self.max_candidates,
                top_n_for_type_freq=self.top_n_for_type_freq,
                features=self.feature.selected_features,
            )
            p.start()
            processes.append(p)

        trace_work = TraceWorker(
            self._mongo_uri,
            self._DB_NAME,
            self._INPUT_COLLECTION,
            self._DATASET_TRACE_COLLECTION,
            self._TABLE_TRACE_COLLECTION,
            self._TIMING_COLLECTION,
        )
        trace_work.start()
        processes.append(trace_work)

        for p in processes:
            p.join()

        self.close_mongo_connection()
        print("All tasks have been processed.")

        if self._save_output_to_csv:
            extracted_rows = self.fetch_results()
            pd.DataFrame(extracted_rows).to_csv(self.output_csv, index=False)
            print(f"Results saved to '{self.output_csv}'.")
