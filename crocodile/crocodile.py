import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

import pandas as pd
from column_classifier import ColumnClassifier

from crocodile.feature import Feature
from crocodile.fetchers import BowFetcher, CandidateFetcher
from crocodile.ml import MLWorker
from crocodile.processors import RowBatchProcessor
from crocodile.typing import ColType


class Crocodile:
    """
    Crocodile entity linking system with hidden MongoDB configuration.
    """

    _DEFAULT_MONGO_URI = "mongodb://mongodb:27017/"  # Change this to a class-level default
    _DB_NAME = "crocodile_db"
    _INPUT_COLLECTION = "input_data"
    _TRAINING_COLLECTION = "training_data"
    _ERROR_LOG_COLLECTION = "error_logs"
    _CACHE_COLLECTION = "candidate_cache"
    _BOW_CACHE_COLLECTION = "bow_cache"

    def __init__(
        self,
        input_csv: str | Path | pd.DataFrame,
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
        from crocodile.mongo import MongoWrapper

        self.input_csv = input_csv
        self.output_csv = output_csv
        if self.output_csv is None and kwargs.get("save_output_to_csv", True):
            if isinstance(self.input_csv, pd.DataFrame):
                raise ValueError(
                    "An output name must be specified is the input is a `pd.Dataframe`"
                )
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
        self._mongo_uri = kwargs.pop("mongo_uri", None) or self._DEFAULT_MONGO_URI
        self._save_output_to_csv = kwargs.pop("save_output_to_csv", True)
        self.mongo_wrapper = MongoWrapper(
            self._mongo_uri, self._DB_NAME, self._ERROR_LOG_COLLECTION
        )
        self.feature = Feature(selected_features)

        # Instantiate our helper objects
        self._candidate_fetcher = CandidateFetcher(
            self.entity_retrieval_endpoint,
            self.entity_retrieval_token,
            self.candidate_retrieval_limit,
            self.feature,
            db_name=self._DB_NAME,
            mongo_uri=self._mongo_uri,
            input_collection=self._INPUT_COLLECTION,
            cache_collection=self._CACHE_COLLECTION,
        )
        self._bow_fetcher = BowFetcher(
            self._entity_bow_endpoint,
            self.entity_retrieval_token,
            self._max_bow_batch_size,
            self.feature,
            db_name=self._DB_NAME,
            mongo_uri=self._mongo_uri,
            input_collection=self._INPUT_COLLECTION,
            bow_cache_collection=self._BOW_CACHE_COLLECTION,
        )
        self._row_processor = RowBatchProcessor(
            self._candidate_fetcher,
            self.max_training_candidates,
            self.max_candidates,
            self._bow_fetcher if self._entity_bow_endpoint else None,
            db_name=self._DB_NAME,
            mongo_uri=self._mongo_uri,
            input_collection=self._INPUT_COLLECTION,
            training_collection=self._TRAINING_COLLECTION,
        )

        # Create indexes
        self.mongo_wrapper.create_indexes()

    def get_db(self):
        """Get MongoDB database connection for current process"""
        from crocodile.mongo import MongoConnectionManager

        client = MongoConnectionManager.get_client(self._mongo_uri)
        return client[self._DB_NAME]

    def close_mongo_connection(self):
        """Cleanup when instance is destroyed"""
        from crocodile.mongo import MongoConnectionManager

        try:
            MongoConnectionManager.close_connection()
        except Exception:
            pass

    def process_rows_batch(self, docs, dataset_name, table_name):
        self._row_processor.process_rows_batch(docs, dataset_name, table_name)

    def claim_todo_batch(self, input_collection, batch_size=16):
        docs = []
        for _ in range(batch_size):
            doc = input_collection.find_one_and_update(
                {
                    "dataset_name": self.dataset_name,
                    "table_name": self.table_name,
                    "status": "TODO",
                },
                {"$set": {"status": "DOING"}},
            )
            if doc is None:
                break
            docs.append(doc)
        return docs

    def onboard_data(
        self,
        dataset_name: str = None,
        table_name: str = None,
        columns_type: ColType | None = None,
    ):
        """Efficiently load data into MongoDB using batched inserts."""
        start_time = time.perf_counter()

        # Get database connection
        db = self.get_db()
        input_collection = db["input_data"]

        # Step 1: Determine data source and extract sample for classification
        if isinstance(self.input_csv, pd.DataFrame):
            df = self.input_csv
            sample = df
            total_rows = len(df)
            is_csv_path = False
        else:
            sample = pd.read_csv(self.input_csv, nrows=1024)
            total_rows = "unknown"
            is_csv_path = True

        print(f"Onboarding {total_rows} rows for dataset '{dataset_name}', table '{table_name}'")

        # Step 2: Perform column classification
        if columns_type is None:
            classifier = ColumnClassifier(model_type="fast")
            classification_results = classifier.classify_multiple_tables([sample])
            table_classification = classification_results[0].get("table_1", {})

            ne_cols, lit_cols, ignored_cols = {}, {}, []
            NE_classifications = {"PERSON", "OTHER", "ORGANIZATION", "LOCATION"}

            for idx, col in enumerate(sample.columns):
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
        all_cols = set([str(i) for i in range(len(sample.columns))])
        if len(all_recognized_cols) != len(all_cols):
            ignored_cols.extend(list(all_cols - all_recognized_cols))
        ignored_cols = list(set(ignored_cols))
        context_cols = list(set([str(i) for i in range(len(sample.columns))]) - set(ignored_cols))

        # Step 3: Define a chunk generator function
        def get_chunks():
            """Generator that yields chunks of rows, handling both DF and CSV."""
            if is_csv_path:
                chunk_size = 2048
                row_count = 0
                for chunk in pd.read_csv(self.input_csv, chunksize=chunk_size):
                    yield chunk, row_count
                    row_count += len(chunk)
            else:
                chunk_size = 1024 if total_rows > 100000 else 2048 if total_rows > 10000 else 4096
                total_chunks = (total_rows + chunk_size - 1) // chunk_size
                for chunk_idx in range(total_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, total_rows)
                    yield df.iloc[chunk_start:chunk_end], chunk_start

        # Step 4: Process all chunks using the generator
        processed_rows = 0
        chunk_idx = 0

        for chunk, start_idx in get_chunks():
            chunk_idx += 1
            documents = []
            for i, (_, row) in enumerate(chunk.iterrows()):
                row_id = start_idx + i
                document = {
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "row_id": row_id,
                    "data": row.tolist(),
                    "classified_columns": {
                        "NE": ne_cols,
                        "LIT": lit_cols,
                        "IGNORED": ignored_cols,
                    },
                    "context_columns": context_cols,
                    "correct_qids": {},
                    "status": "TODO",
                }
                documents.append(document)

            if documents:
                try:
                    input_collection.insert_many(documents, ordered=False)
                    chunk_size = len(documents)
                    processed_rows += chunk_size
                    elapsed = time.perf_counter() - start_time
                    rows_per_second = processed_rows / elapsed if elapsed > 0 else 0

                    if is_csv_path:
                        print(
                            f"Chunk {chunk_idx}: "
                            f"Processed {chunk_size} rows (total: {processed_rows}) "
                            f"({rows_per_second:.1f} rows/sec)"
                        )
                    else:
                        chunk_start = start_idx + 1
                        chunk_end = start_idx + chunk_size
                        total_chunks = (total_rows + chunk_size - 1) // chunk_size
                        print(
                            f"Chunk {chunk_idx}/{total_chunks}: "
                            f"Onboarded rows {chunk_start}-{chunk_end} "
                            f"({rows_per_second:.1f} rows/sec)"
                        )
                except Exception as e:
                    print(f"Error inserting batch {chunk_idx}: {str(e)}")
                    if "duplicate key" not in str(e).lower():
                        raise

        total_time = time.perf_counter() - start_time
        print(f"Data onboarding complete for dataset '{dataset_name}' and table '{table_name}'")
        print(
            f"Onboarded {processed_rows} rows in {total_time:.2f} seconds "
            f"({processed_rows/total_time:.1f} rows/sec)"
        )

    def fetch_results(self):
        """Retrieves processed documents from MongoDB using memory-efficient streaming.

        For large tables, this avoids loading all results into memory at once.
        Instead, it processes documents in batches and writes directly to CSV.
        """
        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]

        # Determine if we need to write to CSV or return full results
        stream_to_csv = self._save_output_to_csv and isinstance(self.output_csv, (str, Path))

        # Get header information
        header = None
        if isinstance(self.input_csv, pd.DataFrame):
            header = self.input_csv.columns.tolist()
        elif isinstance(self.input_csv, str):
            header = pd.read_csv(self.input_csv, nrows=0).columns.tolist()

        # Get first document to determine column count if header is still None
        sample_doc = input_collection.find_one(
            {"dataset_name": self.dataset_name, "table_name": self.table_name}
        )
        if not sample_doc:
            print("No documents found for the specified dataset and table.")
            return []

        if header is None:
            print("Could not extract header from input table, using generic column names.")
            header = [f"col_{i}" for i in range(len(sample_doc["data"]))]

        # Create extended header with entity columns
        extended_header = header.copy()
        for col_idx in sample_doc["classified_columns"].get("NE", {}).keys():
            try:
                col_index = int(col_idx)
                col_header = header[col_index]
            except (ValueError, IndexError):
                col_header = f"col_{col_idx}"

            extended_header.extend(
                [
                    f"{col_header}_id",
                    f"{col_header}_name",
                    f"{col_header}_desc",
                    f"{col_header}_score",
                ]
            )

        # Process in batches with cursor
        batch_size = 1024  # Process 1024 documents at a time

        # Only fetch fields we actually need to reduce network transfer
        projection = {"data": 1, "el_results": 1, "classified_columns.NE": 1}
        cursor = input_collection.find(
            {"dataset_name": self.dataset_name, "table_name": self.table_name},
            projection=projection,
        ).batch_size(batch_size)

        # Determine whether to stream to CSV or collect results
        if stream_to_csv:
            total_docs = input_collection.count_documents(
                {"dataset_name": self.dataset_name, "table_name": self.table_name}
            )
            print(f"Streaming {total_docs} documents to CSV...")

            # Process in chunks to maintain memory efficiency
            chunk_size = 256  # Write to CSV in chunks of 256 rows
            current_chunk = []
            processed_count = 0

            for doc in cursor:
                # Create base row data with original values
                row_data = dict(zip(header, doc["data"]))
                el_results = doc.get("el_results", {})

                # Add entity linking results
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

                    # Get first candidate or empty placeholder
                    candidate = el_results.get(col_idx, [{}])[0]

                    row_data[id_field] = candidate.get("id", "")
                    row_data[name_field] = candidate.get("name", "")
                    row_data[desc_field] = candidate.get("description", "")
                    row_data[score_field] = candidate.get("score", 0)

                # Add row to current chunk
                current_chunk.append(row_data)
                processed_count += 1

                # When chunk reaches desired size, write to CSV
                if len(current_chunk) >= chunk_size:
                    # Create DataFrame and append to CSV
                    chunk_df = pd.DataFrame(current_chunk)

                    # Use mode='a' (append) for all chunks after the first
                    mode = "w" if processed_count <= chunk_size else "a"
                    # Only include header for the first chunk
                    header_option = True if processed_count <= chunk_size else False

                    # Write chunk to CSV
                    chunk_df.to_csv(self.output_csv, index=False, mode=mode, header=header_option)

                    # Clear chunk and report progress
                    current_chunk = []
                    print(f"Processed {processed_count}/{total_docs} rows...")

            # Write any remaining rows
            if current_chunk:
                chunk_df = pd.DataFrame(current_chunk)
                # Append mode if this isn't the first (and only) chunk
                mode = "w" if processed_count == len(current_chunk) else "a"
                header_option = True if processed_count == len(current_chunk) else False

                chunk_df.to_csv(self.output_csv, index=False, mode=mode, header=header_option)

            print(f"Results saved to '{self.output_csv}'. Total rows: {processed_count}")
            return []
        else:
            # If not streaming to CSV, collect all results in memory
            all_rows = []
            processed_count = 0

            for doc in cursor:
                # Create base row data with original values
                row_data = dict(zip(header, doc["data"]))
                el_results = doc.get("el_results", {})

                # Add entity linking results (same as above)
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

                    candidate = el_results.get(col_idx, [{}])[0]

                    row_data[id_field] = candidate.get("id", "")
                    row_data[name_field] = candidate.get("name", "")
                    row_data[desc_field] = candidate.get("description", "")
                    row_data[score_field] = candidate.get("score", 0)

                all_rows.append(row_data)
                processed_count += 1

                # Report progress periodically
                if processed_count % batch_size == 0:
                    print(f"Processed {processed_count} rows...")

            print(f"Retrieved {processed_count} rows total")
            return all_rows

    def ml_worker(self, rank: int):
        """Wrapper function to create and run an MLWorker with the correct parameters"""
        worker = MLWorker(
            rank,
            table_name=self.table_name,
            dataset_name=self.dataset_name,
            training_collection_name=self._TRAINING_COLLECTION,
            error_log_collection_name=self._ERROR_LOG_COLLECTION,
            input_collection=self._INPUT_COLLECTION,
            model_path=self.model_path,
            batch_size=self.batch_size,
            max_candidates=self.max_candidates,
            top_n_for_type_freq=self.top_n_for_type_freq,
            features=self.feature.selected_features,
            mongo_uri=self._mongo_uri,
            db_name=self._DB_NAME,
        )
        return worker.run()  # Call run directly

    def worker(self, rank: int):
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

    def run(self):
        self.onboard_data(self.dataset_name, self.table_name, columns_type=self.columns_type)

        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]

        total_rows = self.mongo_wrapper.count_documents(input_collection, {"status": "TODO"})
        print(f"Found {total_rows} tasks to process.")

        with mp.Pool(processes=self.max_workers) as pool:
            pool.map(self.worker, range(self.max_workers))

        with mp.Pool(processes=self.ml_ranking_workers) as pool:
            pool.map(self.ml_worker, range(self.ml_ranking_workers))

        self.close_mongo_connection()
        print("All tasks have been processed.")

        extracted_rows = self.fetch_results()
        return extracted_rows
