import multiprocessing as mp
import os
import time
import uuid
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    _ERROR_LOG_COLLECTION = "error_logs"
    _CACHE_COLLECTION = "candidate_cache"
    _BOW_CACHE_COLLECTION = "bow_cache"

    def __init__(
        self,
        input_csv: str | Path | pd.DataFrame | None = None,
        output_csv: str | Path | None = None,
        client_id: str = None,  
        dataset_name: str = None,
        table_name: str = None,
        columns_type: ColType | None = None,
        max_workers: Optional[int] = None,
        max_candidates_in_result: int = 5,
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
        if self.output_csv is None and kwargs.get("save_output_to_csv", True) and not kwargs.get("fetch_result_mode_only", False):
            if isinstance(self.input_csv, pd.DataFrame):
                raise ValueError(
                    "An output name must be specified is the input is a `pd.Dataframe`"
                )
            self.output_csv = os.path.splitext(input_csv)[0] + "_output.csv"
        
        # Ensure we have proper defaults for all components of the triple identifier
        self.client_id = client_id or str(uuid.uuid4())
        self.dataset_name = dataset_name or uuid.uuid4().hex
        if table_name is None:
            self.table_name = os.path.basename(self.input_csv).split(".")[0] if not isinstance(self.input_csv, pd.DataFrame) else "unnamed_table"
        else:
            self.table_name = table_name
            
        self.columns_type = columns_type
        self.max_workers = max_workers or mp.cpu_count()
        self.max_candidates_in_result = max_candidates_in_result
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
        self._fetch_result_mode_only = kwargs.pop("fetch_result_mode_only", False)
        self.mongo_wrapper = MongoWrapper(
            self._mongo_uri, self._DB_NAME, self._ERROR_LOG_COLLECTION
        )
        self.feature = Feature(
            self.client_id,
            self.dataset_name,
            self.table_name,
            top_n_for_type_freq=top_n_for_type_freq,
            features=selected_features,
            db_name=self._DB_NAME,
            mongo_uri=self._mongo_uri,
            input_collection=self._INPUT_COLLECTION,
        )

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
            self.max_candidates_in_result,
            self._bow_fetcher if self._entity_bow_endpoint else None,
            db_name=self._DB_NAME,
            mongo_uri=self._mongo_uri,
            input_collection=self._INPUT_COLLECTION,
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
                    "client_id": self.client_id,  # Add client_id to the query
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

        # Use class-level values if not provided
        dataset_name = dataset_name or self.dataset_name
        table_name = table_name or self.table_name

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
                    "client_id": self.client_id,  # Add client_id to each document
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

    def export_results_to_csv(self):
        """Processes documents from MongoDB and writes directly to CSV.
        
        This method efficiently processes entity linking results by streaming
        them directly to a CSV file without loading all results into memory.
        This approach is optimized for large datasets of millions of rows.
        
        The method doesn't return any data - results are written to the output
        CSV file specified during initialization.
        """
        # Check if saving to CSV is enabled
        if not self._save_output_to_csv:
            print("CSV output generation is disabled. Skipping export.")
            return None

        # Verify that we have a valid output path for CSV
        if not isinstance(self.output_csv, (str, Path)):
            print("No valid output CSV path specified. Cannot save results.")
            return None

        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]

        # Get header information
        header = None
        if isinstance(self.input_csv, pd.DataFrame):
            header = self.input_csv.columns.tolist()
        elif isinstance(self.input_csv, str):
            header = pd.read_csv(self.input_csv, nrows=0).columns.tolist()
        num_cols = len(header)

        # Get first document to determine column count if header is still None
        sample_doc = input_collection.find_one(
            {
                "client_id": self.client_id,
                "dataset_name": self.dataset_name,
                "table_name": self.table_name
            }
        )
        if not sample_doc:
            print("No documents found for the specified dataset and table.")
            return None

        if header is None:
            print("Could not extract header from input table, using generic column names.")
            header = [f"col_{i}" for i in range(num_cols)]

        # Process in batches with cursor
        batch_size = 1024  # Process 1024 documents at a time

        # Only fetch fields we actually need to reduce network transfer
        projection = {"data": 1, "el_results": 1, "classified_columns.NE": 1}
        cursor = input_collection.find(
            {
                "client_id": self.client_id,
                "dataset_name": self.dataset_name,
                "table_name": self.table_name
            },
            projection=projection,
        ).batch_size(batch_size)

        # Count documents for progress reporting
        total_docs = input_collection.count_documents(
            {
                "client_id": self.client_id,
                "dataset_name": self.dataset_name,
                "table_name": self.table_name
            }
        )
        print(f"Streaming {total_docs} documents to CSV at {self.output_csv}...")

        # Setup for handling results
        current_chunk = []
        processed_count = 0
        chunk_size = 256  # Size for chunked CSV writing

        # Process all documents and stream to CSV
        for doc in cursor:
            # Process each document
            row_data = self._extract_row_data(doc, header)
            
            # For CSV streaming, collect into chunk
            current_chunk.append(row_data)
            processed_count += 1

            # Write chunk when it reaches desired size
            if len(current_chunk) >= chunk_size:
                self._write_csv_chunk(current_chunk, processed_count, chunk_size, total_docs)
                current_chunk = []

        # Handle any remaining rows
        if current_chunk:
            self._write_csv_chunk(current_chunk, processed_count, chunk_size, total_docs)
            
        print(f"Results saved to '{self.output_csv}'. Total rows: {processed_count}")

    def _extract_row_data(self, doc, header):
        """Extract row data from a MongoDB document.

        Encapsulates the common logic for formatting a row from a document.
        """
        # Create base row data with original values
        row_data = dict(zip(header, doc["data"]))
        el_results = doc.get("el_results", {})

        # Add entity linking results
        for col_idx, col_type in doc["classified_columns"].get("NE", {}).items():
            col_index = int(col_idx)
            col_header = header[col_index]

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

        return row_data

    def _write_csv_chunk(self, chunk, processed_count, chunk_size, total_docs):
        """Write a chunk of data to CSV.

        Encapsulates the CSV writing logic.
        """
        # Create DataFrame and append to CSV
        chunk_df = pd.DataFrame(chunk)

        # Use mode='a' (append) for all chunks after the first
        mode = "w" if processed_count <= len(chunk) else "a"
        # Only include header for the first chunk
        header_option = True if processed_count <= len(chunk) else False

        # Write chunk to CSV
        chunk_df.to_csv(self.output_csv, index=False, mode=mode, header=header_option)

        # Report progress
        print(f"Processed {processed_count}/{total_docs} rows...")

    def ml_worker(self, rank: int, global_type_counts: Dict[Any, Counter]):
        """Wrapper function to create and run an MLWorker with the correct parameters"""
       
        worker = MLWorker(
            rank,
            client_id=self.client_id,
            dataset_name=self.dataset_name,
            table_name=self.table_name,
            error_log_collection_name=self._ERROR_LOG_COLLECTION,
            input_collection=self._INPUT_COLLECTION,
            model_path=self.model_path,
            batch_size=self.batch_size,
            max_candidates_in_result=self.max_candidates_in_result,
            top_n_for_type_freq=self.top_n_for_type_freq,
            features=self.feature.selected_features,
            mongo_uri=self._mongo_uri,
            db_name=self._DB_NAME,
        )
        return worker.run(global_type_counts=global_type_counts)  # Call run directly

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

        total_rows = self.mongo_wrapper.count_documents(
            input_collection, 
            {
                "client_id": self.client_id,  # Add client_id to the query
                "dataset_name": self.dataset_name,
                "table_name": self.table_name,
                "status": "TODO"
            }
        )
        print(f"Found {total_rows} tasks to process.")

        with mp.Pool(processes=self.max_workers) as pool:
            pool.map(self.worker, range(self.max_workers))

        global_type_counts = self.feature.compute_global_type_frequencies(
            docs_to_process=0.7, 
            random_sample=True
        )

        with mp.Pool(processes=self.ml_ranking_workers) as pool:
            ml_worker = partial(self.ml_worker, global_type_counts=global_type_counts)
            pool.map(ml_worker, range(self.ml_ranking_workers))

        # self.close_mongo_connection()
        print("All tasks have been processed.")

        # Export results to CSV
        self.export_results_to_csv()
        
    def fetch_results(self, row_ids=None):
        """Fetch entity linking results by specific row_ids.
        
        This method efficiently retrieves raw entity linking results for specific row IDs.
        
        Args:
            row_ids: List of specific row IDs to retrieve or a single row_id. 
                     If None, raises an error (use export_results_to_csv for writing all results to CSV).
            
        Returns:
            A list of dictionaries containing the raw entity linking results for the specified row_ids,
            along with row_id and status fields for tracking processing state.
        
        Raises:
            ValueError: If row_ids is None or empty
        """
        if row_ids is None or (isinstance(row_ids, list) and len(row_ids) == 0):
            raise ValueError("row_ids parameter must be specified with one or more row IDs")
            
        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]
        
        # Normalize row_ids to list format
        if not isinstance(row_ids, list):
            row_ids = [row_ids]
            
        # Construct the query
        query = {
            "client_id": self.client_id,
            "dataset_name": self.dataset_name,
            "table_name": self.table_name,
            "row_id": {"$in": row_ids}
        }
        
        # Setup projection to only fetch fields we need
        projection = {
            "el_results": 1, 
            "row_id": 1,
            "status": 1,
            "ml_status": 1
        }
        
        # Query directly for the specified row_ids
        cursor = input_collection.find(query, projection=projection).sort("row_id", 1)
        
        # Process results - just grab the raw el_results and metadata
        results = []
        for doc in cursor:
            result = {
                "row_id": doc["row_id"],
                "status": doc["status"],
                "ml_status": doc["ml_status"],
                "el_results": doc.get("el_results", {})
            }
            results.append(result)
        
        # Return results directly as a list
        return results
