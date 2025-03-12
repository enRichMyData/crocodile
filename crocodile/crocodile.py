import asyncio
import multiprocessing as mp
from typing import List, Optional

from crocodile.feature import Feature

# 1) Import our newly created helper classes
from crocodile.fetchers import BowFetcher, CandidateFetcher
from crocodile.ml import MLWorker
from crocodile.mongo import MongoCache, MongoConnectionManager, MongoWrapper
from crocodile.processors import RowBatchProcessor
from crocodile.trace import TraceWorker


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
        mongo_uri: Optional[str] = None,  # Allow passing the MongoDB URI
        max_workers: Optional[int] = None,
        max_candidates: int = 5,
        max_training_candidates: int = 10,
        entity_retrieval_endpoint: Optional[str] = None,
        entity_bow_endpoint: Optional[str] = None,
        entity_retrieval_token: Optional[str] = None,
        selected_features: Optional[List[str]] = None,
        candidate_retrieval_limit: int = 100,
        model_path: Optional[str] = None,
        batch_size: int = 10000,
        ml_ranking_workers: int = 2,
        top_n_for_type_freq: int = 3,
        max_bow_batch_size: int = 100,
    ) -> None:
        # Use the provided mongo_uri or fallback to the default
        self._MONGO_URI = mongo_uri or self._DEFAULT_MONGO_URI
        self.max_workers = max_workers or mp.cpu_count()
        self.max_candidates = max_candidates
        self.max_training_candidates = max_training_candidates
        self.entity_retrieval_endpoint = entity_retrieval_endpoint
        self.entity_retrieval_token = entity_retrieval_token
        self.candidate_retrieval_limit = candidate_retrieval_limit
        self.model_path = model_path
        self.entity_bow_endpoint = entity_bow_endpoint
        self.batch_size = batch_size
        self.ml_ranking_workers = ml_ranking_workers
        self.top_n_for_type_freq = top_n_for_type_freq
        self.MAX_BOW_BATCH_SIZE = max_bow_batch_size
        self.mongo_wrapper = MongoWrapper(
            self._MONGO_URI, self._DB_NAME, self._TIMING_COLLECTION, self._ERROR_LOG_COLLECTION
        )
        self.feature = Feature(selected_features)

        # Instantiate our helper objects
        self._candidate_fetcher = CandidateFetcher(self)
        self._bow_fetcher = BowFetcher(self)
        self._row_processor = RowBatchProcessor(self)

    def get_db(self):
        """Get MongoDB database connection for current process"""
        client = MongoConnectionManager.get_client(self._MONGO_URI)
        return client[self._DB_NAME]

    def __del__(self):
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

    def claim_todo_batch(self, input_collection, batch_size=10):
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

    def run(self):
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
                db_uri=self._MONGO_URI,
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
            self._MONGO_URI,
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

        self.__del__()
        print("All tasks have been processed.")