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
    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017/",
        db_name: str = "crocodile_db",
        table_trace_collection_name: str = "table_trace",
        dataset_trace_collection_name: str = "dataset_trace",
        input_collection: str = "input_data",
        training_collection_name: str = "training_data",
        error_log_collection_name: str = "error_logs",
        timing_collection_name: str = "timing_trace",
        cache_collection_name: str = "candidate_cache",
        bow_cache_collection_name: str = "bow_cache",
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
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.input_collection = input_collection
        self.table_trace_collection_name = table_trace_collection_name
        self.dataset_trace_collection_name = dataset_trace_collection_name
        self.training_collection_name = training_collection_name
        self.error_log_collection_name = error_log_collection_name
        self.timing_collection_name = timing_collection_name
        self.cache_collection_name = cache_collection_name
        self.bow_cache_collection_name = bow_cache_collection_name
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
            mongo_uri, db_name, timing_collection_name, error_log_collection_name
        )
        self.feature = Feature(selected_features)

        # Instantiate our helper objects
        self._candidate_fetcher = CandidateFetcher(self)
        self._bow_fetcher = BowFetcher(self)
        self._row_processor = RowBatchProcessor(self)

    def get_db(self):
        """Get MongoDB database connection for current process"""
        client = MongoConnectionManager.get_client(self.mongo_uri)
        return client[self.db_name]

    def __del__(self):
        """Cleanup when instance is destroyed"""
        try:
            MongoConnectionManager.close_connection()
        except Exception:
            pass

    def get_candidate_cache(self):
        db = self.get_db()
        return MongoCache(db, self.cache_collection_name)

    def get_bow_cache(self):
        db = self.get_db()
        return MongoCache(db, self.bow_cache_collection_name)

    # -- Public method that calls the candidate fetcher
    def fetch_candidates_batch(self, entities, row_texts, fuzzies, qids):
        """
        Now we just run the async fetch in a synchronous manner.
        """
        return asyncio.run(
            self._candidate_fetcher.fetch_candidates_batch_async(
                entities, row_texts, fuzzies, qids
            )
        )

    # -- Public method that calls the bow fetcher
    def fetch_bow_vectors_batch(self, row_hash, row_text, qids):
        async def runner():
            return await self._bow_fetcher.fetch_bow_vectors_batch_async(row_hash, row_text, qids)

        return asyncio.run(runner())

    # -- Public method that calls our row-batch processor
    def process_rows_batch(self, docs, dataset_name, table_name):
        self._row_processor.process_rows_batch(docs, dataset_name, table_name)

    # The rest (claim_todo_batch, worker, run) remain basically the same
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
        input_collection = db[self.input_collection]

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
        input_collection = db[self.input_collection]

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
                db_uri=self.mongo_uri,
                db_name=self.db_name,
                table_trace_collection_name=self.table_trace_collection_name,
                training_collection_name=self.training_collection_name,
                timing_collection_name=self.timing_collection_name,
                error_log_collection_name=self.error_log_collection_name,
                input_collection=self.input_collection,
                model_path=self.model_path,
                batch_size=self.batch_size,
                max_candidates=self.max_candidates,
                top_n_for_type_freq=self.top_n_for_type_freq,
                features=self.feature.selected_features,
            )
            p.start()
            processes.append(p)

        trace_work = TraceWorker(
            self.mongo_uri,
            self.db_name,
            self.input_collection,
            self.dataset_trace_collection_name,
            self.table_trace_collection_name,
            self.timing_collection_name,
        )
        trace_work.start()
        processes.append(trace_work)

        for p in processes:
            p.join()

        self.__del__()
        print("All tasks have been processed.")
