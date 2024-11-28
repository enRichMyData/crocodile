import requests
import time
from pymongo import MongoClient
import multiprocessing as mp
import traceback
from datetime import datetime
import base64
import gzip
from nltk.tokenize import word_tokenize
import nltk
from collections import OrderedDict
from multiprocessing import Lock
import numpy as np
import warnings
import absl.logging
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

# Suppress specific Keras warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Do not pass an `input_shape`/`input_dim` argument to a layer.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Compiled the loaded model, but the compiled metrics have yet to be built.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Error in loading the saved optimizer state.*")

# Set Abseil and TensorFlow logging levels to suppress specific warnings
tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)

import pickle

# Download NLTK resources if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Global stopwords to avoid reinitializing repeatedly
stop_words = set(stopwords.words('english'))

# Define a shared LRU cache for compressed data
class LRUCacheCompressed:
    """In-memory LRU cache with compression and eviction policy, shared across processes."""
    
    def __init__(self, max_size):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.cache_lock = Lock()  # Use an instance-specific lock

    def get(self, key):
        with self.cache_lock:
            if key in self.cache:
                # Move accessed item to the end (most recently used)
                self.cache.move_to_end(key)
                # Decompress data before returning
                compressed_data = self.cache[key]
                decompressed_data = pickle.loads(gzip.decompress(base64.b64decode(compressed_data)))
                return decompressed_data
            return None

    def put(self, key, value):
        with self.cache_lock:
            # Compress the data before storing it
            compressed_data = base64.b64encode(gzip.compress(pickle.dumps(value))).decode('utf-8')
            if key in self.cache:
                # Update existing item and mark as most recently used
                self.cache.move_to_end(key)
            self.cache[key] = compressed_data
            # Evict the oldest item if the cache exceeds the max size
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

# Define compressed LRU caches with descriptive names
CACHE_MAX_SIZE = 100_000  # Set maximum size for each cache

# Initialize compressed caches
candidate_cache = LRUCacheCompressed(CACHE_MAX_SIZE)  # For candidate entities
bow_cache = LRUCacheCompressed(CACHE_MAX_SIZE)  # For Bag-of-Words vectors
 
# Dictionary to store client instances per process
_process_clients = {}

def get_mongo_client(uri="mongodb://localhost:27017/"):
    """
    Get or create a MongoDB client specific to the current process.
    """
    pid = mp.current_process().pid  # Identify the current process
    if pid not in _process_clients:
        # Create a new client for the current process
        _process_clients[pid] = MongoClient(uri, maxPoolSize=10)  # Adjust pool size as needed
    return _process_clients[pid]

class Crocodile:
    def __init__(self, mongo_uri="mongodb://mongodb:27017/", db_name="crocodile_db", 
                 table_trace_collection_name="table_trace", dataset_trace_collection_name="dataset_trace", 
                 collection_name="input_data", training_collection_name="training_data", 
                 error_log_collection_name="error_logs",
                 timing_collection_name="timing_trace",  
                 max_workers=None, max_candidates=5, max_training_candidates=10, 
                 entity_retrieval_endpoint=None, entity_bow_endpoint=None, entity_retrieval_token=None, 
                 selected_features=None, candidate_retrieval_limit=100,
                 model_path=None):
        
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.table_trace_collection_name = table_trace_collection_name
        self.dataset_trace_collection_name = dataset_trace_collection_name
        self.training_collection_name = training_collection_name
        self.error_log_collection_name = error_log_collection_name 
        self.timing_collection_name = timing_collection_name  
        self.max_workers = max_workers or mp.cpu_count() 
        self.max_candidates = max_candidates
        self.max_training_candidates = max_training_candidates
        self.entity_retrieval_endpoint = entity_retrieval_endpoint
        self.entity_bow_endpoint = entity_bow_endpoint
        self.entity_retrieval_token = entity_retrieval_token
        self.candidate_retrieval_limit = candidate_retrieval_limit
        self.selected_features = selected_features or [
            "ntoken_mention", "ntoken_entity", "length_mention", "length_entity",
            "popularity", "ed_score", "jaccard_score", "jaccardNgram_score", "desc", "descNgram", 
            "bow_similarity", "kind", "NERtype", "column_NERtype"
        ]

        # ML Model-related parameters
        self.model_path = model_path
        self.current_dataset = None
        self.current_table = None
    
    def time_mongo_operation(self, operation_name, query_function, *args, **kwargs):
        """
        Time a MongoDB operation and log its duration in the timing trace collection.

        Parameters:
            operation_name (str): The name of the operation being performed.
            query_function (callable): The MongoDB query function to execute.
            *args: Positional arguments for the query function.
            **kwargs: Keyword arguments for the query function.

        Returns:
            The result of the query function.
        """
        start_time = time.time()
        db = self.get_db()
        timing_trace_collection = db[self.timing_collection_name]

        try:
            result = query_function(*args, **kwargs)  # Execute the MongoDB operation
        except Exception as e:
            # Capture and log exception details
            end_time = time.time()  # Even in failure, log the time
            duration = end_time - start_time
            timing_trace_collection.insert_one({
                "operation_name": operation_name,
                "start_time": datetime.fromtimestamp(start_time),
                "end_time": datetime.fromtimestamp(end_time),
                "duration_seconds": duration,
                "args": str(args),  # Log arguments for debugging
                "kwargs": str(kwargs),
                "error": str(e),
                "status": "FAILED",
            })
            # Re-raise the exception after logging
            raise
        else:
            # Log successful operation timing
            end_time = time.time()
            duration = end_time - start_time
            timing_trace_collection.insert_one({
                "operation_name": operation_name,
                "start_time": datetime.fromtimestamp(start_time),
                "end_time": datetime.fromtimestamp(end_time),
                "duration_seconds": duration,
                "args": str(args),  # Optional: Log arguments for debugging
                "kwargs": str(kwargs),
                "status": "SUCCESS",
            })
            return result

    def update_document(self, collection, query, update, upsert=False):
        """
        Wrapper for MongoDB update_one query with timing logging.
        """
        operation_name = f"update_document:{collection.name}"
        return self.time_mongo_operation(
            operation_name,
            collection.update_one,
            query,
            update,
            upsert=upsert
        )

    def update_documents(self, collection, query, update, upsert=False):
        """
        Wrapper for MongoDB update_many query with timing logging.
        """
        operation_name = f"update_documents:{collection.name}"
        return self.time_mongo_operation(
            operation_name,
            collection.update_many,
            query,
            update,
            upsert=upsert
        )

    def find_documents(self, collection, query, projection=None, limit=None):
        """
        Wrapper for MongoDB find query with timing logging.
        """
        operation_name = f"find_documents:{collection.name}"

        # Inner function to handle the actual query
        def query_function(query, projection=None):
            cursor = collection.find(query, projection)
            if limit is not None:  # Apply limit if specified
                cursor = cursor.limit(limit)
            return list(cursor)

        # Use the time_mongo_operation wrapper
        return self.time_mongo_operation(operation_name, query_function, query, projection)


    def count_documents(self, collection, query):
        """
        Wrapper for MongoDB count_documents query with timing logging.
        """
        operation_name = f"count_documents:{collection.name}"
        return self.time_mongo_operation(
            operation_name,
            collection.count_documents,
            query
        )

    def find_one_document(self, collection, query, projection=None):
        """
        Wrapper for MongoDB find_one query with timing logging.
        """
        operation_name = f"find_one_document:{collection.name}"
        return self.time_mongo_operation(
            operation_name,
            collection.find_one,
            query,
            projection=projection
        )

    def find_one_and_update(self, collection, query, update, return_document=False):
        """
        Wrapper for MongoDB find_one_and_update query with timing logging.
        """
        operation_name = f"find_one_and_update:{collection.name}"
        return self.time_mongo_operation(
            operation_name,
            collection.find_one_and_update,
            query,
            update,
            return_document=return_document
        )

    def insert_one_document(self, collection, document):
        """
        Wrapper for MongoDB insert_one query with timing logging.
        """
        operation_name = f"insert_one_document:{collection.name}"
        return self.time_mongo_operation(
            operation_name,
            collection.insert_one,
            document
        )

    def insert_many_documents(self, collection, documents):
        """
        Wrapper for MongoDB insert_many query with timing logging.
        """
        operation_name = f"insert_many_documents:{collection.name}"
        return self.time_mongo_operation(
            operation_name,
            collection.insert_many,
            documents
        )

    def delete_documents(self, collection, query):
        """
        Wrapper for MongoDB delete_many query with timing logging.
        """
        operation_name = f"delete_documents:{collection.name}"
        return self.time_mongo_operation(
            operation_name,
            collection.delete_many,
            query
        )
    
    def get_db(self):
        """
        Get the database instance using the process-local MongoClient.
        """
        client = get_mongo_client(self.mongo_uri)
        return client[self.db_name]
    
    def set_context(self, dataset_name, table_name):
        """Set the current dataset and table for the context."""
        self.current_dataset = dataset_name
        self.current_table = table_name
    
    def log_time(self, operation_name, dataset_name, table_name, start_time, end_time, details=None):
        """
        Log the timing for an operation with optional additional details.

        Parameters:
            operation_name (str): The name of the operation being logged.
            dataset_name (str): Name of the dataset being processed.
            table_name (str): Name of the table being processed.
            start_time (float): Start time of the operation.
            end_time (float): End time of the operation.
            details (dict, optional): Additional details to log, such as payload or query information.
        """
        db = self.get_db()
        self.timing_collection = db[self.timing_collection_name]
        duration = end_time - start_time

        log_entry = {
            "operation_name": operation_name,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "start_time": datetime.fromtimestamp(start_time),
            "end_time": datetime.fromtimestamp(end_time),
            "duration_seconds": duration,
        }

        # Add additional details if provided
        if details:
            log_entry["details"] = details

        self.timing_collection.insert_one(log_entry)

    def log_to_db(self, level, message, trace=None):
        """
        Log a message to MongoDB with the specified level (e.g., INFO, WARNING, ERROR), optional traceback,
        and associated dataset/table information.

        Parameters:
            level (str): Log level (e.g., INFO, WARNING, ERROR).
            message (str): Log message.
            trace (str, optional): Optional traceback information.
            dataset_name (str, optional): The name of the dataset associated with the log.
            table_name (str, optional): The name of the table associated with the log.
        """
        db = self.get_db()
        log_collection = db[self.error_log_collection_name]

        log_entry = {
            "dataset_name": self.current_dataset,
            "table_name": self.current_table,
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "traceback": trace
        }

        log_collection.insert_one(log_entry)

    def update_table_trace(self, dataset_name, table_name, increment=0, status=None, start_time=None, end_time=None, row_time=None, ml_ranking_status=None):
        """Update the processing trace for a given table."""
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        table_trace_collection = db[self.table_trace_collection_name]

        trace = table_trace_collection.find_one({"dataset_name": dataset_name, "table_name": table_name})

        if not trace:
            return

        total_rows = trace.get("total_rows", 0)
        processed_rows = trace.get("processed_rows", 0) + increment
        update_fields = {}

        if increment:
            update_fields["processed_rows"] = increment

        if status:
            update_fields["status"] = status

        if ml_ranking_status:
            update_fields["ml_ranking_status"] = ml_ranking_status  # Add ML ranking status

        start_time = trace.get("start_time") or start_time
        if start_time:
            update_fields["start_time"] = start_time
        else:
            start_time = trace.get("start_time")

        if end_time:
            update_fields["end_time"] = end_time
            update_fields["duration"] = (end_time - start_time).total_seconds()

        if row_time:
            update_fields["last_row_time"] = row_time

        if processed_rows >= total_rows and total_rows > 0:
            update_fields["estimated_seconds"] = 0
            update_fields["estimated_hours"] = 0
            update_fields["estimated_days"] = 0
            update_fields["percentage_complete"] = 100.0
        elif start_time and total_rows > 0 and processed_rows > 0:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            avg_time_per_row = elapsed_time / processed_rows
            remaining_rows = total_rows - processed_rows
            estimated_time_left = remaining_rows * avg_time_per_row

            estimated_seconds = estimated_time_left
            estimated_hours = estimated_seconds / 3600
            estimated_days = estimated_hours / 24
            percentage_complete = (processed_rows / total_rows) * 100

            update_fields["estimated_seconds"] = round(estimated_seconds, 2)
            update_fields["estimated_hours"] = round(estimated_hours, 2)
            update_fields["estimated_days"] = round(estimated_days, 2)
            update_fields["percentage_complete"] = round(percentage_complete, 2)

        update_query = {}
        if update_fields:
            if "processed_rows" in update_fields:
                update_query["$inc"] = {"processed_rows": update_fields.pop("processed_rows")}
            update_query["$set"] = update_fields

        table_trace_collection.update_one(
            {"dataset_name": dataset_name, "table_name": table_name},
            update_query,
            upsert=True
        )

    def update_dataset_trace(self, dataset_name, start_time=None):
        """Update the dataset-level trace based on the progress of tables."""
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        table_trace_collection = db[self.table_trace_collection_name]
        dataset_trace_collection = db[self.dataset_trace_collection_name]

        # Fetch the dataset trace document
        dataset_trace = dataset_trace_collection.find_one({"dataset_name": dataset_name})

        # Use the provided start_time only if it's the first call (no existing trace or no recorded start_time)
        if dataset_trace:
            if "start_time" in dataset_trace:
                start_time = datetime.fromisoformat(dataset_trace["start_time"])  # Retrieve start_time from the document
            elif start_time is not None:
                # Save the initial start_time
                dataset_trace_collection.update_one(
                    {"dataset_name": dataset_name},
                    {"$set": {"start_time": start_time.isoformat()}},
                    upsert=True
                )

        # Fetch all tables for the dataset
        tables = list(table_trace_collection.find({"dataset_name": dataset_name}))

        total_tables = len(tables)
        processed_tables = sum(1 for table in tables if table.get("status") == "COMPLETED")

        total_rows = sum(table.get("total_rows", 0) for table in tables)
        processed_rows = sum(table.get("processed_rows", 0) for table in tables)

        # Check if dataset processing is completed
        status = "IN_PROGRESS"
        if processed_tables == total_tables:
            status = "COMPLETED"

        # Calculate duration from table-level traces
        duration_from_tables = sum(table.get("duration", 0) for table in tables if table.get("duration", 0) > 0)

        # Calculate duration from start_time if available
        if start_time:
            duration_from_start = (datetime.now() - start_time).total_seconds()
        else:
            duration_from_start = None

        if processed_rows > 0:
            avg_time_per_row = duration_from_tables / processed_rows
            remaining_rows = total_rows - processed_rows
            estimated_time_left = remaining_rows * avg_time_per_row

            estimated_seconds = round(estimated_time_left, 2)
            estimated_hours = round(estimated_seconds / 3600, 2)
            estimated_days = round(estimated_hours / 24, 2)
        else:
            estimated_seconds = estimated_hours = estimated_days = 0

        percentage_complete = round((processed_rows / total_rows) * 100, 2) if total_rows > 0 else 0

        update_fields = {
            "total_tables": total_tables,
            "processed_tables": processed_tables,
            "total_rows": total_rows,
            "processed_rows": processed_rows,
            "estimated_seconds": estimated_seconds,
            "estimated_hours": estimated_hours,
            "estimated_days": estimated_days,
            "percentage_complete": percentage_complete,
            "status": status,
            "duration_from_tables": round(duration_from_tables, 2)
        }

        # Add duration from start if available
        if duration_from_start is not None:
            update_fields["duration_from_start"] = round(duration_from_start, 2)

        dataset_trace_collection.update_one(
            {"dataset_name": dataset_name},
            {"$set": update_fields},
            upsert=True
        )

    def fetch_candidates(self, entity_name, row_text, fuzzy=False, qid=None):
        """
        Fetch candidates for a given entity with retry and backoff logic.

        Parameters:
            entity_name (str): The name of the entity to fetch candidates for.
            row_text (str): Contextual row text for fetching candidates.
            fuzzy (bool): Whether to perform fuzzy matching.
            qid (str, optional): Specific QID to filter candidates.

        Returns:
            list: List of processed candidates or an empty list on failure.
        """
        start_time = time.time()  # Start timing
        cache_key = f"{entity_name}_{fuzzy}"  # Cache key
        cached_result = candidate_cache.get(cache_key)  # Check if result is cached
        if cached_result is not None:
            end_time = time.time()  # End timing
            self.log_time("Fetch Candidates (from Cache)", self.current_dataset, self.current_table, start_time, end_time)
            return cached_result

        # Define URL for the API request
        url = f"{self.entity_retrieval_endpoint}?name={entity_name}&limit={self.candidate_retrieval_limit}&fuzzy={fuzzy}&token={self.entity_retrieval_token}"
        if qid:
            url += f"&ids={qid}"

        attempts = 0  # Track the number of attempts
        backoff = 1  # Initial backoff in seconds

        while attempts < 5:  # Retry up to 5 times
            attempts += 1
            try:
                start_time = time.time()  # Start timing
                response = requests.get(url, headers={'accept': 'application/json'}, timeout=4)
                response.raise_for_status()  # Ensure the request was successful

                # Process candidates and add to compressed cache
                candidates = response.json()
                row_tokens = set(self.tokenize_text(row_text))
                filtered_candidates = self._process_candidates(candidates, entity_name, row_tokens)

                # Cache the processed candidates
                candidate_cache.put(cache_key, filtered_candidates)

                # Log success timing
                end_time = time.time()
                self.log_time(
                    "Fetch Candidates (from API)",
                    self.current_dataset,
                    self.current_table,
                    start_time,
                    end_time,
                    details={"attempts": attempts, "entity_name": entity_name}
                )
                return filtered_candidates

            except requests.exceptions.RequestException as e:
                self.log_to_db(
                    "WARNING",
                    f"Fetch Candidates attempt {attempts} failed for '{entity_name}': {str(e)}"
                )
            except Exception as e:
                self.log_to_db(
                    "ERROR",
                    f"Unexpected error on Fetch Candidates attempt {attempts} for '{entity_name}': {traceback.format_exc()}"
                )

            # Exponential backoff with a cap
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)  # Cap the backoff at 16 seconds

        # Log final failure after retries
        self.log_to_db(
            "ERROR",
            f"Fetch Candidates failed after {attempts} attempts for '{entity_name}'"
        )
        return []

    def _process_candidates(self, candidates, entity_name, row_tokens):
        """
        Process retrieved candidates by adding features and formatting.
        
        Parameters:
            candidates (list): List of raw candidate entities.
            entity_name (str): The name of the entity.
            row_tokens (set): Tokenized words from row text.

        Returns:
            list: List of processed candidates with calculated features.
        """
        processed_candidates = []
        for candidate in candidates:
            candidate_name = candidate.get('name', '')
            candidate_description = candidate.get('description', '') or ""
            kind_numeric = self.map_kind_to_numeric(candidate.get('kind', 'entity'))
            nertype_numeric = self.map_nertype_to_numeric(candidate.get('NERtype', 'OTHERS'))
        
            features = {
                'ntoken_mention': round(candidate.get('ntoken_mention', len(entity_name.split())), 4),
                'ntoken_entity': round(candidate.get('ntoken_entity', len(candidate_name.split())), 4),
                'length_mention': round(candidate.get('length_mention', len(entity_name)), 4),
                'length_entity': round(candidate.get('length_entity', len(candidate_name)), 4),
                'popularity': round(candidate.get('popularity', 0.0), 4),
                'ed_score': round(candidate.get('ed_score', 0.0), 4),
                'jaccard_score': round(candidate.get('jaccard_score', 0.0), 4),
                'jaccardNgram_score': round(candidate.get('jaccardNgram_score', 0.0), 4),
                'desc': round(self.calculate_token_overlap(row_tokens, set(self.tokenize_text(candidate_description))), 4),
                'descNgram': round(self.calculate_ngram_similarity(entity_name, candidate_description), 4),
                'bow_similarity': 0.0,
                'kind': kind_numeric,
                'NERtype': nertype_numeric,
                'column_NERtype': None  # Placeholder for column NER type
            }

            processed_candidate = {
                'id': candidate.get('id'),
                'name': candidate_name,
                'description': candidate_description,
                'types': candidate.get('types'),
                'features': features
            }
            processed_candidates.append(processed_candidate)
        return processed_candidates

    def map_kind_to_numeric(self, kind):
        """Map kind to a numeric value."""
        mapping = {
            'entity': 1,
            'type': 2,
            'disambiguation': 3,
            'predicate': 4
        }
        return mapping.get(kind, 1)

    def map_nertype_to_numeric(self, nertype):
        """Map NERtype to a numeric value."""
        mapping = {
            'LOCATION': 1,
            'LOC': 1,
            'ORGANIZATION': 2,
            'ORG': 2,
            'PERSON': 3,
            'PERS': 3,
            'OTHER': 4,
            'OTHERS': 4
        }
        return mapping.get(nertype, 4)

    def calculate_token_overlap(self, tokens_a, tokens_b):
        """Calculate the proportion of common tokens between two token sets."""
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0

    def calculate_ngram_similarity(self, a, b, n=3):
        """Calculate N-gram similarity between two strings."""
        a_ngrams = self.ngrams(a, n)
        b_ngrams = self.ngrams(b, n)
        intersection = len(set(a_ngrams) & set(b_ngrams))
        union = len(set(a_ngrams) | set(b_ngrams))
        return intersection / union if union > 0 else 0

    def ngrams(self, string, n=3):
        """Generate n-grams from a string."""
        tokens = [string[i:i+n] for i in range(len(string)-n+1)]
        return tokens

    def score_candidate(self, candidate):
        """Score a candidate entity based on its features."""
        ed_score = candidate['features'].get('ed_score', 0.0)
        desc_score = candidate['features'].get('desc', 0.0)
        desc_ngram_score = candidate['features'].get('descNgram', 0.0)
        bow_similarity = candidate['features'].get('bow_similarity', 0.0)

        # Incorporate BoW similarity into the total score
        total_score = (ed_score + desc_score + desc_ngram_score + bow_similarity) / 4
       
        candidate['score'] = round(total_score, 2)

        return candidate

    def rank_with_feature_scoring(self, candidates):
        """Rank candidates using a feature-based scoring method."""
        scored_candidates = []
        for candidate in candidates:
            scored_candidate = self.score_candidate(candidate)
            scored_candidates.append(scored_candidate)
        
        # Sort by feature-based score in descending order
        return sorted(scored_candidates, key=lambda x: x['score'], reverse=True)
        
    def log_retry_attempt(self, retry_state):
        """
        Logs retry attempt details to MongoDB.
        """
        attempt_number = retry_state.attempt_number
        operation_name = retry_state.fn.__name__
        last_exception = str(retry_state.outcome.exception()) if retry_state.outcome else "Unknown error"

        self.log_collection.insert_one({
            "operation_name": operation_name,
            "attempt": attempt_number,
            "timestamp": datetime.now(),
            "error_message": last_exception
        })

    def get_bow_from_api(self, row_text, qids):
        """
        Fetch BoW vectors for specific QIDs and a text using caching.

        Parameters:
            row_text (str): The text for which to compute BoW vectors.
            qids (list): List of QIDs to query.

        Returns:
            dict: BoW vectors with QID as key or an empty dictionary on failure.
        """
        if not self.entity_bow_endpoint:
            raise ValueError("BoW API endpoint must be provided.")

        # Start timing for cache check
        cache_start_time = time.time()

        # Prepare cache keys for each QID based on text and QID
        cache_hits = {}
        qids_to_fetch = []
        for qid in qids:
            cache_key = f"{hash(row_text)}_{qid}"
            cached_result = bow_cache.get(cache_key)
            if cached_result is not None:
                cache_hits[qid] = cached_result
            else:
                qids_to_fetch.append(qid)

        # Measure time if all QIDs are cached
        if not qids_to_fetch:
            cache_end_time = time.time()
            self.log_time(
                "BoW Retrieval (from Cache)",
                self.current_dataset,
                self.current_table,
                cache_start_time,
                cache_end_time,
                details={"qids": qids, "row_text": row_text[:50]}  # Truncate text for logging
            )
            return cache_hits

        # Prepare payload for API request
        url = f"{self.entity_bow_endpoint}?token={self.entity_retrieval_token}"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        payload = {"json":{"text": row_text, "qids": qids_to_fetch}}

        # Fetch missing BoW vectors
        attempts = 0
        backoff = 1
        while attempts < 5:  # Retry up to 5 times
            attempts += 1
            try:
                # Start timing for API call
                api_start_time = time.time()

                response = requests.post(url, headers=headers, json=payload, timeout=4)
                response.raise_for_status()  # Ensure the request was successful

                # Parse response and update cache
                result = response.json()
                for qid, bow_vector in result.items():
                    cache_key = f"{hash(row_text)}_{qid}"
                    bow_cache.put(cache_key, bow_vector)
                    cache_hits[qid] = bow_vector

                # Log success timing
                api_end_time = time.time()
                self.log_time(
                    "BoW Fetch (from API)",
                    self.current_dataset,
                    self.current_table,
                    api_start_time,
                    api_end_time,
                    details={"attempts": attempts, "qids_fetched": qids_to_fetch, "row_text": row_text[:50]}
                )
                return cache_hits

            except requests.exceptions.RequestException as e:
                self.log_to_db(
                    "WARNING",
                    f"BoW Fetch attempt {attempts} failed for QIDs {qids_to_fetch}: {str(e)}"
                )
            except Exception as e:
                self.log_to_db(
                    "ERROR",
                    f"Unexpected error on BoW Fetch attempt {attempts} for QIDs {qids_to_fetch}: {traceback.format_exc()}"
                )

            # Exponential backoff with a cap
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)

        # Log final failure after retries
        self.log_to_db(
            "ERROR",
            f"BoW Fetch failed after {attempts} attempts for QIDs: {qids_to_fetch}"
        )
        return cache_hits

    def tokenize_text(self, text):
        """Tokenize and clean the text."""
        tokens = word_tokenize(text.lower())
        return set(t for t in tokens if t not in stop_words)

    def compute_bow_similarity(self, row_text, candidate_vectors):
        """New BoW similarity computation using Jaccard similarity."""
        if candidate_vectors is None:
            self.log_to_db("ERROR", "No candidate vectors available to compute BoW similarity.")
            return {}

        row_tokens = self.tokenize_text(row_text)

        similarity_scores = {}
        matched_words = {}
        for qid, candidate_bow in candidate_vectors.items():
            candidate_tokens = set(candidate_bow.keys())
            intersection = row_tokens.intersection(candidate_tokens)
            union = row_tokens.union(candidate_tokens)
            if union:
                similarity = len(intersection) / len(row_tokens)
            else:
                similarity = 0
            similarity_scores[qid] = similarity
            matched_words[qid] = list(intersection)

        return similarity_scores, matched_words

    def link_entity(self, row, ne_columns, context_columns, correct_qids, dataset_name, table_name, row_index):
        linked_entities = {}
        training_candidates_by_ne_column = {}
        # Build the row_text using context columns only (converting to integers since context_columns are strings)
        row_text = ' '.join([str(row[int(col_index)]) for col_index in context_columns if int(col_index) < len(row)])

        for col_index, ner_type in ne_columns.items():
            col_index = str(col_index)  # Column index as a string
            if int(col_index) < len(row):  # Avoid out-of-range access
                ne_value = row[int(col_index)]
                if ne_value and pd.notna(ne_value):
                    ne_value = str(ne_value)  # Convert to string for consistency
                    correct_qid = correct_qids.get(f"{row_index}-{col_index}", None)  # Access the correct QID using (row_index, col_index)

                    candidates = self.fetch_candidates(ne_value, row_text, qid=correct_qid)
                    if len(candidates) == 1:
                        candidates = self.fetch_candidates(ne_value, row_text, fuzzy=True, qid=correct_qid)

                    # Fetch BoW vectors for the candidates
                    candidate_qids = [candidate['id'] for candidate in candidates]
                    candidate_bows = self.get_bow_from_api(row_text, candidate_qids)
                    # Map NER type from input to numeric (extended names)
                    ner_type_numeric = self.map_nertype_to_numeric(ner_type)
                    
                    # Add BoW similarity to each candidate's features
                    for candidate in candidates:
                        qid = candidate['id']
                        bow_data = candidate_bows.get(qid, {})
                        candidate['matched_words'] = bow_data.get('matched_words', [])
                        candidate['features']['bow_similarity'] = bow_data.get('similarity_score', 0.0)
                        candidate['features']['column_NERtype'] = ner_type_numeric  # Add the NER type from the column

                    # Rank candidates based on the selected method (ML or traditional)
                    ranked_candidates = self.rank_with_feature_scoring(candidates)
                   
                    if correct_qid and correct_qid not in [c['id'] for c in ranked_candidates[:self.max_training_candidates]]:
                        correct_candidate = next((c for c in ranked_candidates if c['id'] == correct_qid), None)
                        if correct_candidate:
                            ranked_candidates = ranked_candidates[:self.max_training_candidates - 1] + [correct_candidate]

                    el_results_candidates = ranked_candidates[:self.max_candidates]
                    linked_entities[col_index] = el_results_candidates

                    training_candidates_by_ne_column[col_index] = ranked_candidates[:self.max_training_candidates]

        self.save_candidates_for_training(training_candidates_by_ne_column, dataset_name, table_name, row_index)

        return linked_entities

    def save_candidates_for_training(self, candidates_by_ne_column, dataset_name, table_name, row_index):
        """Save all candidates for a row into the training data."""
        db = self.get_db()
        training_collection = db[self.training_collection_name]

        training_document = {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_index,
            "candidates": candidates_by_ne_column,
            "ml_ranked": False
        }

        training_collection.insert_one(training_document)

    def process_row(self, doc_id, dataset_name, table_name):
        """Process a single row from the input data."""
        row_start_time = datetime.now()
        try:
            db = self.get_db()
            collection = db[self.collection_name]

            # doc = collection.find_one_and_update(
            #     {"_id": doc_id, "status": "TODO"},
            #     {"$set": {"status": "DOING"}},
            #     return_document=True
            # )
            doc = self.find_one_and_update(collection, {"_id": doc_id, "status": "TODO"}, {"$set": {"status": "DOING"}}, return_document=True)

            if not doc:
                return

            row = doc['data']
            ne_columns = doc['classified_columns']['NE']
            context_columns = doc.get('context_columns', [])
            correct_qids = doc.get('correct_qids', {})

            row_index = doc.get("row_id", None)

            linked_entities = self.link_entity(row, ne_columns, context_columns, correct_qids, dataset_name, table_name, row_index)

            collection.update_one({'_id': doc['_id']}, {'$set': {'el_results': linked_entities, 'status': 'DONE'}})

            row_end_time = datetime.now()
            row_duration = (row_end_time - row_start_time).total_seconds()
            self.update_table_trace(dataset_name, table_name, increment=1, row_time=row_duration)

            # Save row duration to trace average speed
            self.log_processing_speed(dataset_name, table_name)
        except Exception as e:
            self.log_to_db("ERROR", f"Error processing row with _id {doc_id}", traceback.format_exc())
            collection.update_one({'_id': doc_id}, {'$set': {'status': 'TODO'}})

    def log_processing_speed(self, dataset_name, table_name):
        """Log the processing speed for a table."""
        db = self.get_db()
        table_trace_collection = db[self.table_trace_collection_name]

        # Retrieve trace document for the table
        trace = table_trace_collection.find_one({"dataset_name": dataset_name, "table_name": table_name})
        if not trace:
            return

        processed_rows = trace.get("processed_rows", 1)
        start_time = trace.get("start_time")
        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Calculate and update average speed
        rows_per_second = processed_rows / elapsed_time if elapsed_time > 0 else 0
        table_trace_collection.update_one(
            {"dataset_name": dataset_name, "table_name": table_name},
            {"$set": {"rows_per_second": rows_per_second}}
        )

    def run(self):
        db = self.get_db()
        collection = db[self.collection_name]

        total_rows = self.count_documents(collection, {"status": "TODO"})
        if total_rows == 0:
            print("No more tasks to process.")
            return

        print(f"Found {total_rows} tasks to process.")

        with tqdm(total=total_rows, desc="Processing tasks", unit="rows") as pbar:
            with mp.Pool(processes=self.max_workers) as pool:
                while True:
                    todo_docs = list(self.find_documents(collection, {"status": "TODO"}, {"_id": 1, "dataset_name": 1, "table_name": 1}, limit=self.max_workers))
                    if not todo_docs:
                        print("No more tasks to process.")
                        break

                    tasks_by_table = {}
                    for doc in todo_docs:
                        dataset_table_key = (doc["dataset_name"], doc["table_name"])
                        tasks_by_table.setdefault(dataset_table_key, []).append(doc["_id"])

                    for (dataset_name, table_name), doc_ids in tasks_by_table.items():
                        self.set_context(dataset_name, table_name)

                        start_time = datetime.now()
                        self.update_dataset_trace(dataset_name, start_time=start_time)
                        self.update_table_trace(dataset_name, table_name, status="IN_PROGRESS", start_time=start_time)

                        tasks = [(doc_id, dataset_name, table_name) for doc_id in doc_ids]
                        pool.starmap(self.process_row, tasks)

                        pbar.update(len(doc_ids))

                        processed_count = self.count_documents(collection, {
                            "dataset_name": dataset_name,
                            "table_name": table_name,
                            "status": "DONE"
                        })
                        total_count = self.count_documents(collection, {
                            "dataset_name": dataset_name,
                            "table_name": table_name
                        })

                        if processed_count == total_count:
                            end_time = datetime.now()
                            self.update_table_trace(dataset_name, table_name, status="COMPLETED", end_time=end_time, start_time=start_time)
                            self.apply_ml_ranking(dataset_name, table_name)

                        self.update_dataset_trace(dataset_name)

        print("All tasks have been processed.")


    def apply_ml_ranking(self, dataset_name, table_name):
        """Perform ML-based ranking on candidates and update their scores."""
        # Load the ML model once for this task
        model = self.load_ml_model()
        db = self.get_db()
        training_collection = db[self.training_collection_name]
        input_collection = db[self.collection_name]
        table_trace_collection = db[self.table_trace_collection_name]

        batch_size = 1000
        processed_count = 0
        # total_count = training_collection.count_documents(
        #     {"dataset_name": dataset_name, "table_name": table_name, "ml_ranked": False}
        # )
        total_count = self.count_documents(training_collection, {"dataset_name": dataset_name, "table_name": table_name, "ml_ranked": False})
        print(f"Total unprocessed documents: {total_count}")

        while processed_count < total_count:
            # Retrieve 1000 unprocessed documents at a time from training_data
            batch_docs = list(training_collection.find(
                {"dataset_name": dataset_name, "table_name": table_name, "ml_ranked": False},
                limit=batch_size
            ))

            if not batch_docs:
                break  # Exit if there are no more unprocessed documents

            # Create a dictionary for direct access by document _id
            doc_map = {doc["_id"]: doc for doc in batch_docs}

            # Prepare features for the ML model
            all_candidates = []
            doc_info = []

            for doc in batch_docs:
                row_index = doc["row_id"]
                candidates_by_column = doc["candidates"]

                for col_index, candidates in candidates_by_column.items():
                    features = [self.extract_features(candidate) for candidate in candidates]
                    all_candidates.extend(features)
                    doc_info.extend([(doc["_id"], row_index, col_index, idx) for idx in range(len(candidates))])
            
            if len(all_candidates) == 0:
                print(f"No candidates to predict for dataset {dataset_name}, table {table_name}. Skipping...")
                return  # Safely exit this batch if no candidates are found

            # Predict scores for all candidates in the batch
            candidate_features = np.array(all_candidates)
            print(f"Predicting scores for {len(candidate_features)} candidates...")
            ml_scores = model.predict(candidate_features, batch_size=128)[:, 1]  # Assuming index 1 is the score for the positive class
            print("Scores predicted.")

            # Map predicted scores back to candidates using doc_map
            for i, (doc_id, row_index, col_index, candidate_idx) in enumerate(doc_info):
                candidate = doc_map[doc_id]["candidates"][col_index][candidate_idx]
                candidate["score"] = float(ml_scores[i])

            # Update ranked candidates in training_data and input_data
            for doc_id, doc in doc_map.items():
                row_index = doc["row_id"]
                updated_candidates_by_column = {}

                for col_index, candidates in doc["candidates"].items():
                    # Sort candidates by score in descending order
                    ranked_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
                    updated_candidates_by_column[col_index] = ranked_candidates[:self.max_candidates]

                # Update training_data with ranked candidates and mark as ML-ranked
                training_collection.update_one(
                    {"_id": doc_id},
                    {"$set": {"candidates": doc["candidates"], "ml_ranked": True}}
                )

                # Update input_data for entity linking results
                input_collection.update_one(
                    {"dataset_name": dataset_name, "table_name": table_name, "row_id": row_index},
                    {"$set": {"el_results": updated_candidates_by_column}}
                )

            # Update progress
            processed_count += len(batch_docs)
            progress = (processed_count / total_count) * 100

            # Log progress in table_trace_collection
            self.update_table_trace(
                dataset_name, table_name,
                ml_ranking_status=f"{progress:.2f}%"
            )
            print(f"ML ranking progress: {progress:.2f}% completed")

        # Finalize ML ranking status
        self.update_table_trace(
            dataset_name, table_name,
            ml_ranking_status="ML_RANKING_COMPLETED"
        )
        print("ML ranking completed.")

    def extract_features(self, candidate):
        """
        Extract only numerical features from a candidate's feature dictionary,
        excluding any target label.

        Parameters:
            candidate (dict): A dictionary containing candidate information with features.

        Returns:
            list: A list of extracted numerical feature values.
        """
        numerical_features = [
            'ntoken_mention', 'length_mention', 'ntoken_entity', 'length_entity',
            'popularity', 'ed_score', 'desc', 'descNgram', 'bow_similarity',
            'kind', 'NERtype', 'column_NERtype'
        ]
        return [candidate['features'].get(feature, 0.0) for feature in numerical_features]

    def load_ml_model(self):
        from tensorflow.keras.models import load_model
        return load_model(self.model_path)