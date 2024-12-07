import requests
import time
import asyncio
import aiohttp
from pymongo import MongoClient
import multiprocessing as mp
import traceback
from datetime import datetime
import nltk
import warnings
import absl.logging
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

# Suppress certain Keras/TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)

# NLTK setup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

class MongoCache:
    def __init__(self, db, collection_name):
        self.collection = db[collection_name]
        self.collection.create_index('key', unique=True)

    def get(self, key):
        result = self.collection.find_one({'key': key})
        return result['value'] if result else None

    def put(self, key, value):
        self.collection.update_one({'key': key}, {'$set': {'value': value}}, upsert=True)

class Crocodile:
    def __init__(self, mongo_uri="mongodb://localhost:27017/",
                 db_name="crocodile_db",
                 table_trace_collection_name="table_trace",
                 dataset_trace_collection_name="dataset_trace",
                 collection_name="input_data",
                 training_collection_name="training_data",
                 error_log_collection_name="error_logs",
                 timing_collection_name="timing_trace",
                 cache_collection_name="candidate_cache",
                 bow_cache_collection_name="bow_cache",
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
        self.cache_collection_name = cache_collection_name
        self.bow_cache_collection_name = bow_cache_collection_name
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
        self.model_path = model_path
        self.current_dataset = None
        self.current_table = None

    def get_db(self):
        client = MongoClient(self.mongo_uri, maxPoolSize=10)
        return client[self.db_name]

    def get_candidate_cache(self):
        return MongoCache(self.get_db(), self.cache_collection_name)

    def get_bow_cache(self):
        return MongoCache(self.get_db(), self.bow_cache_collection_name)

    def set_context(self, dataset_name, table_name):
        self.current_dataset = dataset_name
        self.current_table = table_name

    def time_mongo_operation(self, operation_name, query_function, *args, **kwargs):
        start_time = time.time()
        db = self.get_db()
        timing_trace_collection = db[self.timing_collection_name]
        try:
            result = query_function(*args, **kwargs)
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            timing_trace_collection.insert_one({
                "operation_name": operation_name,
                "start_time": datetime.fromtimestamp(start_time),
                "end_time": datetime.fromtimestamp(end_time),
                "duration_seconds": duration,
                "args": str(args),
                "kwargs": str(kwargs),
                "error": str(e),
                "status": "FAILED",
            })
            raise
        else:
            end_time = time.time()
            duration = end_time - start_time
            timing_trace_collection.insert_one({
                "operation_name": operation_name,
                "start_time": datetime.fromtimestamp(start_time),
                "end_time": datetime.fromtimestamp(end_time),
                "duration_seconds": duration,
                "args": str(args),
                "kwargs": str(kwargs),
                "status": "SUCCESS",
            })
            return result

    def log_time(self, operation_name, dataset_name, table_name, start_time, end_time, details=None):
        db = self.get_db()
        timing_collection = db[self.timing_collection_name]
        duration = end_time - start_time
        log_entry = {
            "operation_name": operation_name,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "start_time": datetime.fromtimestamp(start_time),
            "end_time": datetime.fromtimestamp(end_time),
            "duration_seconds": duration,
        }
        if details:
            log_entry["details"] = details
        timing_collection.insert_one(log_entry)

    def log_to_db(self, level, message, trace=None):
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

    def find_documents(self, collection, query, projection=None, limit=None):
        operation_name = f"find_documents:{collection.name}"
        def query_function(query, projection=None):
            cursor = collection.find(query, projection)
            if limit is not None:
                cursor = cursor.limit(limit)
            return list(cursor)
        return self.time_mongo_operation(operation_name, query_function, query, projection)

    def count_documents(self, collection, query):
        operation_name = f"count_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.count_documents, query)

    def find_one_document(self, collection, query, projection=None):
        operation_name = f"find_one_document:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.find_one, query, projection=projection)

    def find_one_and_update(self, collection, query, update, return_document=False):
        operation_name = f"find_one_and_update:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.find_one_and_update, query, update, return_document=return_document)

    def insert_one_document(self, collection, document):
        operation_name = f"insert_one_document:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.insert_one, document)

    def insert_many_documents(self, collection, documents):
        operation_name = f"insert_many_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.insert_many, documents)

    def update_document(self, collection, query, update, upsert=False):
        operation_name = f"update_document:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.update_one, query, update, upsert=upsert)

    def update_documents(self, collection, query, update, upsert=False):
        operation_name = f"update_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.update_many, query, update, upsert=upsert)

    def delete_documents(self, collection, query):
        operation_name = f"delete_documents:{collection.name}"
        return self.time_mongo_operation(operation_name, collection.delete_many, query)

    def update_table_trace(self, dataset_name, table_name, increment=0, status=None, start_time=None, end_time=None, row_time=None, ml_ranking_status=None):
        # Implementation from previous answer (clamping processed_rows)
        # ...
        db = self.get_db()
        table_trace_collection = db[self.table_trace_collection_name]

        update_query = {}
        if increment:
            update_query["$inc"] = {"processed_rows": increment}

        set_fields = {}
        if status:
            set_fields["status"] = status
        if ml_ranking_status:
            set_fields["ml_ranking_status"] = ml_ranking_status

        trace = table_trace_collection.find_one({"dataset_name": dataset_name, "table_name": table_name})
        if trace and trace.get("total_rows", 0) == 0 and status is None:
            set_fields["status"] = "COMPLETED"

        if trace:
            if "start_time" not in trace and start_time:
                set_fields["start_time"] = start_time
            elif "start_time" in trace:
                start_time = trace["start_time"]
            else:
                start_time = datetime.now()
                set_fields["start_time"] = start_time

            if end_time:
                set_fields["end_time"] = end_time
                set_fields["duration"] = (end_time - start_time).total_seconds()
            if row_time:
                set_fields["last_row_time"] = row_time

        if set_fields:
            if "$set" in update_query:
                update_query["$set"].update(set_fields)
            else:
                update_query["$set"] = set_fields

        self.time_mongo_operation(f"update_table_trace:{table_name}", table_trace_collection.update_one,
                                  {"dataset_name": dataset_name, "table_name": table_name}, update_query, upsert=True)

        trace = table_trace_collection.find_one({"dataset_name": dataset_name, "table_name": table_name})
        if trace:
            start_time = trace.get("start_time", start_time)
            total_rows = trace.get("total_rows", 0)
            processed_rows = trace.get("processed_rows", 0)

            if total_rows > 0 and processed_rows > total_rows:
                processed_rows = total_rows
                table_trace_collection.update_one(
                    {"dataset_name": dataset_name, "table_name": table_name},
                    {"$set": {"processed_rows": processed_rows}}
                )

            if processed_rows >= total_rows and total_rows > 0:
                estimated_seconds = 0
                estimated_hours = 0
                estimated_days = 0
                percentage_complete = 100.0
            elif start_time and total_rows > 0 and processed_rows > 0:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                avg_time_per_row = elapsed_time / processed_rows
                remaining_rows = total_rows - processed_rows
                estimated_time_left = remaining_rows * avg_time_per_row
                estimated_seconds = estimated_time_left
                estimated_hours = estimated_seconds / 3600
                estimated_days = estimated_hours / 24
                percentage_complete = (processed_rows / total_rows) * 100
            else:
                estimated_seconds = estimated_hours = estimated_days = 0
                percentage_complete = 0

            update_fields = {
                "estimated_seconds": round(estimated_seconds, 2),
                "estimated_hours": round(estimated_hours, 2),
                "estimated_days": round(estimated_days, 2),
                "percentage_complete": round(percentage_complete, 2)
            }

            table_trace_collection.update_one(
                {"dataset_name": dataset_name, "table_name": table_name},
                {"$set": update_fields}
            )

    def update_dataset_trace(self, dataset_name, start_time=None):
        # Implementation from previous answer (clamping processed_rows)
        # ...
        db = self.get_db()
        table_trace_collection = db[self.table_trace_collection_name]
        dataset_trace_collection = db[self.dataset_trace_collection_name]

        dataset_trace = dataset_trace_collection.find_one({"dataset_name": dataset_name})
        if dataset_trace:
            if "start_time" in dataset_trace:
                start_time = dataset_trace["start_time"]
            elif start_time is not None:
                dataset_trace_collection.update_one(
                    {"dataset_name": dataset_name},
                    {"$set": {"start_time": start_time}},
                    upsert=True
                )
        else:
            if start_time is None:
                start_time = datetime.now()
            dataset_trace_collection.update_one(
                {"dataset_name": dataset_name},
                {"$set": {"start_time": start_time}},
                upsert=True
            )

        tables = list(table_trace_collection.find({"dataset_name": dataset_name}))
        for t in tables:
            if t.get("total_rows", 0) == 0 and t.get("status") != "COMPLETED":
                self.update_table_trace(dataset_name, t["table_name"], status="COMPLETED")

        tables = list(table_trace_collection.find({"dataset_name": dataset_name}))
        total_tables = len(tables)
        processed_tables = sum(1 for t in tables if t.get("status") == "COMPLETED")
        total_rows = sum(t.get("total_rows", 0) for t in tables)
        processed_rows = sum(t.get("processed_rows", 0) for t in tables)

        if total_rows > 0 and processed_rows > total_rows:
            processed_rows = total_rows

        status = "IN_PROGRESS"
        if processed_tables == total_tables and total_tables > 0:
            status = "COMPLETED"

        duration_from_tables = sum(t.get("duration", 0) for t in tables if t.get("duration", 0) > 0)

        if start_time:
            duration_from_start = (datetime.now() - start_time).total_seconds()
        else:
            duration_from_start = None

        if processed_rows > 0 and total_rows > 0:
            avg_time_per_row = duration_from_tables / processed_rows if duration_from_tables > 0 else 0
            remaining_rows = max(total_rows - processed_rows, 0)
            estimated_time_left = remaining_rows * avg_time_per_row
            estimated_seconds = round(estimated_time_left, 2)
            estimated_hours = round(estimated_seconds / 3600, 2)
            estimated_days = round(estimated_hours / 24, 2)
            percentage_complete = round((processed_rows / total_rows)*100, 2)
        else:
            estimated_seconds = estimated_hours = estimated_days = 0
            percentage_complete = 0

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

        if duration_from_start is not None:
            update_fields["duration_from_start"] = round(duration_from_start, 2)

        dataset_trace_collection.update_one(
            {"dataset_name": dataset_name},
            {"$set": update_fields},
            upsert=True
        )

    async def _fetch_candidate(self, entity_name, row_text, fuzzy, qid, session):
        # Implementation from previous answer including logging time
        # ...
        url = f"{self.entity_retrieval_endpoint}?name={entity_name}&limit={self.candidate_retrieval_limit}&fuzzy={fuzzy}&token={self.entity_retrieval_token}"
        if qid:
            url += f"&ids={qid}"
        attempts = 0
        backoff = 1

        while attempts < 5:
            attempts += 1
            start_time = time.time()
            try:
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    candidates = await response.json()
                    end_time = time.time()
                    self.log_time("FetchCandidates", self.current_dataset, self.current_table, start_time, end_time, details={"url": url})

                    row_tokens = set(self.tokenize_text(row_text))
                    filtered_candidates = self._process_candidates(candidates, entity_name, row_tokens)
                    cache = self.get_candidate_cache()
                    cache_key = f"{entity_name}_{fuzzy}"
                    cache.put(cache_key, filtered_candidates)
                    return entity_name, filtered_candidates
            except Exception as e:
                self.log_to_db(
                    "WARNING" if attempts < 5 else "ERROR",
                    f"Fetch Candidates attempt {attempts} failed for '{entity_name}': {str(e)}"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 16)
        return entity_name, []

    async def fetch_candidates_batch_async(self, entities, row_texts, fuzzies, qids):
        # Implementation from previous answer
        # ...
        results = {}
        cache = self.get_candidate_cache()
        to_fetch = []
        for entity_name, fuzzy in zip(entities, fuzzies):
            cache_key = f"{entity_name}_{fuzzy}"
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                results[entity_name] = cached_result
            else:
                to_fetch.append((entity_name, fuzzy))

        if not to_fetch:
            return results

        async with aiohttp.ClientSession() as session:
            tasks = []
            for ((entity_name, fuzzy), row_text, qid) in zip(to_fetch, row_texts, qids):
                tasks.append(self._fetch_candidate(entity_name, row_text, fuzzy, qid, session))
            done = await asyncio.gather(*tasks, return_exceptions=False)
            for entity_name, candidates in done:
                results[entity_name] = candidates

        return results

    def fetch_candidates_batch(self, entities, row_texts, fuzzies, qids):
        return asyncio.run(self.fetch_candidates_batch_async(entities, row_texts, fuzzies, qids))

    async def _fetch_bow_for_multiple_qids(self, row_hash, row_text, qids, session):
        # Implementation from previous answer including logging time
        # ...
        bow_cache = self.get_bow_cache()
        to_fetch = []
        bow_results = {}

        for qid in qids:
            if qid:
                cache_key = f"{row_hash}_{qid}"
                cached_result = bow_cache.get(cache_key)
                if cached_result is not None:
                    bow_results[qid] = cached_result
                else:
                    to_fetch.append(qid)

        if not to_fetch:
            return bow_results

        url = f"{self.entity_bow_endpoint}?token={self.entity_retrieval_token}"
        attempts = 0
        backoff = 1
        payload = {"json": {"text": row_text, "qids": to_fetch}}

        while attempts < 5:
            attempts += 1
            start_time = time.time()
            try:
                async with session.post(url, json=payload, timeout=10) as response:
                    response.raise_for_status()
                    bow_data = await response.json()
                    end_time = time.time()
                    self.log_time("FetchBoW", self.current_dataset, self.current_table, start_time, end_time, details={"url": url, "qids": to_fetch})

                    for qid in to_fetch:
                        qid_data = bow_data.get(qid, {"similarity_score": 0.0, "matched_words": []})
                        cache_key = f"{row_hash}_{qid}"
                        bow_cache.put(cache_key, qid_data)
                        bow_results[qid] = qid_data
                    return bow_results
            except Exception as e:
                self.log_to_db(
                    "WARNING" if attempts < 5 else "ERROR",
                    f"BoW Fetch attempt {attempts} failed for QIDs {to_fetch}: {str(e)}"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 16)

        return bow_results

    def fetch_bow_vectors_batch(self, row_hash, row_text, qids):
        async def runner():
            async with aiohttp.ClientSession() as session:
                return await self._fetch_bow_for_multiple_qids(row_hash, row_text, qids, session)

        return asyncio.run(runner())

    def process_rows_batch(self, docs, dataset_name, table_name):
        row_start_time = datetime.now()
        db = self.get_db()
        collection = db[self.collection_name]

        entities_to_process = []
        row_texts = []
        fuzzies = []
        qids = []
        row_indices = []
        col_indices = []
        ner_types = []
        row_data_list = []

        try:
            for doc in docs:
                row = doc['data']
                ne_columns = doc['classified_columns']['NE']
                context_columns = doc.get('context_columns', [])
                correct_qids = doc.get('correct_qids', {})
                row_index = doc.get("row_id", None)

                row_text = ' '.join([str(row[int(c)]) for c in context_columns if int(c) < len(row)])
                row_hash = str(hash(row_text))

                row_data_list.append((doc['_id'], row, ne_columns, context_columns, correct_qids, row_index, row_text, row_hash))

                for c, ner_type in ne_columns.items():
                    c = str(c)
                    if int(c) < len(row):
                        ne_value = row[int(c)]
                        if ne_value and pd.notna(ne_value):
                            ne_value = str(ne_value)
                            correct_qid = correct_qids.get(f"{row_index}-{c}", None)
                            entities_to_process.append(ne_value)
                            row_texts.append(row_text)
                            fuzzies.append(False)
                            qids.append(correct_qid)
                            row_indices.append(row_index)
                            col_indices.append(c)
                            ner_types.append(ner_type)

            candidates_results = self.fetch_candidates_batch(entities_to_process, row_texts, fuzzies, qids)

            # Fuzzy retry if needed
            entities_to_retry = []
            row_texts_retry = []
            fuzzies_retry = []
            qids_retry = []
            row_indices_retry = []
            col_indices_retry = []
            ner_types_retry = []

            for ne_value, r_i, c_i, nt in zip(entities_to_process, row_indices, col_indices, ner_types):
                candidates = candidates_results.get(ne_value, [])
                if len(candidates) == 1:
                    entities_to_retry.append(ne_value)
                    idx = entities_to_process.index(ne_value)
                    row_texts_retry.append(row_texts[idx])
                    fuzzies_retry.append(True)
                    correct_qid = qids[idx]
                    qids_retry.append(correct_qid)
                    row_indices_retry.append(r_i)
                    col_indices_retry.append(c_i)
                    ner_types_retry.append(nt)
                else:
                    candidates_results[ne_value] = candidates

            if entities_to_retry:
                retry_results = self.fetch_candidates_batch(entities_to_retry, row_texts_retry, fuzzies_retry, qids_retry)
                for ne_value in entities_to_retry:
                    candidates_results[ne_value] = retry_results.get(ne_value, [])

            all_candidate_qids = []
            for ne_value, candidates in candidates_results.items():
                for c in candidates:
                    if c['id']:
                        all_candidate_qids.append(c['id'])
            all_candidate_qids = list(set([q for q in all_candidate_qids if q]))

            if row_data_list:
                sample_row = row_data_list[0]
                sample_row_text = sample_row[6]
                sample_row_hash = sample_row[7]
            else:
                sample_row_text = ""
                sample_row_hash = "no_text"

            if all_candidate_qids and self.entity_bow_endpoint and self.entity_retrieval_token and sample_row_hash is not None:
                bow_data = self.fetch_bow_vectors_batch(sample_row_hash, sample_row_text, all_candidate_qids)
            else:
                bow_data = {}

            for ne_value, candidates in candidates_results.items():
                for c in candidates:
                    qid = c['id']
                    if qid in bow_data:
                        c['features']['bow_similarity'] = bow_data[qid].get('similarity_score', 0.0)
                    else:
                        c['features']['bow_similarity'] = 0.0

            # Process each doc, set status=DONE, save training data
            for doc_id, row, ne_columns, context_columns, correct_qids, row_index, row_text, row_hash in row_data_list:
                linked_entities = {}
                training_candidates_by_ne_column = {}

                for c, ner_type in ne_columns.items():
                    c = str(c)
                    if int(c) < len(row):
                        ne_value = row[int(c)]
                        if ne_value and pd.notna(ne_value):
                            ne_value = str(ne_value)
                            correct_qid = correct_qids.get(f"{row_index}-{c}", None)
                            candidates = candidates_results.get(ne_value, [])

                            ner_type_numeric = self.map_nertype_to_numeric(ner_type)
                            for candidate in candidates:
                                candidate['features']['column_NERtype'] = ner_type_numeric

                            ranked_candidates = self.rank_with_feature_scoring(candidates)
                            if correct_qid and correct_qid not in [can['id'] for can in ranked_candidates[:self.max_training_candidates]]:
                                correct_candidate = next((x for x in ranked_candidates if x['id'] == correct_qid), None)
                                if correct_candidate:
                                    ranked_candidates = ranked_candidates[:self.max_training_candidates - 1] + [correct_candidate]

                            el_results_candidates = ranked_candidates[:self.max_candidates]
                            linked_entities[c] = el_results_candidates
                            training_candidates_by_ne_column[c] = ranked_candidates[:self.max_training_candidates]

                # Save training data
                self.save_candidates_for_training(training_candidates_by_ne_column, dataset_name, table_name, row_index)
                db[self.collection_name].update_one({'_id': doc_id}, {'$set': {'el_results': linked_entities, 'status': 'DONE'}})

            # Increment processed_rows by number of docs processed
            self.update_table_trace(dataset_name, table_name, increment=len(row_data_list))
            self.log_processing_speed(dataset_name, table_name)

        except Exception as e:
            self.log_to_db("ERROR", f"Error processing batch of rows", traceback.format_exc())

    def map_kind_to_numeric(self, kind):
        mapping = {
            'entity': 1,
            'type': 2,
            'disambiguation': 3,
            'predicate': 4
        }
        return mapping.get(kind, 1)

    def map_nertype_to_numeric(self, nertype):
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
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0

    def calculate_ngram_similarity(self, a, b, n=3):
        a_ngrams = self.ngrams(a, n)
        b_ngrams = self.ngrams(b, n)
        intersection = len(set(a_ngrams) & set(b_ngrams))
        union = len(set(a_ngrams) | set(b_ngrams))
        return intersection / union if union > 0 else 0

    def ngrams(self, string, n=3):
        tokens = [string[i:i+n] for i in range(len(string)-n+1)]
        return tokens

    def tokenize_text(self, text):
        tokens = word_tokenize(text.lower())
        return set(t for t in tokens if t not in stop_words)

    def _process_candidates(self, candidates, entity_name, row_tokens):
        processed_candidates = []
        for candidate in candidates:
            candidate_name = candidate.get('name', '')
            candidate_description = candidate.get('description', '') or ""
            kind_numeric = self.map_kind_to_numeric(candidate.get('kind', 'entity'))
            nertype_numeric = self.map_nertype_to_numeric(candidate.get('NERtype', 'OTHERS'))

            features = {
                'ntoken_mention': round(candidate.get('ntoken_mention', len(entity_name.split())), 4),
                'ntoken_entity': round(candidate.get('ntoken_entity', len(candidate_name.split())), 4),
                'length_mention': round(len(entity_name), 4),
                'length_entity': round(len(candidate_name), 4),
                'popularity': round(candidate.get('popularity', 0.0), 4),
                'ed_score': round(candidate.get('ed_score', 0.0), 4),
                'jaccard_score': round(candidate.get('jaccard_score', 0.0), 4),
                'jaccardNgram_score': round(candidate.get('jaccardNgram_score', 0.0), 4),
                'desc': round(self.calculate_token_overlap(row_tokens, set(self.tokenize_text(candidate_description))), 4),
                'descNgram': round(self.calculate_ngram_similarity(entity_name, candidate_description), 4),
                'bow_similarity': 0.0,
                'kind': kind_numeric,
                'NERtype': nertype_numeric,
                'column_NERtype': None
            }

            processed_candidates.append({
                'id': candidate.get('id'),
                'name': candidate_name,
                'description': candidate_description,
                'types': candidate.get('types'),
                'features': features
            })
        return processed_candidates

    def score_candidate(self, candidate):
        ed_score = candidate['features'].get('ed_score', 0.0)
        desc_score = candidate['features'].get('desc', 0.0)
        desc_ngram_score = candidate['features'].get('descNgram', 0.0)
        feature_sum = ed_score + desc_score + desc_ngram_score
        total_score = (feature_sum / 3) if feature_sum > 0 else 0.0
        candidate['score'] = round(total_score, 2)
        return candidate

    def rank_with_feature_scoring(self, candidates):
        scored_candidates = [self.score_candidate(c) for c in candidates]
        return sorted(scored_candidates, key=lambda x: x['score'], reverse=True)

    def save_candidates_for_training(self, candidates_by_ne_column, dataset_name, table_name, row_index):
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

    def log_processing_speed(self, dataset_name, table_name):
        db = self.get_db()
        table_trace_collection = db[self.table_trace_collection_name]

        trace = table_trace_collection.find_one({"dataset_name": dataset_name, "table_name": table_name})
        if not trace:
            return
        processed_rows = trace.get("processed_rows", 1)
        start_time = trace.get("start_time")
        elapsed_time = (datetime.now() - start_time).total_seconds() if start_time else 0
        rows_per_second = processed_rows / elapsed_time if elapsed_time > 0 else 0
        table_trace_collection.update_one(
            {"dataset_name": dataset_name, "table_name": table_name},
            {"$set": {"rows_per_second": rows_per_second}}
        )

    def worker(self):
        model = self.load_ml_model()
        db = self.get_db()
        collection = db[self.collection_name]
        table_trace_collection = db[self.table_trace_collection_name]

        while True:
            todo_docs = self.find_documents(collection, {"status": "TODO"}, {"_id": 1, "dataset_name":1, "table_name":1}, limit=10)
            if not todo_docs:
                print("No more tasks to process.")
                break

            tasks_by_table = {}
            for doc in todo_docs:
                dataset_name = doc["dataset_name"]
                table_name = doc["table_name"]
                tasks_by_table.setdefault((dataset_name, table_name), []).append(doc["_id"])

            for (dataset_name, table_name), doc_ids in tasks_by_table.items():
                self.set_context(dataset_name, table_name)
                total_count = self.count_documents(collection, {"dataset_name": dataset_name, "table_name": table_name})
                t_trace = table_trace_collection.find_one({"dataset_name": dataset_name, "table_name": table_name})
                if t_trace is None or "total_rows" not in t_trace or t_trace["total_rows"] != total_count:
                    table_trace_collection.update_one(
                        {"dataset_name": dataset_name, "table_name": table_name},
                        {"$set": {"total_rows": total_count}},
                        upsert=True
                    )

                start_time = datetime.now()
                self.update_dataset_trace(dataset_name, start_time=start_time)
                self.update_table_trace(dataset_name, table_name, status="IN_PROGRESS", start_time=start_time)

                self.update_documents(collection, {"_id": {"$in": doc_ids}, "status":"TODO"}, {"$set":{"status":"DOING"}})
                docs = self.find_documents(collection, {"_id": {"$in": doc_ids}, "status":"DOING"})
                self.process_rows_batch(docs, dataset_name, table_name)

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
                    self.apply_ml_ranking(dataset_name, table_name, model)

                self.update_dataset_trace(dataset_name)

    def run(self):
        mp.set_start_method("spawn", force=True)

        db = self.get_db()
        collection = db[self.collection_name]

        total_rows = self.count_documents(collection, {"status": "TODO"})
        if total_rows == 0:
            print("No more tasks to process.")
            return

        print(f"Found {total_rows} tasks to process.")

        processes = []
        for _ in range(self.max_workers):
            p = mp.Process(target=self.worker)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("All tasks have been processed.")

    def apply_ml_ranking(self, dataset_name, table_name, model):
        db = self.get_db()
        training_collection = db[self.training_collection_name]
        input_collection = db[self.collection_name]

        batch_size = 1000
        processed_count = 0
        total_count = self.count_documents(training_collection, {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "ml_ranked": False
        })
        print(f"Total unprocessed documents: {total_count}")

        while processed_count < total_count:
            batch_docs = list(training_collection.find(
                {"dataset_name": dataset_name, "table_name": table_name, "ml_ranked": False},
                limit=batch_size
            ))

            if not batch_docs:
                break

            doc_map = {doc["_id"]: doc for doc in batch_docs}

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
                return

            import numpy as np
            candidate_features = np.array(all_candidates)
            print(f"Predicting scores for {len(candidate_features)} candidates...")
            ml_scores = model.predict(candidate_features, batch_size=128)[:, 1]
            print("Scores predicted.")

            for i, (doc_id, row_index, col_index, candidate_idx) in enumerate(doc_info):
                candidate = doc_map[doc_id]["candidates"][col_index][candidate_idx]
                candidate["score"] = float(ml_scores[i])

            for doc_id, doc in doc_map.items():
                row_index = doc["row_id"]
                updated_candidates_by_column = {}
                for col_index, candidates in doc["candidates"].items():
                    ranked_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
                    updated_candidates_by_column[col_index] = ranked_candidates[:self.max_candidates]

                training_collection.update_one(
                    {"_id": doc_id},
                    {"$set": {"candidates": doc["candidates"], "ml_ranked": True}}
                )

                input_collection.update_one(
                    {"dataset_name": dataset_name, "table_name": table_name, "row_id": row_index},
                    {"$set": {"el_results": updated_candidates_by_column}}
                )

            processed_count += len(batch_docs)
            progress = (processed_count / total_count) * 100
            progress = min(progress, 100.0)
            print(f"ML ranking progress: {progress:.2f}% completed")

            self.update_table_trace(dataset_name, table_name, ml_ranking_status=f"{progress:.2f}%")

        self.update_table_trace(dataset_name, table_name, ml_ranking_status="ML_RANKING_COMPLETED")
        print("ML ranking completed.")

    def extract_features(self, candidate):
        numerical_features = [
            'ntoken_mention', 'length_mention', 'ntoken_entity', 'length_entity',
            'popularity', 'ed_score', 'desc', 'descNgram',
            'bow_similarity', 'kind', 'NERtype', 'column_NERtype'
        ]
        return [candidate['features'].get(feature, 0.0) for feature in numerical_features]
    
    def load_ml_model(self):
        from tensorflow.keras.models import load_model
        return load_model(self.model_path)

# To run:
# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     crocodile = Crocodile(
#         mongo_uri="mongodb://localhost:27017/",
#         db_name="crocodile_db",
#         entity_retrieval_endpoint="http://example.com/api",
#         entity_retrieval_token="your_token",
#         entity_bow_endpoint="http://example.com/api_bow",
#         model_path="path_to_your_model.h5"
#     )
#     crocodile.run()