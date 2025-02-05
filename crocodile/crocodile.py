import asyncio
import hashlib
import multiprocessing as mp
import time
import traceback
from datetime import datetime
from urllib.parse import quote

import aiohttp
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from crocodile import MY_TIMEOUT
from crocodile.ml import MLWorker
from crocodile.mongo import MongoCache, MongoConnectionManager, MongoWrapper
from crocodile.trace import TraceWorker

stop_words = set(stopwords.words("english"))


class Crocodile:
    def __init__(
        self,
        mongo_uri="mongodb://localhost:27017/",
        db_name="crocodile_db",
        table_trace_collection_name="table_trace",
        dataset_trace_collection_name="dataset_trace",
        input_collection="input_data",
        training_collection_name="training_data",
        error_log_collection_name="error_logs",
        timing_collection_name="timing_trace",
        cache_collection_name="candidate_cache",
        bow_cache_collection_name="bow_cache",
        max_workers=None,
        max_candidates=5,
        max_training_candidates=10,
        entity_retrieval_endpoint=None,
        entity_bow_endpoint=None,
        entity_retrieval_token=None,
        selected_features=None,
        candidate_retrieval_limit=100,
        model_path=None,
        batch_size=10000,
        ml_ranking_workers=2,
        top_n_for_type_freq=3,
        max_bow_batch_size=100,
    ):
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
        self.selected_features = selected_features or [
            "ntoken_mention",
            "ntoken_entity",
            "length_mention",
            "length_entity",
            "popularity",
            "ed_score",
            "jaccard_score",
            "jaccardNgram_score",
            "desc",
            "descNgram",
            "bow_similarity",
            "kind",
            "NERtype",
            "column_NERtype",
            "typeFreq1",
            "typeFreq2",
            "typeFreq3",
            "typeFreq4",
            "typeFreq5",
        ]
        self.model_path = model_path
        self.entity_bow_endpoint = entity_bow_endpoint
        self.batch_size = batch_size
        self.ml_ranking_workers = ml_ranking_workers
        self.top_n_for_type_freq = top_n_for_type_freq
        self.MAX_BOW_BATCH_SIZE = max_bow_batch_size
        self.mongo_wrapper = MongoWrapper(
            mongo_uri, db_name, timing_collection_name, error_log_collection_name
        )

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

    async def _fetch_candidates(self, entity_name, row_text, fuzzy, qid, session):
        db = self.get_db()
        timing_trace_collection = db[self.timing_collection_name]

        # Encode the entity_name to handle special characters
        encoded_entity_name = quote(entity_name)
        url = (
            f"{self.entity_retrieval_endpoint}?name={encoded_entity_name}"
            f"&limit={self.candidate_retrieval_limit}&fuzzy={fuzzy}&token={self.entity_retrieval_token}"
        )

        if qid:
            url += f"&ids={qid}"
        backoff = 1

        # We'll attempt up to 5 times
        for attempts in range(5):
            start_time = time.time()
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    candidates = await response.json()
                    row_tokens = set(self.tokenize_text(row_text))
                    fetched_candidates = self._process_candidates(
                        candidates, entity_name, row_tokens
                    )

                    # Ensure all QIDs are included by adding placeholders for missing ones
                    required_qids = qid.split() if qid else []
                    existing_qids = {c["id"] for c in fetched_candidates if c.get("id")}
                    missing_qids = set(required_qids) - existing_qids

                    for missing_qid in missing_qids:
                        fetched_candidates.append(
                            {
                                "id": missing_qid,  # Placeholder for missing QID
                                "name": None,
                                "description": None,
                                "features": None,  # Explicitly set features to None
                                "is_placeholder": True,  # Explicitly mark as placeholder
                            }
                        )

                    # Merge with existing cache if present
                    cache = self.get_candidate_cache()
                    cache_key = f"{entity_name}_{fuzzy}"
                    cached_result = cache.get(cache_key)

                    if cached_result:
                        # Use a dict keyed by QID to ensure uniqueness
                        all_candidates = {c["id"]: c for c in cached_result if "id" in c}
                        for c in fetched_candidates:
                            if c.get("id"):
                                all_candidates[c["id"]] = c
                        merged_candidates = list(all_candidates.values())
                    else:
                        merged_candidates = fetched_candidates

                    # Update cache with merged results
                    cache.put(cache_key, merged_candidates)

                    # Log success
                    end_time = time.time()
                    timing_trace_collection.insert_one(
                        {
                            "operation_name": "_fetch_candidate",
                            "url": url,
                            "start_time": datetime.fromtimestamp(start_time),
                            "end_time": datetime.fromtimestamp(end_time),
                            "duration_seconds": end_time - start_time,
                            "status": "SUCCESS",
                            "attempt": attempts + 1,
                        }
                    )

                    return entity_name, merged_candidates
            except Exception:
                end_time = time.time()
                if attempts == 4:
                    # Log the error if all attempts failed
                    self.mongo_wrapper.log_to_db(
                        "FETCH_CANDIDATES_ERROR",
                        f"Error fetching candidates for {entity_name}",
                        traceback.format_exc(),
                        attempt=attempts + 1,
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 16)

        # If all attempts fail, return empty
        return entity_name, []

    async def fetch_candidates_batch_async(self, entities, row_texts, fuzzies, qids):
        results = {}
        cache = self.get_candidate_cache()
        to_fetch = []

        # Decide which entities need to be fetched
        for entity_name, fuzzy, row_text, qid_str in zip(entities, fuzzies, row_texts, qids):
            cache_key = f"{entity_name}_{fuzzy}"
            cached_result = cache.get(cache_key)
            forced_qids = qid_str.split() if qid_str else []

            if cached_result is not None:
                if forced_qids:
                    # Check if all forced QIDs are already present
                    cached_qids = {c["id"] for c in cached_result if "id" in c}
                    if all(q in cached_qids for q in forced_qids):
                        # All required QIDs are present, no fetch needed
                        results[entity_name] = cached_result
                    else:
                        # Forced QIDs not all present, must fetch
                        to_fetch.append((entity_name, fuzzy, row_text, qid_str))
                else:
                    # No forced QIDs, just use cached
                    results[entity_name] = cached_result
            else:
                # No cache entry, must fetch
                to_fetch.append((entity_name, fuzzy, row_text, qid_str))

        # If nothing to fetch, return what we have
        if not to_fetch:
            return self._remove_placeholders(results)

        # Fetch missing data
        async with aiohttp.ClientSession(
            timeout=MY_TIMEOUT, connector=aiohttp.TCPConnector(ssl=False, limit=10)
        ) as session:
            tasks = []
            for entity_name, fuzzy, row_text, qid_str in to_fetch:
                tasks.append(
                    self._fetch_candidates(entity_name, row_text, fuzzy, qid_str, session)
                )
            done = await asyncio.gather(*tasks, return_exceptions=False)
            for entity_name, candidates in done:
                results[entity_name] = candidates

        return self._remove_placeholders(results)

    def _remove_placeholders(self, results):
        """Removes placeholder candidates from the results based on `is_placeholder` attribute."""
        for entity_name, candidates in results.items():
            results[entity_name] = [c for c in candidates if not c.get("is_placeholder", False)]
        return results

    def fetch_candidates_batch(self, entities, row_texts, fuzzies, qids):
        return asyncio.run(self.fetch_candidates_batch_async(entities, row_texts, fuzzies, qids))

    async def _fetch_bow_for_multiple_qids(self, row_hash, row_text, qids, session):
        """
        Entry point for fetching BoW data for multiple QIDs in one row.
        This function:
          1) Splits QIDs into smaller batches;
          2) Fetches each batch sequentially (or you could parallelize);
          3) Merges results into a single dict;
          4) Returns bow_results (qid -> bow_info).
        """
        db = self.get_db()
        db[self.timing_collection_name]
        bow_cache = self.get_bow_cache()

        # 1) Check which QIDs we actually need to fetch
        to_fetch = []
        bow_results = {}

        for qid in qids:
            cache_key = f"{row_hash}_{qid}"
            cached_result = bow_cache.get(cache_key)
            if cached_result is not None:
                bow_results[qid] = cached_result
            else:
                to_fetch.append(qid)

        # If everything is cached, no need to query
        if len(to_fetch) == 0:
            return bow_results

        # 2) Break the `to_fetch` QIDs into small batches
        #    We define chunk size = MAX_BOW_BATCH_SIZE (e.g. 50).
        chunked_qids = [
            to_fetch[i : i + self.MAX_BOW_BATCH_SIZE]
            for i in range(0, len(to_fetch), self.MAX_BOW_BATCH_SIZE)
        ]

        # 3) Fetch each chunk (serially here, but could use asyncio.gather for concurrency)
        for chunk in chunked_qids:
            # We define a helper method that tries to fetch BoW data for a single chunk
            chunk_results = await self._fetch_bow_for_chunk(row_hash, row_text, chunk, session)
            # Merge the chunk results into bow_results
            for qid, data in chunk_results.items():
                bow_results[qid] = data

        return bow_results

    async def _fetch_bow_for_chunk(self, row_hash, row_text, chunk_qids, session):
        """
        Fetch BoW data for a *subset* (chunk) of QIDs.
        Includes the same backoff/retry logic as before, but only
        for these chunk_qids.
        """
        db = self.get_db()
        timing_trace_collection = db[self.timing_collection_name]
        bow_cache = self.get_bow_cache()

        # Prepare the results dictionary
        chunk_bow_results = {}

        # If empty chunk somehow, just return
        if not chunk_qids:
            return chunk_bow_results

        url = f"{self.entity_bow_endpoint}?token={self.entity_retrieval_token}"
        # The payload includes only the chunk of QIDs
        payload = {"json": {"text": row_text, "qids": chunk_qids}}

        backoff = 1
        for attempts in range(5):
            start_time = time.time()
            try:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    bow_data = await response.json()

                    # Cache the results and populate
                    for qid in chunk_qids:
                        qid_data = bow_data.get(
                            qid, {"similarity_score": 0.0, "matched_words": []}
                        )
                        cache_key = f"{row_hash}_{qid}"
                        bow_cache.put(cache_key, qid_data)
                        chunk_bow_results[qid] = qid_data

                    # Log success
                    end_time = time.time()
                    timing_trace_collection.insert_one(
                        {
                            "operation_name": "_fetch_bow_for_multiple_qids:CHUNK",
                            "url": url,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration_seconds": end_time - start_time,
                            "status": "SUCCESS",
                            "attempt": attempts + 1,
                        }
                    )

                    return chunk_bow_results

            except Exception:
                end_time = time.time()
                if attempts == 4:
                    # Log the error if all attempts failed
                    self.mongo_wrapper.log_to_db(
                        "FETCH_BOW_ERROR",
                        f"Error fetching BoW for row_hash={row_hash}, chunk_qids={chunk_qids}",
                        traceback.format_exc(),
                        attempt=attempts + 1,
                    )
                # Exponential backoff
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 16)

        # If all attempts fail, return partial or empty
        return chunk_bow_results

    def fetch_bow_vectors_batch(self, row_hash, row_text, qids):
        """
        Public method that logs the request, calls the async method,
        and returns the final BoW results.
        """
        self.get_db()

        async def runner():
            async with aiohttp.ClientSession(
                timeout=MY_TIMEOUT, connector=aiohttp.TCPConnector(ssl=False, limit=10)
            ) as session:
                return await self._fetch_bow_for_multiple_qids(row_hash, row_text, qids, session)

        return asyncio.run(runner())

    def process_rows_batch(self, docs, dataset_name, table_name):
        db = self.get_db()
        try:
            # Step 1: Collect all entities from all rows (batch) for candidate fetch
            all_entities_to_process = []
            all_row_texts = []
            all_fuzzies = []
            all_qids = []
            all_row_indices = []
            all_col_indices = []
            all_ner_types = []

            # This list will hold info for each row so we can process them individually later
            row_data_list = []

            # ---------------------------------------------------------------------
            # Gather row data and NE columns for candidate fetching
            # ---------------------------------------------------------------------
            for doc in docs:
                row = doc["data"]
                ne_columns = doc["classified_columns"]["NE"]
                context_columns = doc.get("context_columns", [])
                correct_qids = doc.get("correct_qids", {})
                row_index = doc.get("row_id", None)

                # Build row_text from the context columns
                raw_context_text = " ".join(
                    str(row[int(c)])
                    for c in sorted(context_columns, key=lambda col: str(row[int(col)]))
                )
                # Normalize row text: lowercase and remove extra spaces
                normalized_row_text = raw_context_text.lower()
                normalized_row_text = " ".join(normalized_row_text.split())

                # Hash the normalized text for caching
                row_hash = hashlib.sha256(normalized_row_text.encode()).hexdigest()

                # Save row-level info for later
                row_data_list.append(
                    (
                        doc["_id"],
                        row,
                        ne_columns,
                        context_columns,
                        correct_qids,
                        row_index,
                        raw_context_text,
                        row_hash,
                    )
                )

                # For each NE column, prepare entity lookups
                for c, ner_type in ne_columns.items():
                    c = str(c)
                    if int(c) < len(row):
                        ne_value = row[int(c)]
                        if ne_value and pd.notna(ne_value):
                            # Normalize entity value for consistent lookups
                            ne_value = str(ne_value).strip().replace("_", " ").lower()
                            correct_qid = correct_qids.get(f"{row_index}-{c}", None)

                            all_entities_to_process.append(ne_value)
                            all_row_texts.append(raw_context_text)
                            all_fuzzies.append(False)
                            all_qids.append(correct_qid)
                            all_row_indices.append(row_index)
                            all_col_indices.append(c)
                            all_ner_types.append(ner_type)

            # ---------------------------------------------------------------------
            # Fetch candidates (batch) for all entities
            # ---------------------------------------------------------------------
            # 1. Initial fetch
            candidates_results = self.fetch_candidates_batch(
                all_entities_to_process, all_row_texts, all_fuzzies, all_qids
            )

            # ---------------------------------------------------------------------
            # 2. Fuzzy retry for items that returned exactly 1 candidate
            # ---------------------------------------------------------------------
            entities_to_retry = []
            row_texts_retry = []
            fuzzies_retry = []
            qids_retry = []
            row_indices_retry = []
            col_indices_retry = []
            ner_types_retry = []

            for ne_value, r_index, c_index, n_type in zip(
                all_entities_to_process, all_row_indices, all_col_indices, all_ner_types
            ):
                candidates = candidates_results.get(ne_value, [])
                # If there's exactly 1 candidate, let's attempt a fuzzy retry
                if len(candidates) == 1:
                    entities_to_retry.append(ne_value)
                    idx = all_entities_to_process.index(ne_value)
                    row_texts_retry.append(all_row_texts[idx])
                    fuzzies_retry.append(True)
                    correct_qid = all_qids[idx]
                    qids_retry.append(correct_qid)
                    row_indices_retry.append(r_index)
                    col_indices_retry.append(c_index)
                    ner_types_retry.append(n_type)
                else:
                    # Keep the existing candidates
                    candidates_results[ne_value] = candidates

            if entities_to_retry:
                retry_results = self.fetch_candidates_batch(
                    entities_to_retry, row_texts_retry, fuzzies_retry, qids_retry
                )
                for ne_value in entities_to_retry:
                    candidates_results[ne_value] = retry_results.get(ne_value, [])

            # ---------------------------------------------------------------------
            # Process each row individually (including BoW retrieval for that row)
            # ---------------------------------------------------------------------
            for (
                doc_id,
                row,
                ne_columns,
                context_columns,
                correct_qids,
                row_index,
                raw_context_text,
                row_hash,
            ) in row_data_list:
                # -------------------------------------------------------------
                # 1. Gather the QIDs relevant to this row
                # -------------------------------------------------------------
                row_qids = []
                for c, ner_type in ne_columns.items():
                    if int(c) < len(row):
                        ne_value = row[int(c)]
                        if ne_value and pd.notna(ne_value):
                            ne_value = str(ne_value).strip().replace("_", " ").lower()
                            candidates = candidates_results.get(ne_value, [])
                            for cand in candidates:
                                if cand["id"]:
                                    row_qids.append(cand["id"])
                row_qids = list(set(q for q in row_qids if q))

                # -------------------------------------------------------------
                # 2. Fetch BoW vectors for this row’s QIDs
                # -------------------------------------------------------------
                if row_qids and self.entity_bow_endpoint and self.entity_retrieval_token:
                    bow_data = self.fetch_bow_vectors_batch(row_hash, raw_context_text, row_qids)
                else:
                    bow_data = {}

                # -------------------------------------------------------------
                # 3. Build final linked_entities + training_candidates
                # -------------------------------------------------------------
                linked_entities = {}
                training_candidates_by_ne_column = {}

                for c, ner_type in ne_columns.items():
                    c = str(c)
                    if int(c) < len(row):
                        ne_value = row[int(c)]
                        if ne_value and pd.notna(ne_value):
                            ne_value = str(ne_value).strip().replace("_", " ").lower()
                            candidates = candidates_results.get(ne_value, [])

                            # Assign the BoW score + numeric NER type
                            for cand in candidates:
                                qid = cand["id"]
                                cand["features"]["bow_similarity"] = bow_data.get(qid, {}).get(
                                    "similarity_score", 0.0
                                )
                                cand["features"]["column_NERtype"] = self.map_nertype_to_numeric(
                                    ner_type
                                )

                            # 4. Rank candidates by feature scoring
                            ranked_candidates = self.rank_with_feature_scoring(candidates)

                            # 4.1 Ensure they are sorted by 'score' descending
                            ranked_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)

                            # If there's a correct_qid, ensure it appears in the top training slice
                            correct_qid = correct_qids.get(f"{row_index}-{c}", None)
                            if correct_qid and correct_qid not in [
                                rc["id"]
                                for rc in ranked_candidates[: self.max_training_candidates]
                            ]:
                                correct_candidate = next(
                                    (x for x in ranked_candidates if x["id"] == correct_qid), None
                                )
                                if correct_candidate:
                                    top_slice = ranked_candidates[
                                        : self.max_training_candidates - 1
                                    ]
                                    top_slice.append(correct_candidate)
                                    ranked_candidates = top_slice
                                    # Re-sort after adding the correct candidate
                                    ranked_candidates.sort(
                                        key=lambda x: x.get("score", 0.0), reverse=True
                                    )

                            el_results_candidates = ranked_candidates[: self.max_candidates]
                            linked_entities[c] = el_results_candidates
                            training_candidates_by_ne_column[c] = ranked_candidates[
                                : self.max_training_candidates
                            ]

                # -------------------------------------------------------------
                # 5. Save to DB: training candidates + final EL results
                # -------------------------------------------------------------
                self.save_candidates_for_training(
                    training_candidates_by_ne_column, dataset_name, table_name, row_index
                )
                db[self.input_collection].update_one(
                    {"_id": doc_id}, {"$set": {"el_results": linked_entities, "status": "DONE"}}
                )

            # Optionally track speed after full batch
            self.log_processing_speed(dataset_name, table_name)

        except Exception:
            self.mongo_wrapper.log_to_db(
                "ERROR", "Error processing batch of rows", traceback.format_exc()
            )

    def map_kind_to_numeric(self, kind):
        mapping = {"entity": 1, "type": 2, "disambiguation": 3, "predicate": 4}
        return mapping.get(kind, 1)

    def map_nertype_to_numeric(self, nertype):
        mapping = {
            "LOCATION": 1,
            "LOC": 1,
            "ORGANIZATION": 2,
            "ORG": 2,
            "PERSON": 3,
            "PERS": 3,
            "OTHER": 4,
            "OTHERS": 4,
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
        tokens = [string[i : i + n] for i in range(len(string) - n + 1)]
        return tokens

    def tokenize_text(self, text):
        tokens = word_tokenize(text.lower())
        return set(t for t in tokens if t not in stop_words)

    def _process_candidates(self, candidates, entity_name, row_tokens):
        processed_candidates = []
        for candidate in candidates:
            candidate_name = candidate.get("name", "")
            candidate_description = candidate.get("description", "") or ""
            kind_numeric = self.map_kind_to_numeric(candidate.get("kind", "entity"))
            nertype_numeric = self.map_nertype_to_numeric(candidate.get("NERtype", "OTHERS"))

            features = {
                "ntoken_mention": candidate.get("ntoken_mention", len(entity_name.split())),
                "ntoken_entity": candidate.get("ntoken_entity", len(candidate_name.split())),
                "length_mention": len(entity_name),
                "length_entity": len(candidate_name),
                "popularity": candidate.get("popularity", 0.0),
                "ed_score": candidate.get("ed_score", 0.0),
                "jaccard_score": candidate.get("jaccard_score", 0.0),
                "jaccardNgram_score": candidate.get("jaccardNgram_score", 0.0),
                "desc": self.calculate_token_overlap(
                    row_tokens, set(self.tokenize_text(candidate_description))
                ),
                "descNgram": self.calculate_ngram_similarity(entity_name, candidate_description),
                "bow_similarity": 0.0,
                "kind": kind_numeric,
                "NERtype": nertype_numeric,
                "column_NERtype": None,
            }

            processed_candidates.append(
                {
                    "id": candidate.get("id"),
                    "name": candidate_name,
                    "description": candidate_description,
                    "types": candidate.get("types"),
                    "features": features,
                }
            )
        return processed_candidates

    def score_candidate(self, candidate):
        """
        Equal-weight formula across key features [0..1].
        Includes 'popularity' as well.
        Result is simply the mean of all chosen features, each in [0..1].
        """
        feat_names = [
            "ed_score",
            "jaccard_score",
            "jaccardNgram_score",
            "desc",
            "descNgram",
            "bow_similarity",
            "popularity",
        ]

        # Collect them, defaulting to 0.0 if missing
        feats = [candidate["features"].get(fname, 0.0) for fname in feat_names]

        # Just compute the average
        if not feats:
            total_score = 0.0
        else:
            total_score = sum(feats) / len(feats)

        candidate["score"] = round(total_score, 3)
        return candidate

    def rank_with_feature_scoring(self, candidates):
        scored_candidates = [self.score_candidate(c) for c in candidates]
        return sorted(scored_candidates, key=lambda x: x["score"], reverse=True)

    def save_candidates_for_training(
        self, candidates_by_ne_column, dataset_name, table_name, row_index
    ):
        db = self.get_db()
        training_collection = db[self.training_collection_name]
        training_document = {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_index,
            "candidates": candidates_by_ne_column,
            "ml_ranked": False,
        }
        training_collection.insert_one(training_document)

    def log_processing_speed(self, dataset_name, table_name):
        db = self.get_db()
        table_trace_collection = db[self.table_trace_collection_name]

        trace = table_trace_collection.find_one(
            {"dataset_name": dataset_name, "table_name": table_name}
        )
        if not trace:
            return
        processed_rows = trace.get("processed_rows", 1)
        start_time = trace.get("start_time")
        elapsed_time = (datetime.now() - start_time).total_seconds() if start_time else 0
        rows_per_second = processed_rows / elapsed_time if elapsed_time > 0 else 0
        table_trace_collection.update_one(
            {"dataset_name": dataset_name, "table_name": table_name},
            {"$set": {"rows_per_second": rows_per_second}},
        )

    def claim_todo_batch(self, input_collection, batch_size=10):
        """
        Atomically claims a batch of TODO documents by setting them to DOING,
        and returns the full documents so we don't have to fetch them again.
        """
        docs = []
        for _ in range(batch_size):
            doc = input_collection.find_one_and_update(
                {"status": "TODO"}, {"$set": {"status": "DOING"}}
            )
            if doc is None:
                # No more TODO docs
                break
            docs.append(doc)
        return docs

    def worker(self):
        db = self.get_db()
        input_collection = db[self.input_collection]

        while True:
            # Atomically claim a batch of documents (full docs, not partial)
            todo_docs = self.claim_todo_batch(input_collection)
            if not todo_docs:
                print("No more tasks to process.")
                break

            # Group the claimed documents by (dataset_name, table_name)
            tasks_by_table = {}
            for doc in todo_docs:
                dataset_name = doc["dataset_name"]
                table_name = doc["table_name"]
                # Accumulate full docs in a list
                tasks_by_table.setdefault((dataset_name, table_name), []).append(doc)

            # Process each group as a batch
            for (dataset_name, table_name), docs in tasks_by_table.items():
                self.process_rows_batch(docs, dataset_name, table_name)

    def run(self):
        # mp.set_start_method("spawn", force=True) # it slows down everything

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
                ml_model_path=self.model_path,
                batch_size=self.batch_size,
                max_candidates=self.max_candidates,
                top_n_for_type_freq=self.top_n_for_type_freq,
            )
            p.start()
            processes.append(p)

        trace_thread = TraceWorker(
            self.mongo_uri,
            self.db_name,
            self.input_collection,
            self.dataset_trace_collection_name,
            self.table_trace_collection_name,
            self.timing_collection_name,
        )
        trace_thread.start()
        processes.append(trace_thread)

        for p in processes:
            p.join()

        self.__del__()
        print("All tasks have been processed.")
