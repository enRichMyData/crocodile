import hashlib
import traceback

import pandas as pd

from crocodile.feature import map_nertype_to_numeric
from crocodile.fetchers import BowFetcher, CandidateFetcher
from crocodile.mongo import MongoWrapper


class RowBatchProcessor:
    """
    Extracted logic for process_rows_batch (and associated scoring helpers).
    Takes the Crocodile instance so we can reference .mongo_wrapper, .feature, etc.
    """

    def __init__(
        self,
        candidate_fetcher: CandidateFetcher,
        input_collection: str,
        training_collection: str,
        max_training_candidates: int = 16,
        max_candidates: int = 5,
        bow_fetcher: BowFetcher | None = None,
        **kwargs,
    ):
        self.candidate_fetcher = candidate_fetcher
        self.input_collection = input_collection
        self.training_collection = training_collection
        self.max_training_candidates = max_training_candidates
        self.max_candidates = max_candidates
        self.bow_fetcher = bow_fetcher
        self._db_name = kwargs.get("db_name", "crocodile_db")
        self._mongo_uri = kwargs.get("mongo_uri", "mongodb://mongodb:27017")
        self.mongo_wrapper = MongoWrapper(self._mongo_uri, self._db_name)

    def get_db(self):
        """Get MongoDB database connection for current process"""
        from crocodile.mongo import MongoConnectionManager

        client = MongoConnectionManager.get_client(self._mongo_uri)
        return client[self._db_name]

    def process_rows_batch(self, docs, dataset_name, table_name):
        """
        Orchestrates the overall flow:
          1) Collect all entities from the batch for candidate fetching.
          2) Fetch initial candidates (batch).
          3) Attempt fuzzy retry if needed.
          4) Process each row individually, fetching BoW data as needed.
          5) Save results and update DB.
        """
        db = self.get_db()
        try:
            # 1) Gather all needed info from docs
            (
                all_entities_to_process,
                all_row_texts,
                all_fuzzies,
                all_qids,
                row_data_list,
                all_row_indices,
                all_col_indices,
                all_ner_types,
            ) = self._collect_batch_info(docs)

            # 2) Fetch initial candidates in one batch
            candidates_results = self._fetch_all_candidates(
                all_entities_to_process,
                all_row_texts,
                all_fuzzies,
                all_qids,
                all_row_indices,
                all_col_indices,
                all_ner_types,
            )

            # 3) Process each row (BoW fetch, final ranking, DB update)
            self._process_rows_individually(
                row_data_list, candidates_results, dataset_name, table_name, db
            )

        except Exception:
            self.mongo_wrapper.log_to_db(
                "ERROR", "Error processing batch of rows", traceback.format_exc()
            )

    # --------------------------------------------------------------------------
    # 1) GATHER BATCH INFO
    # --------------------------------------------------------------------------
    def _collect_batch_info(self, docs):
        """
        Collects and returns all the lists needed for the candidate-fetch step
        plus a row_data_list for further processing.
        """
        all_entities_to_process = []
        all_row_texts = []
        all_fuzzies = []
        all_qids = []
        all_row_indices = []
        all_col_indices = []
        all_ner_types = []
        row_data_list = []

        for doc in docs:
            row = doc["data"]
            ne_columns = doc["classified_columns"]["NE"]
            context_columns = doc.get("context_columns", [])
            correct_qids = doc.get("correct_qids", {})
            row_index = doc.get("row_id", None)

            # Build a text from the "context_columns"
            raw_context_text = " ".join(
                str(row[int(c)])
                for c in sorted(context_columns, key=lambda col: str(row[int(col)]))
            )
            normalized_row_text = raw_context_text.lower()
            normalized_row_text = " ".join(normalized_row_text.split())
            row_hash = hashlib.sha256(normalized_row_text.encode()).hexdigest()

            # Collect row-level info
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

            # Collect all named-entity columns for candidate fetch
            for c, ner_type in ne_columns.items():
                c = str(c)
                if int(c) < len(row):
                    ne_value = row[int(c)]
                    if ne_value and pd.notna(ne_value):
                        ne_value = str(ne_value).strip().replace("_", " ").lower()
                        correct_qid = correct_qids.get(f"{row_index}-{c}", None)

                        all_entities_to_process.append(ne_value)
                        all_row_texts.append(raw_context_text)
                        all_fuzzies.append(False)
                        all_qids.append(correct_qid)
                        all_row_indices.append(row_index)
                        all_col_indices.append(c)
                        all_ner_types.append(ner_type)

        return (
            all_entities_to_process,
            all_row_texts,
            all_fuzzies,
            all_qids,
            row_data_list,
            all_row_indices,
            all_col_indices,
            all_ner_types,
        )

    # --------------------------------------------------------------------------
    # 2) FETCH INITIAL CANDIDATES + FUZZY RETRY
    # --------------------------------------------------------------------------
    def _fetch_all_candidates(
        self,
        all_entities_to_process,
        all_row_texts,
        all_fuzzies,
        all_qids,
        all_row_indices,
        all_col_indices,
        all_ner_types,
    ):
        """
        Performs the initial batch fetch of candidates, then does fuzzy retry
        for any entity that returned <= 1 candidate.
        """
        # 1) Initial fetch
        candidates_results = self.candidate_fetcher.fetch_candidates_batch(
            all_entities_to_process, all_row_texts, all_fuzzies, all_qids
        )

        # 2) Fuzzy retry if needed
        entities_to_retry = []
        row_texts_retry = []
        fuzzies_retry = []
        qids_retry = []

        for ne_value, r_index, c_index, n_type in zip(
            all_entities_to_process, all_row_indices, all_col_indices, all_ner_types
        ):
            candidates = candidates_results.get(ne_value, [])
            if len(candidates) <= 1:
                entities_to_retry.append(ne_value)
                idx = all_entities_to_process.index(ne_value)
                row_texts_retry.append(all_row_texts[idx])
                fuzzies_retry.append(True)
                qids_retry.append(all_qids[idx])

        if entities_to_retry:
            retry_results = self.candidate_fetcher.fetch_candidates_batch(
                entities_to_retry, row_texts_retry, fuzzies_retry, qids_retry
            )
            for ne_value in entities_to_retry:
                candidates_results[ne_value] = retry_results.get(ne_value, [])

        return candidates_results

    # --------------------------------------------------------------------------
    # 3) PROCESS EACH ROW INDIVIDUALLY
    # --------------------------------------------------------------------------
    def _process_rows_individually(
        self, row_data_list, candidates_results, dataset_name, table_name, db
    ):
        """
        For each row:
          - Gather QIDs
          - Fetch BoW vectors
          - Rank candidates
          - Save final results + training candidates
        """
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
            # Gather all QIDs in this row
            row_qids = self._collect_row_qids(ne_columns, row, candidates_results)

            # Fetch BoW data if needed
            bow_data = {}
            if row_qids and self.bow_fetcher is not None:
                bow_data = self.bow_fetcher.fetch_bow_vectors_batch(
                    row_hash, raw_context_text, row_qids
                )

            # Rank and build final results
            (
                linked_entities,
                training_candidates_by_ne_column,
            ) = self._build_linked_entities_and_training(
                ne_columns, row, correct_qids, row_index, candidates_results, bow_data
            )

            # Save to DB - Ensure status is explicitly set to "DONE"
            self.save_candidates_for_training(
                training_candidates_by_ne_column, dataset_name, table_name, row_index
            )
            db[self.input_collection].update_one(
                {"_id": doc_id}, {"$set": {"el_results": linked_entities, "status": "DONE"}}
            )

    def _collect_row_qids(self, ne_columns, row, candidates_results):
        """
        Collects the QIDs for all entities in a single row.
        """
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
        return list(set(q for q in row_qids if q))

    def _build_linked_entities_and_training(
        self, ne_columns, row, correct_qids, row_index, candidates_results, bow_data
    ):
        """
        For each NE column in the row:
          - Insert bow_similarity and column_NERtype features
          - Rank candidates
          - Insert correct candidate if missing
          - Return final top K + training slice
        """
        linked_entities = {}
        training_candidates_by_ne_column = {}

        for c, ner_type in ne_columns.items():
            c = str(c)
            if int(c) < len(row):
                ne_value = row[int(c)]
                if ne_value and pd.notna(ne_value):
                    ne_value = str(ne_value).strip().replace("_", " ").lower()
                    candidates = candidates_results.get(ne_value, [])

                    # Update features with BoW data
                    for cand in candidates:
                        qid = cand["id"]
                        if cand.get("features"):
                            cand["features"]["bow_similarity"] = bow_data.get(qid, {}).get(
                                "similarity_score", 0.0
                            )
                            cand["features"]["column_NERtype"] = map_nertype_to_numeric(ner_type)

                    # Rank
                    ranked_candidates = self.rank_with_feature_scoring(candidates)
                    ranked_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)

                    # If correct QID is missing in top training slice, insert it
                    correct_qid = correct_qids.get(f"{row_index}-{c}", None)
                    if correct_qid and correct_qid not in [
                        rc["id"] for rc in ranked_candidates[: self.max_training_candidates]
                    ]:
                        correct_candidate = next(
                            (x for x in ranked_candidates if x["id"] == correct_qid), None
                        )
                        if correct_candidate:
                            top_slice = ranked_candidates[: self.max_training_candidates - 1]
                            top_slice.append(correct_candidate)
                            ranked_candidates = top_slice
                            ranked_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)

                    # Slice final results
                    el_results_candidates = ranked_candidates[: self.max_candidates]
                    training_candidates = ranked_candidates[: self.max_training_candidates]

                    linked_entities[c] = el_results_candidates
                    training_candidates_by_ne_column[c] = training_candidates

        return linked_entities, training_candidates_by_ne_column

    # --------------------------------------------------------------------------
    # SCORING + TRAINING
    # --------------------------------------------------------------------------
    def score_candidate(self, candidate):
        """
        This used to be Crocodile.score_candidate.
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
        feats = []
        if candidate.get("features"):
            feats = [candidate["features"].get(fname, 0.0) for fname in feat_names]
        total_score = sum(feats) / len(feats) if feats else 0.0
        candidate["score"] = total_score
        return candidate

    def rank_with_feature_scoring(self, candidates):
        """
        This used to be Crocodile.rank_with_feature_scoring.
        """
        scored_candidates = [self.score_candidate(c) for c in candidates]
        return sorted(scored_candidates, key=lambda x: x["score"], reverse=True)

    def save_candidates_for_training(
        self, candidates_by_ne_column, dataset_name, table_name, row_index
    ):
        """
        This used to be Crocodile.save_candidates_for_training.
        """
        db = self.get_db()
        training_collection = db[self.training_collection]
        training_document = {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_index,
            "candidates": candidates_by_ne_column,
            "ml_ranked": False,
        }
        training_collection.insert_one(training_document)
