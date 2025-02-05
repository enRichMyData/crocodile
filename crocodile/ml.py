from collections import Counter, defaultdict
from multiprocessing import Process

import numpy as np

from crocodile.mongo import MongoConnectionManager, MongoWrapper


class MLWorker(Process):
    def __init__(
        self,
        db_uri,
        db_name,
        table_trace_collection_name,
        training_collection_name,
        timing_collection_name,
        error_log_collection_name,
        input_collection,
        ml_model_path,
        batch_size=100,
        max_candidates=5,
        top_n_for_type_freq=3,
    ):
        super(MLWorker, self).__init__()
        self.db_uri = db_uri
        self.db_name = db_name
        self.table_trace_collection_name = table_trace_collection_name
        self.training_collection_name = training_collection_name
        self.timing_collection_name = timing_collection_name
        self.error_log_collection_name = error_log_collection_name
        self.input_collection = input_collection
        self.ml_model_path = ml_model_path
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        self.top_n_for_type_freq = top_n_for_type_freq
        self.mongo_wrapper = MongoWrapper(
            db_uri,
            db_name,
            timing_collection_name=self.timing_collection_name,
            error_log_collection_name=self.error_log_collection_name,
        )

    def get_db(self):
        return MongoConnectionManager.get_client(self.db_uri)[self.db_name]

    def load_ml_model(self):
        from tensorflow.keras.models import load_model

        return load_model(self.ml_model_path)

    def run(self):
        db = self.get_db()
        model = self.load_ml_model()
        table_trace_collection = db[self.table_trace_collection_name]
        while True:
            todo_table = self.mongo_wrapper.find_one_document(
                table_trace_collection, {"ml_ranking_status": None}
            )
            if todo_table is None:  # no more tasks to process
                break
            table_trace_obj = self.mongo_wrapper.find_one_and_update(
                table_trace_collection,
                {"$and": [{"status": "COMPLETED"}, {"ml_ranking_status": None}]},
                {"$set": {"ml_ranking_status": "PENDING"}},
            )
            if table_trace_obj:
                dataset_name = table_trace_obj.get("dataset_name")
                table_name = table_trace_obj.get("table_name")
                self.apply_ml_ranking(dataset_name, table_name, model)

    def apply_ml_ranking(self, dataset_name, table_name, model):
        db = self.get_db()
        training_collection = db[self.training_collection_name]
        input_collection = db[self.input_collection]
        table_trace_collection = db[self.table_trace_collection_name]

        processed_count = 0
        total_count = self.mongo_wrapper.count_documents(
            training_collection,
            {"dataset_name": dataset_name, "table_name": table_name, "ml_ranked": False},
        )
        print(f"Total unprocessed documents (for ML ranking): {total_count}")

        # We'll restrict type-freq counting to the top N candidates in each row/col
        top_n_for_type_freq = self.top_n_for_type_freq  # e.g., 3

        while processed_count < total_count:
            batch_docs = list(
                training_collection.find(
                    {"dataset_name": dataset_name, "table_name": table_name, "ml_ranked": False},
                    limit=self.batch_size,
                )
            )
            if not batch_docs:
                break

            # Map documents by _id so we can update them later
            doc_map = {doc["_id"]: doc for doc in batch_docs}

            # For each column, we'll store a Counter of type_id -> how many rows had that type
            type_freq_by_column = defaultdict(Counter)

            # We also track how many rows in the batch actually had that column
            # so we can normalize frequencies to 0..1
            rows_count_by_column = Counter()

            # --------------------------------------------------------------------------
            # 1) Collect "top N" type IDs per column, ignoring duplicates in same row
            # --------------------------------------------------------------------------
            for doc in batch_docs:
                candidates_by_column = doc["candidates"]
                for col_index, candidates in candidates_by_column.items():
                    top_candidates_for_freq = candidates[:top_n_for_type_freq]

                    # Collect distinct type IDs from those top candidates
                    row_qids = set()
                    for cand in top_candidates_for_freq:
                        for t_dict in cand.get("types", []):
                            qid = t_dict.get("id")
                            if qid:
                                row_qids.add(qid)

                    # Increase counts for each distinct type in this row
                    for qid in row_qids:
                        type_freq_by_column[col_index][qid] += 1

                    # Mark that this row *had* that column (so we can normalize)
                    rows_count_by_column[col_index] += 1

            # --------------------------------------------------------------------------
            # 2) Convert raw counts to frequencies in [0..1].
            # --------------------------------------------------------------------------
            for col_index, freq_counter in type_freq_by_column.items():
                row_count = rows_count_by_column[col_index]
                if row_count == 0:
                    continue
                # Convert each type's raw count => ratio in [0..1]
                for qid in freq_counter:
                    freq_counter[qid] = freq_counter[qid] / row_count

            # --------------------------------------------------------------------------
            # 3) Assign new features (typeFreq1..typeFreq5) for each candidate
            # --------------------------------------------------------------------------
            for doc in batch_docs:
                candidates_by_column = doc["candidates"]
                for col_index, candidates in candidates_by_column.items():
                    # If we never built a freq for this column, default to empty
                    freq_counter = type_freq_by_column.get(col_index, {})

                    for cand in candidates:
                        # Ensure we have a features dict
                        if "features" not in cand:
                            cand["features"] = {}

                        # Gather candidate's type IDs
                        cand_qids = [
                            t_obj.get("id") for t_obj in cand.get("types", []) if t_obj.get("id")
                        ]

                        # Find the frequency for each type
                        cand_type_freqs = [freq_counter.get(qid, 0.0) for qid in cand_qids]

                        # Sort descending
                        cand_type_freqs.sort(reverse=True)

                        # Assign typeFreq1..typeFreq5
                        for i in range(1, 6):
                            # If the candidate has fewer than i types, default to 0
                            freq_val = (
                                cand_type_freqs[i - 1] if (i - 1) < len(cand_type_freqs) else 0.0
                            )
                            cand["features"][f"typeFreq{i}"] = round(freq_val, 3)

            # --------------------------------------------------------------------------
            # 4) Build final feature matrix & do ML predictions
            # --------------------------------------------------------------------------
            all_candidates = []
            doc_info = []
            for doc in batch_docs:
                row_index = doc["row_id"]
                candidates_by_column = doc["candidates"]
                for col_index, candidates in candidates_by_column.items():
                    features_list = [self.extract_features(c) for c in candidates]
                    all_candidates.extend(features_list)

                    # doc_info: track the location of each candidate
                    doc_info.extend(
                        [(doc["_id"], row_index, col_index, idx) for idx in range(len(candidates))]
                    )

            if len(all_candidates) == 0:
                print(
                    f"No candidates to predict for dataset={dataset_name}, table={table_name}. "
                    "Skipping..."
                )
                return

            candidate_features = np.array(all_candidates)
            print(f"Predicting scores for {len(candidate_features)} candidates...")
            ml_scores = model.predict(candidate_features, batch_size=128)[:, 1]
            print("Scores predicted.")

            # Assign new ML scores back to candidates
            for i, (doc_id, row_index, col_index, cand_idx) in enumerate(doc_info):
                candidate = doc_map[doc_id]["candidates"][col_index][cand_idx]
                candidate["score"] = float(ml_scores[i])

            # --------------------------------------------------------------------------
            # 5) Sort by final 'score' and trim to max_candidates; update DB
            # --------------------------------------------------------------------------
            for doc_id, doc in doc_map.items():
                row_index = doc["row_id"]
                updated_candidates_by_column = {}
                for col_index, candidates in doc["candidates"].items():
                    ranked_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
                    updated_candidates_by_column[col_index] = ranked_candidates[
                        : self.max_candidates
                    ]

                # Mark doc as ML-ranked
                training_collection.update_one(
                    {"_id": doc_id}, {"$set": {"candidates": doc["candidates"], "ml_ranked": True}}
                )

                # Update final results in input_collection
                input_collection.update_one(
                    {"dataset_name": dataset_name, "table_name": table_name, "row_id": row_index},
                    {"$set": {"el_results": updated_candidates_by_column}},
                )

            processed_count += len(batch_docs)
            progress = min((processed_count / total_count) * 100, 100.0)
            print(
                f"ML ranking progress for {dataset_name}.{table_name}: {progress:.2f}% completed"
            )

            # Update progress in table_trace
            table_trace_collection.update_one(
                {"dataset_name": dataset_name, "table_name": table_name},
                {"$set": {"ml_ranking_progress": progress}},
                upsert=True,
            )

        # Mark the table as COMPLETED
        table_trace_collection.update_one(
            {"dataset_name": dataset_name, "table_name": table_name},
            {"$set": {"ml_ranking_status": "COMPLETED"}},
        )
        print("ML ranking completed.")

    def extract_features(self, candidate):
        numerical_features = [
            "ntoken_mention",
            "length_mention",
            "ntoken_entity",
            "length_entity",
            "popularity",
            "ed_score",
            "desc",
            "descNgram",
            "bow_similarity",
            "kind",
            "NERtype",
            "column_NERtype",
        ]
        return [candidate["features"].get(feature, 0.0) for feature in numerical_features]
