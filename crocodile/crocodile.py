import requests
import time
from pymongo import MongoClient
import multiprocessing as mp
import logging
import traceback
from datetime import datetime
import base64
import gzip
from nltk.tokenize import word_tokenize
import nltk
import pickle

# Download NLTK resources if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Global stopwords to avoid reinitializing repeatedly
stop_words = set(stopwords.words('english'))

class Crocodile:
    def __init__(self, mongo_uri="mongodb://mongodb:27017/", db_name="crocodile_db", 
                 table_trace_collection_name="table_trace", dataset_trace_collection_name="dataset_trace", 
                 collection_name="input_data", training_collection_name="training_data", 
                 max_workers=None, max_candidates=5, max_training_candidates=10, 
                 entity_retrieval_endpoint=None, entity_bow_endpoint=None, entity_retrieval_token=None, 
                 selected_features=None, candidate_retrieval_limit=100):
        
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.table_trace_collection_name = table_trace_collection_name
        self.dataset_trace_collection_name = dataset_trace_collection_name
        self.training_collection_name = training_collection_name
        self.max_workers = max_workers or mp.cpu_count()
        self.max_candidates = max_candidates
        self.max_training_candidates = max_training_candidates
        self.entity_retrieval_endpoint = entity_retrieval_endpoint
        self.entity_bow_endpoint = entity_bow_endpoint
        self.entity_retrieval_token = entity_retrieval_token
        self.candidate_retrieval_limit = candidate_retrieval_limit
        self.selected_features = selected_features or [
            "ntoken_mention", "ntoken_entity", "length_mention", "length_entity",
            "popularity", "ed_score", "jaccard_score", "jaccardNgram_score", "bow_similarity", "desc", "descNgram"
        ]
        logging.basicConfig(filename='crocodile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def update_table_trace(self, dataset_name, table_name, increment=0, status=None, start_time=None, end_time=None, row_time=None):
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

    def update_dataset_trace(self, dataset_name):
        """Update the dataset-level trace based on the progress of tables."""
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        table_trace_collection = db[self.table_trace_collection_name]
        dataset_trace_collection = db[self.dataset_trace_collection_name]

        dataset_trace = dataset_trace_collection.find_one({"dataset_name": dataset_name})

        if not dataset_trace:
            return

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

        elapsed_time = sum(table.get("duration", 0) for table in tables if table.get("duration", 0) > 0)
        if processed_rows > 0:
            avg_time_per_row = elapsed_time / processed_rows
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
            "status": status
        }

        dataset_trace_collection.update_one(
            {"dataset_name": dataset_name},
            {"$set": update_fields},
            upsert=True
        )

    def fetch_candidates(self, entity_name, row_text, fuzzy=False, qid=None):
        """Fetch candidates for a given entity synchronously, with an optional QID parameter."""
        try:
            if not self.entity_retrieval_endpoint or not self.entity_retrieval_token:
                raise ValueError("Entity retrieval endpoint and token must be provided.")

            # Construct the URL
            url = f"{self.entity_retrieval_endpoint}?name={entity_name}&limit={self.candidate_retrieval_limit}&fuzzy={fuzzy}&token={self.entity_retrieval_token}"
            
            # Add QID to the URL if provided
            if qid:
                url += f"&ids={qid}"

            response = requests.get(url, headers={'accept': 'application/json'}, timeout=10)
            response.raise_for_status()

            candidates = response.json()

            # Tokenize the row_text for desc similarity
            row_tokens = set(self.tokenize_text(row_text))

            # Process candidates by adding features and fallback
            filtered_candidates = []
            for candidate in candidates:
                candidate_name = candidate.get('name', '')
                candidate_description = candidate.get('description', '') if candidate.get('description') is not None else ""

                # Add kind and NERtype with mapping to numerical values
                kind = candidate.get('kind', 'entity')
                nertype = candidate.get('NERtype', 'OTHERS')

                kind_numeric = self.map_kind_to_numeric(kind)
                nertype_numeric = self.map_nertype_to_numeric(nertype)

                # Group all features in a dictionary called 'features'
                features = {
                    'ntoken_mention': round(candidate.get('ntoken_mention', len(entity_name.split())), 4),
                    'length_mention': round(candidate.get('length_mention', len(entity_name)), 4),
                    'ntoken_entity': round(candidate.get('ntoken_entity', len(candidate_name.split())), 4),
                    'length_entity': round(candidate.get('length_entity', len(candidate_name)), 4),
                    'popularity': round(candidate.get('popularity', 0.0), 4),
                    'ed_score': round(candidate.get('ed_score', 0.0), 4),
                    'desc': round(self.calculate_token_overlap(row_tokens, set(self.tokenize_text(candidate_description))), 4),
                    'descNgram': round(self.calculate_ngram_similarity(row_text, candidate_description), 4),
                    'bow_similarity': 0.0,
                    'kind': kind_numeric,
                    'NERtype': nertype_numeric
                }

                filtered_candidate = {
                    'id': candidate.get('id'),
                    'name': candidate_name,
                    'description': candidate_description,
                    'types': candidate.get('types'),
                    'features': features
                }

                filtered_candidates.append(filtered_candidate)

            return filtered_candidates

        except Exception as e:
            logging.error(f"Error occurred while fetching candidates for {entity_name}: {str(e)}\n{traceback.format_exc()}")
            return []

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

    def score_candidate(self, ne_value, candidate):
        ne_value = ne_value if ne_value is not None else ""

        ed_score = candidate['features'].get('ed_score', 0.0)
        desc_score = candidate['features'].get('desc', 0.0)
        desc_ngram_score = candidate['features'].get('descNgram', 0.0)
        bow_similarity = candidate['features'].get('bow_similarity', 0.0)

        # Incorporate BoW similarity into the total score
        total_score = (ed_score + desc_score + desc_ngram_score + bow_similarity) / 4
       
        candidate['score'] = round(total_score, 2)

        return candidate

    def score_candidates(self, ne_value, candidates):
        scored_candidates = []
        for candidate in candidates:
            scored_candidate = self.score_candidate(ne_value, candidate)
            scored_candidates.append(scored_candidate)

        # Sort candidates by score in descending order
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        return scored_candidates

    def get_bow_from_api(self, qids):
        """Fetch BoW vectors from the API."""
        if not self.entity_bow_endpoint or not self.entity_retrieval_token:
            raise ValueError("BoW API endpoint and token must be provided.")

        url = f'{self.entity_bow_endpoint}?token={self.entity_retrieval_token}'

        try:
            response = requests.post(
                url,
                headers={'accept': 'application/json', 'Content-Type': 'application/json'},
                json={"json": qids}
            )
            if response.status_code != 200:
                logging.error(f"Error fetching BoW vectors: {response.status_code}")
                return None

            bow_data = response.json()
            decoded_vectors = {}
            for qid, encoded_data in bow_data.items():
                compressed_bytes = base64.b64decode(encoded_data)
                decompressed_vector = pickle.loads(gzip.decompress(compressed_bytes))
                bow_vector = decompressed_vector
                decoded_vectors[qid] = bow_vector

            return decoded_vectors

        except Exception as e:
            logging.error(f"Exception while fetching BoW vectors: {str(e)}")
            return None

    def tokenize_text(self, text):
        """Tokenize and clean the text."""
        tokens = word_tokenize(text.lower())
        return set(t for t in tokens if t not in stop_words)

    def compute_bow_similarity(self, row_text, candidate_vectors):
        """New BoW similarity computation using Jaccard similarity."""
        if candidate_vectors is None:
            logging.error("No candidate vectors available to compute BoW similarity.")
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

        row_values = list(row.values())  # Ensure we access the values in the correct order
        
        # Build the row_text using context columns only (converting to integers since context_columns are strings)
        row_text = ' '.join([str(row_values[int(col_index)]) for col_index in context_columns if int(col_index) < len(row_values)])

        for col_index, ner_type in ne_columns.items():
            col_index = str(col_index)  # Column index as a string
            if int(col_index) < len(row_values):  # Avoid out-of-range access
                ne_value = row_values[int(col_index)]
                if ne_value:
                    correct_qid = correct_qids.get(f"{row_index}-{col_index}", None)  # Access the correct QID using (row_index, col_index)

                    candidates = self.fetch_candidates(ne_value, row_text, qid=correct_qid)

                    # Fetch BoW vectors for the candidates
                    candidate_qids = [candidate['id'] for candidate in candidates]
                    candidate_bows = self.get_bow_from_api(candidate_qids)

                    # Compute BoW similarity with the row
                    bow_similarities, matched_words = self.compute_bow_similarity(row_text, candidate_bows)

                    # Map NER type from input to numeric (extended names)
                    ner_type_numeric = self.map_nertype_to_numeric(ner_type)
                    
                    # Add BoW similarity to each candidate's features
                    for candidate in candidates:
                        candidate['matched_words'] = matched_words.get(candidate['id'], [])
                        candidate['features']['bow_similarity'] = round(bow_similarities.get(candidate['id'], 0.0), 4)
                        candidate['features']['column_NERtype'] = ner_type_numeric  # Add the NER type from the column

                    # Re-score the candidates after computing BoW similarity
                    ranked_candidates = self.score_candidates(ne_value, candidates)

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
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        training_collection = db[self.training_collection_name]

        training_document = {
            "datasetName": dataset_name,
            "tableName": table_name,
            "idRow": row_index,
            "candidates": candidates_by_ne_column
        }

        training_collection.insert_one(training_document)

    def process_row(self, doc_id, dataset_name, table_name):
        row_start_time = datetime.now()
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.db_name]
            collection = db[self.collection_name]

            doc = collection.find_one_and_update(
                {"_id": doc_id, "status": "TODO"},
                {"$set": {"status": "DOING"}},
                return_document=True
            )

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

        except Exception as e:
            logging.error(f"Error processing row with _id {doc_id}: {str(e)}\n{traceback.format_exc()}")
            collection.update_one({'_id': doc_id}, {'$set': {'status': 'TODO'}})

    def run(self, dataset_name, table_name):
        start_time = datetime.now()
        self.update_table_trace(dataset_name, table_name, status="IN_PROGRESS", start_time=start_time)

        with mp.Pool(processes=self.max_workers) as pool:
            while True:
                client = MongoClient(self.mongo_uri)
                db = client[self.db_name]
                collection = db[self.collection_name]

                todo_docs = list(collection.find({"status": "TODO"}, {"_id": 1}).limit(self.max_workers))

                if not todo_docs:
                    end_time = datetime.now()
                    self.update_table_trace(dataset_name, table_name, status="COMPLETED", end_time=end_time, start_time=start_time)
                    break

                doc_ids = [doc['_id'] for doc in todo_docs]

                pool.starmap(self.process_row, [(doc_id, dataset_name, table_name) for doc_id in doc_ids])

                logging.info("Processed a batch of documents.")
                time.sleep(1)

        # After processing the table, update dataset trace
        self.update_dataset_trace(dataset_name)