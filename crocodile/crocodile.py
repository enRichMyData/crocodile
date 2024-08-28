import requests
import time
import os
from pymongo import MongoClient
import multiprocessing as mp
import logging
import Levenshtein
from difflib import SequenceMatcher
import traceback
from datetime import datetime

class Crocodile:
    def __init__(self, mongo_uri="mongodb://mongodb:27017/", db_name="crocodile_db", collection_name="input_data", trace_collection_name="processing_trace", max_workers=None, max_candidates=5, entity_retrieval_endpoint=None, entity_retrieval_token=None):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.trace_collection_name = trace_collection_name
        self.max_workers = max_workers or mp.cpu_count()
        self.max_candidates = max_candidates
        self.entity_retrieval_endpoint = entity_retrieval_endpoint
        self.entity_retrieval_token = entity_retrieval_token
        logging.basicConfig(filename='crocodile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def update_trace(self, dataset_name, table_name, increment=0, status=None, start_time=None, end_time=None, row_time=None):
        """Update the processing trace for a given dataset and table."""
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        trace_collection = db[self.trace_collection_name]

        update_fields = {}

        # Increment processed_rows if increment is provided
        if increment:
            update_fields["processed_rows"] = increment

        # Update status if provided
        if status:
            update_fields["status"] = status

        # Add start time if provided
        if start_time:
            update_fields["start_time"] = start_time

        # Add end time if provided
        if end_time:
            update_fields["end_time"] = end_time
            update_fields["duration"] = (end_time - start_time).total_seconds()  # Calculate total duration

        # Add row processing time if provided
        if row_time:
            update_fields["last_row_time"] = row_time

        update_query = {}

        if update_fields:
            # Apply $inc for processed_rows and $set for other fields
            if "processed_rows" in update_fields:
                update_query["$inc"] = {"processed_rows": update_fields.pop("processed_rows")}
            if update_fields:
                update_query["$set"] = update_fields

        trace_collection.update_one(
            {"dataset_name": dataset_name, "table_name": table_name},
            update_query,
            upsert=True
        )

    def fetch_candidates(self, entity_name):
        try:
            if not self.entity_retrieval_endpoint or not self.entity_retrieval_token:
                raise ValueError("Entity retrieval endpoint and token must be provided.")

            url = f"{self.entity_retrieval_endpoint}?name={entity_name}&token={self.entity_retrieval_token}"
            response = requests.get(url, headers={'accept': 'application/json'}, timeout=10)  # Added timeout for network issues
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

            candidates = response.json()
            filtered_candidates = []
            for candidate in candidates:
                filtered_candidate = {
                    'id': candidate.get('id'),
                    'name': candidate.get('name'),
                    'description': candidate.get('description'),
                    'types': candidate.get('types'),
                }
                filtered_candidates.append(filtered_candidate)
            return filtered_candidates
        
        except requests.exceptions.Timeout:
            logging.error(f"Timeout occurred while fetching candidates for {entity_name} from {self.entity_retrieval_endpoint}.")
            return []
        
        except requests.exceptions.ConnectionError:
            logging.error(f"Connection error occurred while fetching candidates for {entity_name} from {self.entity_retrieval_endpoint}.")
            return []
        
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred while fetching candidates for {entity_name}: {http_err}")
            return []
        
        except ValueError as val_err:
            logging.error(f"Configuration error: {val_err}")
            return []
        
        except Exception as e:
            logging.error(f"An unexpected error occurred while fetching candidates for {entity_name}: {str(e)}\n{traceback.format_exc()}")
            return []

    def calculate_similarity(self, a, b):
        # Handle None values by converting them to empty strings
        a = a if a is not None else ""
        b = b if b is not None else ""
        
        sequence_similarity = SequenceMatcher(None, a, b).ratio()
        edit_distance_similarity = 1 - (Levenshtein.distance(a, b) / max(len(a), len(b)))
        return (sequence_similarity + edit_distance_similarity) / 2

    def score_candidate(self, ne_value, candidate, context_values):
        # Handle None value for ne_value
        ne_value = ne_value if ne_value is not None else ""

        base_score = 1.0 if candidate['name'].lower() == ne_value.lower() else 0.0
        context_score = 0.0
        
        for context in context_values:
            context_description = candidate.get('description', '')
            context_score += self.calculate_similarity(context, context_description)
        
        # Normalize the context score
        max_context_score = len(context_values)
        if max_context_score > 0:
            context_score /= max_context_score

        total_score = (base_score + context_score) / 2  # Normalize between 0 and 1
        candidate['score'] = round(total_score, 2)  # Add the normalized score to the candidate
        return candidate

    def score_candidates(self, ne_value, candidates, context_values):
        scored_candidates = []
        for candidate in candidates:
            scored_candidate = self.score_candidate(ne_value, candidate, context_values)
            scored_candidates.append(scored_candidate)
        
        # Sort candidates based on score in descending order
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit the number of candidates stored
        return scored_candidates[:self.max_candidates]

    def link_entity(self, row, ne_columns, context_columns):
        linked_entities = {}
        for ne_column in ne_columns:
            entity_name = row.get(ne_column)
            if entity_name:
                candidates = self.fetch_candidates(entity_name)
                context_values = [str(row.get(col, "")) for col in context_columns]  # Ensure context values are strings and handle None
                ranked_candidates = self.score_candidates(entity_name, candidates, context_values)
                linked_entities[ne_column] = ranked_candidates  # Store linked results by column
        return linked_entities

    def process_row(self, doc_id, dataset_name, table_name):
        row_start_time = datetime.now()
        try:
            # Create a new MongoDB client connection in each process
            client = MongoClient(self.mongo_uri)
            db = client[self.db_name]
            collection = db[self.collection_name]

            # Find the document by ID and lock it by setting status to "DOING"
            doc = collection.find_one_and_update(
                {"_id": doc_id, "status": "TODO"},
                {"$set": {"status": "DOING"}},
                return_document=True
            )

            if not doc:
                return  # If document is not found or already being processed, skip it

            # Extract necessary information
            row = doc['data']
            ne_columns = doc['classified_columns']['NE']
            context_columns = doc.get('context_columns', [])

            # Perform entity linking
            linked_entities = self.link_entity(row, ne_columns, context_columns)

            # Update MongoDB document with linked data in `el_results` and mark as "DONE"
            collection.update_one({'_id': doc['_id']}, {'$set': {'el_results': linked_entities, 'status': 'DONE'}})

            # Update trace information
            row_end_time = datetime.now()
            row_duration = (row_end_time - row_start_time).total_seconds()
            self.update_trace(dataset_name, table_name, increment=1, row_time=row_duration)

        except Exception as e:
            logging.error(f"Error processing row with _id {doc_id}: {str(e)}\n{traceback.format_exc()}")
            # Optionally, revert the status to TODO for retry or log for manual intervention
            collection.update_one({'_id': doc_id}, {'$set': {'status': 'TODO'}})

    def run(self, dataset_name, table_name):
        start_time = datetime.now()
        self.update_trace(dataset_name, table_name, status="IN_PROGRESS", start_time=start_time)
        
        with mp.Pool(processes=self.max_workers) as pool:
            while True:
                # Create a new MongoDB client connection to fetch the tasks
                client = MongoClient(self.mongo_uri)
                db = client[self.db_name]
                collection = db[self.collection_name]

                # Fetch a batch of TODO documents to process
                todo_docs = list(collection.find({"status": "TODO"}, {"_id": 1}).limit(self.max_workers))

                if not todo_docs:
                    # Update trace to mark the table as processed
                    end_time = datetime.now()
                    self.update_trace(dataset_name, table_name, status="COMPLETED", end_time=end_time, start_time=start_time)
                    break  # No more data to process

                # Extract the _id of each document to process
                doc_ids = [doc['_id'] for doc in todo_docs]

                # Parallelize the processing of each document by its _id
                pool.starmap(self.process_row, [(doc_id, dataset_name, table_name) for doc_id in doc_ids])

                logging.info("Processed a batch of documents.")

                # Sleep briefly to avoid tight loops when no documents are available
                time.sleep(1)