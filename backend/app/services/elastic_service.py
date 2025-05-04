import logging
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

logger = logging.getLogger(__name__)

class ElasticService:
    """Service for managing Elasticsearch operations."""
    
    def __init__(self, es_client: Elasticsearch):
        self.es = es_client
        self.index_name = "table_rows"
    
    def create_index(self):
        """Creates the table_rows index with appropriate mappings if it doesn't exist."""
        try:
            # Check if the index exists
            if not self.es.indices.exists(index=self.index_name):
                logger.info(f"Creating index {self.index_name} with mappings")
                
                # Define the mapping
                mapping = {
                    "mappings": {
                        "properties": {
                            "user_id": {"type": "keyword"},
                            "dataset_name": {"type": "keyword"},
                            "table_name": {"type": "keyword"},
                            "row_id": {"type": "integer"},
                            "status": {"type": "keyword"},
                            "ml_status": {"type": "keyword"},
                            "manually_annotated": {"type": "boolean"},
                            "created_at": {"type": "date"},
                            "last_updated": {"type": "date"},
                            "data": {
                                "type": "nested",
                                "properties": {
                                    "col_index": {"type": "integer"},
                                    "value": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                                    "confidence_score": {"type": "float"},
                                    "types": {"type": "keyword"}
                                }
                            }
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }
                
                # Create the index with the mapping
                self.es.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Successfully created index {self.index_name}")
                return True
            else:
                logger.info(f"Index {self.index_name} already exists")
                return False
        except Exception as e:
            logger.error(f"Error creating index {self.index_name}: {str(e)}")
            raise

    def index_document(self, document):
        """
        Index a document in Elasticsearch.
        Transforms the MongoDB document structure to the Elasticsearch schema.
        """
        try:
            # Extract top-level fields
            es_doc = {
                "user_id": document.get("user_id"),
                "dataset_name": document.get("dataset_name"),
                "table_name": document.get("table_name"),
                "row_id": document.get("row_id"),
                "status": document.get("status"),
                "ml_status": document.get("ml_status"),
                "manually_annotated": document.get("manually_annotated", False),
                "created_at": document.get("created_at"),
                "last_updated": document.get("last_updated")
            }
            
            # Transform the data array and el_results into the nested data structure
            es_doc["data"] = []
            raw_data = document.get("data", [])
            el_results = document.get("el_results", {})
            
            for col_idx, value in enumerate(raw_data):
                data_item = {
                    "col_index": col_idx,
                    "value": str(value) if value is not None else "",
                    "confidence_score": 0.0,
                    "types": []
                }
                
                # Add entity linking information if available
                candidates = el_results.get(str(col_idx), [])
                if candidates and len(candidates) > 0:
                    # Get the first (highest-ranked) candidate
                    top_candidate = candidates[0]
                    if top_candidate.get("score") is not None:
                        data_item["confidence_score"] = float(top_candidate.get("score", 0.0))
                    
                    # Extract types
                    types = []
                    for type_obj in top_candidate.get("types", []):
                        if "name" in type_obj:
                            types.append(type_obj["name"])
                    data_item["types"] = types
                
                es_doc["data"].append(data_item)
            
            # Index the document
            doc_id = f"{document.get('user_id')}_{document.get('dataset_name')}_{document.get('table_name')}_{document.get('row_id')}"
            self.es.index(index=self.index_name, id=doc_id, body=es_doc)
            return doc_id
        
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise
