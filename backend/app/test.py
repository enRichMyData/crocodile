from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

# Connect to Elasticsearch
es = Elasticsearch(
    hosts=["http://elastic:9200"],
    request_timeout=60
)

# Define the index name
INDEX_NAME = "table_rows"

# Define the index mapping
INDEX_MAPPING = {
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
                    "value": {"type": "text"},
                    "confidence_score": {"type": "float"},
                    "types": {"type": "keyword"}
                }
            }
        }
    }
}

# Check if the index exists, and create it if it doesn't
try:
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
except Exception as e:
    print(f"Error creating index '{INDEX_NAME}': {e}")

# Verify connection
if es.ping():
    print("Connected to Elasticsearch")
else:
    print("Unable to ping Elasticsearch")