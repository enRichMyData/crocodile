import asyncio
import datetime
import json
import logging
import os
import warnings
from typing import List, Optional

import pandas as pd
import strawberry
from database_utils import DatabaseManager
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter
from strawberry.file_uploads import Upload

warnings.filterwarnings("ignore", category=DeprecationWarning)

import nltk
from nltk.corpus import stopwords

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Create a set of English stopwords
STOP_WORDS = set(stopwords.words("english"))


load_dotenv()


# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Initialize Database Manager
db_manager = DatabaseManager(mongo_uri="mongodb://localhost:27020/", db_name="crocodile_db")

# ✅ Initialize Crocodile
from crocodile.crocodile import Crocodile

crocodile_instance = Crocodile(
    mongo_uri="mongodb://localhost:27020/",
    db_name="crocodile_db",
    input_collection="input_data",
    max_candidates=3,
    entity_retrieval_endpoint=os.getenv("ENTITY_RETRIEVAL_ENDPOINT"),
    entity_retrieval_token=os.getenv("ENTITY_RETRIEVAL_TOKEN"),
)

# print("Entity Retrieval Endpoint", os.getenv("ENTITY_RETRIEVAL_ENDPOINT"))
# === GraphQL Types ===


@strawberry.type
class DatasetType:
    dataset_name: str
    tables: List[str]
    status: str  # ✅ Tracks processing status


@strawberry.type
class UploadResponse:
    message: str
    dataset_name: str
    table_name: str


@strawberry.type
class DatasetConnection:
    nodes: List[DatasetType]
    pageInfo: "PageInfo"


@strawberry.type
class PageInfo:
    hasNextPage: bool
    endCursor: Optional[str]


@strawberry.type
class SemanticAnnotation:
    entity_id: str
    entity_type: Optional[str]
    entity_name: Optional[str]
    confidence_score: float
    source_column: str
    row_index: int


@strawberry.type
class Metadata:
    dataset_name: str
    table_name: str
    classified_columns: List[str]
    total_rows: int
    created_at: str
    additional_info: Optional[str]


@strawberry.type
class TableResultWithAnnotations:
    row_id: int
    data: List[str]
    semantic_annotations: List[SemanticAnnotation]


async def process_entity_linking(dataset_name: str, table_name: str):
    """
    Modified process_entity_linking function
    """
    try:
        logging.info(f"🚀 Starting entity linking for {dataset_name}...")
        db_manager.update_status(dataset_name, "processing")
        db_manager.set_todo_status(dataset_name, table_name)

        if not crocodile_instance:
            raise ValueError("❌ crocodile_instance is None!")

        # Run the crocodile process
        try:
            await asyncio.to_thread(crocodile_instance.run)

            # Update status based on actual completion
            final_status = db_manager.check_and_update_dataset_status(dataset_name)
            logging.info(f"Final status after processing: {final_status}")

        except Exception as process_error:
            logging.error(f"❌ Processing error: {process_error}")
            db_manager.update_status(dataset_name, "failed")
            raise

    except Exception as e:
        logging.error(f"❌ Entity linking failed for {dataset_name}: {e}")
        db_manager.update_status(dataset_name, "failed")
        raise


# async def process_entity_linking(dataset_name: str, table_name: str):
#     """
#     Runs entity linking in the background and updates the dataset status in MongoDB.
#     """
#     try:
#         logging.info(f"🚀 Starting entity linking for {dataset_name}...")
#         db_manager.update_status(dataset_name, "processing")
#         db_manager.set_todo_status(dataset_name, table_name)

#         logging.info(f"📌 Calling crocodile_instance.run({dataset_name}, {table_name})")
#         if not crocodile_instance:
#             raise ValueError("❌ crocodile_instance is None!")

#         await asyncio.to_thread(crocodile_instance.run)


#         db_manager.update_status(dataset_name, "completed")
#         logging.info(f"✅ Status updated to completed for {dataset_name}")
#     except Exception as e:
#         logging.error(f"❌ Entity linking failed for {dataset_name}: {e}")
#         db_manager.update_status(dataset_name, "failed")


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def upload_dataset(
        self,
        csv_file: Upload,
        metadata_file: Upload,
    ) -> UploadResponse:
        try:
            # Read file contents
            csv_content = await csv_file.read()
            metadata_content = await metadata_file.read()

            # Extract dataset name from filename
            dataset_name = (
                csv_file.filename.split(".")[0] if csv_file.filename else "unknown_dataset"
            )
            table_name = "MainTable"

            logging.info(f"📤 Received dataset: {dataset_name}")

            # Process CSV data
            df = pd.read_csv(pd.io.common.BytesIO(csv_content))
            data = df.to_dict(orient="records")

            # Process metadata
            try:
                metadata = json.loads(metadata_content.decode("utf-8"))
            except json.JSONDecodeError:
                return UploadResponse(
                    message="Failed to parse metadata JSON file", datasetName="", tableName=""
                )
            rows = []
            # Insert data into MongoDB
            for index, record in enumerate(data):
                record.update(metadata)
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "table_name": table_name,
                        "row_id": index,
                        "data": list(record.values()),
                        "classified_columns": metadata["classified_columns"],
                        "context_columns": [str(i) for i in range(len(record.values()))],
                        "status": "TODO",
                    }
                )

            db_manager.input_data_collection.insert_many(rows)
            print("Rows Inserted!")

            db_manager.table_collection.insert_one(
                {
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "header": list(df.columns),  # Store the header (column names)
                    "total_rows": len(df),
                    "processed_rows": 0,
                    "status": "PENDING",
                    "classified_columns": metadata["classified_columns"],
                }
            )

            db_manager.dataset_collection.update_one(
                {"dataset_name": dataset_name},
                {
                    "$setOnInsert": {
                        "total_tables": 1,
                        "processed_tables": 0,
                        "total_rows": len(df),
                        "processed_rows": 0,
                        "status": "PENDING",
                    }
                },
                upsert=True,
            )

            # Trigger background processing
            asyncio.create_task(process_entity_linking(dataset_name, table_name))

            return UploadResponse(
                message="Dataset uploaded successfully and processing started",
                datasetName=dataset_name,
                tableName=table_name,
            )

        except Exception as e:
            error_msg = f"Upload failed: {str(e)}"
            logging.error(f"❌ {error_msg}")
            return UploadResponse(message=error_msg, datasetName="", tableName="")


@strawberry.type
class Query:
    """GraphQL Queries (Read Operations)"""

    @strawberry.field
    def get_datasets(
        self, dataset_name: Optional[str] = None, cursor: Optional[str] = None, page_size: int = 10
    ) -> DatasetConnection:
        """Fetch datasets with optional dataset name filter"""
        try:
            result = db_manager.get_datasets(cursor, page_size, dataset_name)
            return DatasetConnection(
                nodes=[
                    DatasetType(
                        dataset_name=d["dataset_name"], tables=d["tables"], status=d["status"]
                    )
                    for d in result["datasets"]
                ],
                pageInfo=PageInfo(
                    hasNextPage=result["next_cursor"] is not None, endCursor=result["next_cursor"]
                ),
            )
        except Exception as e:
            logging.error(f"Error in get_datasets query: {e}")
            return DatasetConnection(
                nodes=[], pageInfo=PageInfo(hasNextPage=False, endCursor=None)
            )

    @strawberry.field
    def get_dataset_status(self, dataset_name: str) -> str:
        """Fetch the processing status of a specific dataset"""
        return db_manager.get_status(dataset_name)

    @strawberry.field
    def get_dataset_metadata(self, dataset_name: str) -> Optional[Metadata]:
        """Fetch metadata for a specific dataset"""
        try:
            metadata = db_manager.get_dataset_metadata(dataset_name)
            if not metadata:
                logging.warning(f"No metadata found for dataset: {dataset_name}")
                return None

            # Ensure classified_columns is a list
            classified_columns = metadata.get("classified_columns", [])
            if not isinstance(classified_columns, list):
                classified_columns = [str(classified_columns)]

            # Convert datetime to string if present
            created_at = ""
            if "start_time" in metadata and metadata["start_time"]:
                if isinstance(metadata["start_time"], datetime.datetime):
                    created_at = metadata["start_time"].isoformat()
                else:
                    created_at = str(metadata["start_time"])

            return Metadata(
                dataset_name=metadata["dataset_name"],
                table_name=metadata.get("table_name", ""),
                classified_columns=classified_columns,
                total_rows=metadata.get("total_rows", 0),
                created_at=created_at,
                additional_info=metadata.get("additional_info", ""),
            )
        except Exception as e:
            logging.error(f"Error fetching metadata: {e}")
            return None

    @strawberry.field
    def get_table_with_annotations(
        self, dataset_name: str, table_name: str, cursor: Optional[str] = None, page_size: int = 10
    ) -> List[TableResultWithAnnotations]:
        """Fetch table data with semantic annotations"""
        try:
            result = db_manager.get_table_results_with_annotations(
                dataset_name, table_name, cursor, page_size
            )

            table_results = []
            for row in result.get("results", []):
                annotations = []
                for annotation in row.get("semantic_annotations", []):
                    annotations.append(
                        SemanticAnnotation(
                            entity_id=annotation.get("entity_id", ""),
                            entity_type=annotation.get("entity_type"),
                            entity_name=annotation.get("entity_name"),
                            confidence_score=annotation.get("confidence_score", 0.0),
                            source_column=annotation.get("source_column", ""),
                            row_index=annotation.get("row_index", 0),
                        )
                    )

                table_results.append(
                    TableResultWithAnnotations(
                        row_id=row.get("row_id", 0),
                        data=row.get("data", []),
                        semantic_annotations=annotations,
                    )
                )

            return table_results
        except Exception as e:
            logging.error(f"Error fetching table with annotations: {e}")
            return []


# ✅ Create GraphQL Schema
schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(
    schema,
    allow_queries_via_get=True,
    graphiql=True,
    # allow_multipart_requests=True,  # Enable multipart requests for file uploads
)

# ✅ Initialize FastAPI & Attach GraphQL
app = FastAPI()

# Add support for multipart/form-data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(graphql_app, prefix="/graphql")


# Add a separate FastAPI endpoint for file uploads
@app.post("/upload-dataset/")
async def upload_dataset_endpoint(
    csv_file: UploadFile = File(...),
    metadata_file: UploadFile = File(...),
):
    try:
        # Read file contents
        csv_content = await csv_file.read()
        metadata_content = await metadata_file.read()

        dataset_name = csv_file.filename.split(".")[0] if csv_file.filename else "unknown_dataset"
        table_name = "MainTable"

        logging.info(f"📤 Received dataset: {dataset_name}")

        # Process CSV data
        df = pd.read_csv(pd.io.common.BytesIO(csv_content))
        data = df.to_dict(orient="records")

        # Process metadata
        try:
            metadata = json.loads(metadata_content.decode("utf-8"))
        except json.JSONDecodeError:
            return {
                "success": False,
                "message": "Failed to parse metadata JSON file",
                "datasetName": "",
                "tableName": "",
            }

        # Clear any existing data for this dataset
        db_manager.input_data_collection.delete_many({"dataset_name": dataset_name})
        db_manager.table_collection.delete_many({"dataset_name": dataset_name})
        db_manager.dataset_collection.delete_many({"dataset_name": dataset_name})

        # Prepare rows with proper structure and TODO status
        rows = []
        for index, record in enumerate(data):
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "row_id": index,
                    "data": list(record.values()),
                    "classified_columns": metadata["classified_columns"],
                    "context_columns": [str(i) for i in range(len(record.values()))],
                    "status": "TODO",
                }
            )

        # Insert data into MongoDB
        db_manager.input_data_collection.insert_many(rows)

        # Store the header and metadata in table_trace_collection
        current_time = datetime.datetime.utcnow()
        db_manager.table_collection.insert_one(
            {
                "dataset_name": dataset_name,
                "table_name": table_name,
                "header": list(df.columns),
                "total_rows": len(df),
                "processed_rows": 0,
                "status": "PENDING",
                "classified_columns": metadata["classified_columns"],
                "start_time": current_time,
                "completion_percentage": 0,
                "rows_per_second": 0,
                "status_counts": {"TODO": len(df), "DOING": 0, "DONE": 0},
            }
        )

        # Initialize dataset-level trace
        db_manager.dataset_collection.insert_one(
            {
                "dataset_name": dataset_name,
                "total_tables": 1,
                "processed_tables": 0,
                "total_rows": len(df),
                "processed_rows": 0,
                "status": "PENDING",
                "start_time": current_time,
                "completion_percentage": 0,
                "rows_per_second": 0,
                "status_counts": {"TODO": len(df), "DOING": 0, "DONE": 0},
            }
        )

        # Initialize status in dataset_status collection
        db_manager.insert_dataset_status(dataset_name, "uploaded")

        # Trigger background processing
        asyncio.create_task(process_entity_linking(dataset_name, table_name))

        return {
            "success": True,
            "message": "Dataset uploaded successfully and processing started",
            "datasetName": dataset_name,
            "tableName": table_name,
        }

    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        logging.error(f"❌ {error_msg}")
        return {"success": False, "message": error_msg, "datasetName": "", "tableName": ""}


# ✅ Start Uvicorn Automatically
if __name__ == "__main__":
    import uvicorn

    logging.info("🚀 Starting GraphQL FastAPI server on http://0.0.0.0:8006/graphql ...")
    uvicorn.run("crocodile_upload_file:app", host="0.0.0.0", port=8006, reload=True)
