import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from pymongo import MongoClient
from typing import List, Optional
from bson import ObjectId

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27020/")
db = client["crocodile_db"]

# Define Dataset Type
@strawberry.type
class DatasetType:
    id: str
    dataset_name: str
    status: str
    total_rows: int
    completion_percentage: float

# Define Table Type
@strawberry.type
class TableType:
    id: str
    table_name: str
    dataset_name: str
    status: str
    total_rows: int
    completion_percentage: float

# Define Table Data Type (Input Data)
@strawberry.type
class TableRowType:
    id: str
    dataset_name: str
    table_name: str
    row_id: int
    row_data: List[str]

# Define Pagination Responses
@strawberry.type
class PaginatedDatasets:
    datasets: List[DatasetType]
    next_cursor: Optional[str]
    previous_cursor: Optional[str]
    has_next_page: bool

@strawberry.type
class PaginatedTables:
    tables: List[TableType]
    next_cursor: Optional[str]
    previous_cursor: Optional[str]
    has_next_page: bool

@strawberry.type
class PaginatedTableData:
    rows: List[TableRowType]
    next_cursor: Optional[str]
    previous_cursor: Optional[str]
    has_next_page: bool

@strawberry.type
class Query:

    # 1 Get All Datasets (Paginated)
    @strawberry.field
    def get_datasets(self, cursor: Optional[str] = None, page_size: int = 5, direction: str = "next") -> PaginatedDatasets:
        query = {}

        if cursor:
            query["_id"] = {"$gt" if direction == "next" else "$lt": ObjectId(cursor)}

        datasets = list(db.dataset_trace.find(query).sort("_id", 1 if direction == "next" else -1).limit(page_size))
        
        if direction == "prev":
            datasets.reverse()  # Reverse for proper ordering

        next_cursor = str(datasets[-1]["_id"]) if datasets else None
        previous_cursor = str(datasets[0]["_id"]) if datasets else None
        has_next_page = bool(db.dataset_trace.find_one({"_id": {"$gt": ObjectId(next_cursor)}})) if next_cursor else False

        return PaginatedDatasets(
            datasets=[
                DatasetType(
                    id=str(dataset["_id"]),
                    dataset_name=dataset["dataset_name"],
                    status=dataset["status"],
                    total_rows=dataset["total_rows"],
                    completion_percentage=dataset["completion_percentage"]
                ) for dataset in datasets
            ],
            next_cursor=next_cursor,
            previous_cursor=previous_cursor,
            has_next_page=has_next_page
        )

    # 2 Get Specific Dataset
    @strawberry.field
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetType]:
        dataset = db.dataset_trace.find_one({"dataset_name": dataset_name})
        if not dataset:
            return None

        return DatasetType(
            id=str(dataset["_id"]),
            dataset_name=dataset["dataset_name"],
            status=dataset["status"],
            total_rows=dataset["total_rows"],
            completion_percentage=dataset["completion_percentage"]
        )    

    # 3 Get Tables in a Specific Dataset (Paginated)
    @strawberry.field
    def get_tables_in_dataset(self, dataset_name: str, cursor: Optional[str] = None, page_size: int = 5, direction: str = "next") -> PaginatedTables:
        query = {"dataset_name": dataset_name}

        if cursor:
            query["_id"] = {"$gt" if direction == "next" else "$lt": ObjectId(cursor)}

        tables = list(db.table_trace.find(query).sort("_id", 1 if direction == "next" else -1).limit(page_size))
        
        if direction == "prev":
            tables.reverse()

        next_cursor = str(tables[-1]["_id"]) if tables else None
        previous_cursor = str(tables[0]["_id"]) if tables else None
        has_next_page = bool(db.table_trace.find_one({"_id": {"$gt": ObjectId(next_cursor)}})) if next_cursor else False

        return PaginatedTables(
            tables=[
                TableType(
                    id=str(table["_id"]),
                    table_name=table["table_name"],
                    dataset_name=table["dataset_name"],
                    status=table["status"],
                    total_rows=table["total_rows"],
                    completion_percentage=table["completion_percentage"]
                ) for table in tables
            ],
            next_cursor=next_cursor,
            previous_cursor=previous_cursor,
            has_next_page=has_next_page
        )

    # 4 Get Specific Table Info
    @strawberry.field
    def get_table_info(self, table_name: str) -> Optional[TableType]:
        table = db.table_trace.find_one({"table_name": table_name})
        if not table:
            return None

        return TableType(
            id=str(table["_id"]),
            table_name=table["table_name"],
            dataset_name=table["dataset_name"],
            status=table["status"],
            total_rows=table["total_rows"],
            completion_percentage=table["completion_percentage"]
        )    

    # 5 Get Table Data Rows (Paginated)
    @strawberry.field
    def get_table_data(self, dataset_name: str, table_name: str, cursor: Optional[str] = None, page_size: int = 5, direction: str = "next") -> PaginatedTableData:
        query = {"dataset_name": dataset_name, "table_name": table_name}

        if cursor:
            query["_id"] = {"$gt" if direction == "next" else "$lt": ObjectId(cursor)}

        rows = list(db.input_data.find(query).sort("_id", 1 if direction == "next" else -1).limit(page_size))
        
        if direction == "prev":
            rows.reverse()

        next_cursor = str(rows[-1]["_id"]) if rows else None
        previous_cursor = str(rows[0]["_id"]) if rows else None
        has_next_page = bool(db.input_data.find_one({"_id": {"$gt": ObjectId(next_cursor)}})) if next_cursor else False

        return PaginatedTableData(
            rows=[
                TableRowType(
                    id=str(row["_id"]),
                    dataset_name=row["dataset_name"],
                    table_name=row["table_name"],
                    row_id=row["row_id"],
                    row_data=row.get("data", [])
                ) for row in rows
            ],
            next_cursor=next_cursor,
            previous_cursor=previous_cursor,
            has_next_page=has_next_page
        )

# Create GraphQL Schema
schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)

# FastAPI Setup
app = FastAPI()
app.include_router(graphql_app, prefix="/graphql")

# Run Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
