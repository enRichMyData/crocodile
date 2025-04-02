# How to Use Crocodile API Endpoints

Here's a guide on how to use each endpoint in the API:

## Dataset Operations

### Create a New Dataset

```
POST /datasets
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/datasets" \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "movies_2023"}'
```

**Response:**
```json
{
  "message": "Dataset created successfully",
  "dataset": {
    "_id": "6576e81a9c82a5b2c7d3e4f5",
    "dataset_name": "movies_2023",
    "created_at": "2025-04-02T14:30:15.123456",
    "total_tables": 0,
    "total_rows": 0
  }
}
```

### List All Datasets

```
GET /datasets?limit=10&cursor=<optional_cursor>
```

**Example Request:**
```bash
curl -X GET "http://localhost:8000/datasets?limit=5"
```

**Response:**
```json
{
  "data": [
    {
      "_id": "6576e81a9c82a5b2c7d3e4f5",
      "dataset_name": "movies_2023",
      "created_at": "2025-04-02T14:30:15.123456",
      "total_tables": 2,
      "total_rows": 150
    },
    {...}
  ],
  "pagination": {
    "next_cursor": "6576e9b29c82a5b2c7d3e4f9"
  }
}
```

### Delete a Dataset

```
DELETE /datasets/{dataset_name}
```

**Example Request:**
```bash
curl -X DELETE "http://localhost:8000/datasets/movies_2023"
```

## Table Operations

### Add a JSON Table

```
POST /dataset/{datasetName}/table/json
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/dataset/movies_2023/table/json" \
  -H "Content-Type: application/json" \
  -d '{
    "table_name": "directors",
    "header": ["name", "nationality", "birth_year"],
    "total_rows": 3,
    "data": [
      {"name": "Christopher Nolan", "nationality": "British", "birth_year": "1970"},
      {"name": "Greta Gerwig", "nationality": "American", "birth_year": "1983"},
      {"name": "Bong Joon-ho", "nationality": "Korean", "birth_year": "1969"}
    ]
  }'
```

**Response:**
```json
{
  "message": "Table added successfully.",
  "tableName": "directors",
  "datasetName": "movies_2023"
}
```

### Add a CSV Table

```
POST /dataset/{datasetName}/table/csv
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/dataset/movies_2023/table/csv" \
  -F "table_name=movies" \
  -F "file=@movies.csv" \
  -F "column_classification={\"NE\":{\"0\":\"PERSON\",\"1\":\"PERSON\"},\"LIT\":{\"2\":\"DATE\"},\"IGNORED\":[]}"
```

**Response:**
```json
{
  "message": "CSV table added successfully.",
  "tableName": "movies",
  "datasetName": "movies_2023"
}
```

### List Tables in a Dataset

```
GET /datasets/{dataset_name}/tables?limit=10&cursor=<optional_cursor>
```

**Example Request:**
```bash
curl -X GET "http://localhost:8000/datasets/movies_2023/tables?limit=5"
```

**Response:**
```json
{
  "dataset": "movies_2023",
  "data": [
    {
      "_id": "6576ec219c82a5b2c7d3e4f8",
      "dataset_name": "movies_2023",
      "table_name": "directors",
      "header": ["name", "nationality", "birth_year"],
      "total_rows": 3,
      "created_at": "2025-04-02T15:15:20.123456",
      "status": "DONE",
      "classified_columns": {
        "NE": {"0": "PERSON"},
        "LIT": {"1": "ORGANIZATION", "2": "DATE"},
        "IGNORED": []
      }
    },
    {...}
  ],
  "pagination": {
    "next_cursor": "6576ed459c82a5b2c7d3e4fc"
  }
}
```

### Get Table Data

```
GET /datasets/{dataset_name}/tables/{table_name}?limit=10&cursor=<optional_cursor>
```

**Example Request:**
```bash
curl -X GET "http://localhost:8000/datasets/movies_2023/tables/directors?limit=2"
```

**Response:**
```json
{
  "data": {
    "datasetName": "movies_2023",
    "tableName": "directors",
    "status": "DONE",
    "header": ["name", "nationality", "birth_year"],
    "rows": [
      {
        "idRow": 0,
        "data": ["Christopher Nolan", "British", "1970"],
        "linked_entities": [
          {
            "idColumn": 0,
            "candidates": [
              {"id": "Q41421", "label": "Christopher Nolan", "score": 0.95},
              {"id": "Q56462", "label": "Christopher Nolan (author)", "score": 0.42}
            ]
          }
        ]
      },
      {...}
    ]
  },
  "pagination": {
    "next_cursor": "6576ef129c82a5b2c7d3e501"
  }
}
```

### Delete a Table

```
DELETE /datasets/{dataset_name}/tables/{table_name}
```

**Example Request:**
```bash
curl -X DELETE "http://localhost:8000/datasets/movies_2023/tables/directors"
```

## Key Notes

1. When adding tables, if you don't provide `classified_columns`, the system automatically classifies columns using the `ColumnClassifier`.

2. Column classification has three categories:
   - `NE`: Named Entities (PERSON, ORGANIZATION, LOCATION, OTHER)
   - `LIT`: Literal values (other data types)
   - `IGNORED`: Columns not to process

3. All list endpoints support pagination with `limit` and `cursor` parameters.

4. For `POST /dataset/{datasetName}/table/csv`, you can optionally provide pre-classified columns as a JSON string.