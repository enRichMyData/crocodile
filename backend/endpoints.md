# Crocodile API Endpoints Documentation

This document provides information about the available endpoints in the Crocodile API and how to use them. All endpoints require authentication via a JWT token passed in the request header.

## Authentication

All endpoints in the Crocodile API require a valid JWT token for authentication. The token must be included in the request header as follows:

```
Authorization: Bearer <your_jwt_token>
```

The JWT token contains information about the user and is verified on each request. Tokens have an expiration time after which they become invalid and require regeneration.

A token generation script (`scripts/generate_token.py`) is provided for development and testing purposes. See the script for usage instructions.

## Datasets Management

### List Datasets

```
GET /datasets
```

Retrieves all datasets with bi-directional pagination.

**Query Parameters:**
- `limit` (int, default=10): Maximum number of datasets to return
- `next_cursor` (string, optional): Cursor for forward pagination
- `prev_cursor` (string, optional): Cursor for backward pagination

**Example:**
```bash
curl -H "Authorization: Bearer {your_token}" "http://localhost:8000/datasets?limit=5"
```

**Response:**
```json
{
  "data": [
    {
      "_id": "60ab1234c1d8a45678901234",
      "dataset_name": "my_dataset",
      "created_at": "2023-05-15T10:30:00.000Z",
      "total_tables": 3,
      "total_rows": 150,
      "user_id": "user@example.com"
    },
    ...
  ],
  "pagination": {
    "next_cursor": "60ab1234c1d8a45678901239",
    "prev_cursor": null
  }
}
```

### Create Dataset

```
POST /datasets
```

Creates a new dataset.

**Request Body:**
```json
{
  "dataset_name": "my_new_dataset"
}
```

**Example:**
```bash
curl -X POST \
  -H "Authorization: Bearer {your_token}" \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "my_new_dataset"}' \
  "http://localhost:8000/datasets"
```

**Response:**
```json
{
  "message": "Dataset created successfully",
  "dataset": {
    "_id": "60ab1234c1d8a45678901234",
    "dataset_name": "my_new_dataset",
    "created_at": "2023-07-20T14:15:22.123Z",
    "total_tables": 0,
    "total_rows": 0,
    "user_id": "user@example.com"
  }
}
```

### Delete Dataset

```
DELETE /datasets/{dataset_name}
```

Deletes a dataset and all its tables.

**Example:**
```bash
curl -X DELETE \
  -H "Authorization: Bearer {your_token}" \
  "http://localhost:8000/datasets/my_dataset"
```

## Tables Management

### List Tables

```
GET /datasets/{dataset_name}/tables
```

Lists all tables in a specific dataset with pagination.

**Query Parameters:**
- `limit` (int, default=10): Maximum number of tables to return
- `next_cursor` (string, optional): Cursor for forward pagination
- `prev_cursor` (string, optional): Cursor for backward pagination

**Example:**
```bash
curl -H "Authorization: Bearer {your_token}" \
  "http://localhost:8000/datasets/my_dataset/tables?limit=5"
```

**Response:**
```json
{
  "dataset": "my_dataset",
  "data": [
    {
      "_id": "60ab5678c1d8a45678901234",
      "table_name": "movies",
      "created_at": "2023-05-15T11:30:00.000Z",
      "completed_at": "2023-05-15T11:35:22.000Z",
      "total_rows": 50,
      "header": ["id", "title", "year", "director"],
      "user_id": "user@example.com",
      "dataset_name": "my_dataset"
    },
    ...
  ],
  "pagination": {
    "next_cursor": "60ab5678c1d8a45678901239",
    "prev_cursor": null
  }
}
```

### Get Table Data

```
GET /datasets/{dataset_name}/tables/{table_name}
```

Retrieves table data with rows, columns and linked entities with pagination.

**Query Parameters:**
- `limit` (int, default=10): Maximum number of rows to return
- `next_cursor` (string, optional): Cursor for forward pagination
- `prev_cursor` (string, optional): Cursor for backward pagination

**Example:**
```bash
curl -H "Authorization: Bearer {your_token}" \
  "http://localhost:8000/datasets/my_dataset/tables/movies?limit=5"
```

**Response:**
```json
{
  "data": {
    "datasetName": "my_dataset",
    "tableName": "movies",
    "status": "DONE",
    "header": ["id", "title", "year", "director"],
    "rows": [
      {
        "idRow": 0,
        "data": ["1", "The Godfather", "1972", "Francis Ford Coppola"],
        "linked_entities": [
          {
            "idColumn": 1,
            "candidates": [
              {
                "id": "Q47703",
                "name": "The Godfather",
                "description": "1972 film by Francis Ford Coppola",
                "types": [
                  {"id": "Q11424", "name": "film"}
                ],
                "match": true,
                "score": 0.95
              },
              ...
            ]
          }
        ]
      },
      ...
    ]
  },
  "pagination": {
    "next_cursor": "60ab9012c1d8a45678901234",
    "prev_cursor": null
  }
}
```

### Add Table from JSON

```
POST /datasets/{datasetName}/tables/json
```

Adds a new table to the dataset from JSON data and triggers Crocodile processing.

**Request Body:**
```json
{
  "table_name": "directors",
  "header": ["id", "name", "birth_year", "nationality"],
  "total_rows": 3,
  "classified_columns": {
    "1": {"type": "NE", "subtype": "PERSON"}
  },
  "data": [
    {"id": "1", "name": "Christopher Nolan", "birth_year": "1970", "nationality": "British"},
    {"id": "2", "name": "Quentin Tarantino", "birth_year": "1963", "nationality": "American"},
    {"id": "3", "name": "Martin Scorsese", "birth_year": "1942", "nationality": "American"}
  ]
}
```

**Example:**
```bash
curl -X POST \
  -H "Authorization: Bearer {your_token}" \
  -H "Content-Type: application/json" \
  -d '{...}' \
  "http://localhost:8000/datasets/my_dataset/tables/json"
```

**Response:**
```json
{
  "message": "Table added successfully.",
  "tableName": "directors",
  "datasetName": "my_dataset",
  "userId": "user@example.com"
}
```

### Add Table from CSV

```
POST /datasets/{datasetName}/tables/csv
```

Adds a new table to the dataset from a CSV file and triggers Crocodile processing.

**Form Data:**
- `table_name` (string): Name of the table
- `file` (file): CSV file to upload
- `column_classification` (string, optional): JSON string of column classifications

**Example:**
```bash
curl -X POST \
  -H "Authorization: Bearer {your_token}" \
  -F "table_name=actors" \
  -F "file=@actors.csv" \
  -F "column_classification={\"1\":{\"type\":\"NE\",\"subtype\":\"PERSON\"}}" \
  "http://localhost:8000/datasets/my_dataset/tables/csv"
```

**Response:**
```json
{
  "message": "CSV table added successfully.",
  "tableName": "actors",
  "datasetName": "my_dataset",
  "userId": "user@example.com"
}
```

### Delete Table

```
DELETE /datasets/{dataset_name}/tables/{table_name}
```

Deletes a table from a dataset.

**Example:**
```bash
curl -X DELETE \
  -H "Authorization: Bearer {your_token}" \
  "http://localhost:8000/datasets/my_dataset/tables/movies"
```

## Annotations Management

### Update Annotation

```
PUT /datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}
```

Updates the annotation for a specific cell by marking a candidate as matching or adding a new candidate.

**Request Body:**
```json
{
  "entity_id": "Q11696",
  "match": true,
  "score": 1.0,
  "notes": "Manually verified",
  "candidate_info": null
}
```

**Example:**
```bash
curl -X PUT \
  -H "Authorization: Bearer {your_token}" \
  -H "Content-Type: application/json" \
  -d '{...}' \
  "http://localhost:8000/datasets/my_dataset/tables/movies/rows/0/columns/1"
```

**Response:**
```json
{
  "message": "Annotation updated successfully",
  "dataset_name": "my_dataset",
  "table_name": "movies",
  "row_id": 0,
  "column_id": 1,
  "entity": {
    "id": "Q11696",
    "name": "The Godfather",
    "description": "1972 film directed by Francis Ford Coppola",
    "types": [{"id": "Q11424", "name": "film"}],
    "match": true,
    "score": 1.0,
    "notes": "Manually verified"
  },
  "manually_annotated": true
}
```

### Delete Candidate

```
DELETE /datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}/candidates/{entity_id}
```

Deletes a specific candidate from the entity linking results for a cell.

**Example:**
```bash
curl -X DELETE \
  -H "Authorization: Bearer {your_token}" \
  "http://localhost:8000/datasets/my_dataset/tables/movies/rows/0/columns/1/candidates/Q11696"
```

**Response:**
```json
{
  "message": "Candidate deleted successfully",
  "dataset_name": "my_dataset",
  "table_name": "movies",
  "row_id": 0,
  "column_id": 1,
  "entity_id": "Q11696",
  "remaining_candidates": 2
}
```

## Note on Pagination

All endpoints that return lists support bi-directional pagination using the cursor pattern. To navigate:

1. For the first page, make a request without cursor parameters
2. For the next page, use the `next_cursor` from the previous response
3. For the previous page, use the `prev_cursor` from the current response

Example of moving through pages:
```bash
# First page
curl -H "Authorization: Bearer {token}" "http://localhost:8000/datasets?limit=5"

# Next page (using next_cursor from previous response)
curl -H "Authorization: Bearer {token}" "http://localhost:8000/datasets?limit=5&next_cursor=60ab1234c1d8a45678901239"

# Previous page (using prev_cursor)
curl -H "Authorization: Bearer {token}" "http://localhost:8000/datasets?limit=5&prev_cursor=60ab1234c1d8a45678901234"
```
