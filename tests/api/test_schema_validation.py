# tests/api/test_schema_validation.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import tests.patch_imports as patch_imports

import pytest
from fastapi import status
from unittest.mock import patch, MagicMock
import json
from datetime import datetime
from bson import ObjectId

class TestSchemaValidation:
    """Tests that focus specifically on schema validation without changing the API"""
    
    @pytest.fixture
    def mock_token_payload(self):
        """Mock token payload for authentication."""
        return {"email": "test@example.com"}
    
    def test_dataset_list_schema(self, client, mock_mongodb, mock_token_payload):
        """Test the dataset list endpoint returns data in the expected schema"""
        # Create some test datasets
        mock_mongodb.datasets.delete_many({})
        for i in range(3):
            mock_mongodb.datasets.insert_one({
                "user_id": "test@example.com",
                "dataset_name": f"schema_test_{i}",
                "created_at": datetime.now(),
                "total_tables": i,
                "total_rows": i * 10
            })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.get("/datasets")
            assert response.status_code == status.HTTP_200_OK
            
            # Validate the response schema
            response_data = response.json()
            
            # Top-level structure
            assert "data" in response_data
            assert "pagination" in response_data
            assert isinstance(response_data["data"], list)
            assert isinstance(response_data["pagination"], dict)
            
            # Pagination structure
            pagination = response_data["pagination"]
            assert "next_cursor" in pagination
            assert "prev_cursor" in pagination
            
            # Dataset item structure
            if response_data["data"]:
                dataset = response_data["data"][0]
                assert "_id" in dataset
                assert "dataset_name" in dataset
                assert "created_at" in dataset
                assert "total_tables" in dataset
                assert "total_rows" in dataset
                assert "user_id" in dataset
                
                # Data types
                assert isinstance(dataset["_id"], str)
                assert isinstance(dataset["dataset_name"], str)
                assert isinstance(dataset["created_at"], str)
                assert isinstance(dataset["total_tables"], int)
                assert isinstance(dataset["total_rows"], int)
                assert isinstance(dataset["user_id"], str)
    
    def test_dataset_create_schema(self, client, mock_token_payload):
        """Test the dataset creation endpoint returns data in the expected schema"""
        dataset_data = {"dataset_name": "create_schema_test"}
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post("/datasets", json=dataset_data)
            assert response.status_code == status.HTTP_201_CREATED
            
            # Validate the response schema
            response_data = response.json()
            
            # Top-level structure
            assert "message" in response_data
            assert "dataset" in response_data
            assert isinstance(response_data["message"], str)
            assert isinstance(response_data["dataset"], dict)
            
            # Dataset structure
            dataset = response_data["dataset"]
            assert "_id" in dataset
            assert "dataset_name" in dataset
            assert "created_at" in dataset
            assert "total_tables" in dataset
            assert "total_rows" in dataset
            assert "user_id" in dataset
            
            # Data types
            assert isinstance(dataset["_id"], str)
            assert isinstance(dataset["dataset_name"], str)
            assert isinstance(dataset["created_at"], str)
            assert isinstance(dataset["total_tables"], int)
            assert isinstance(dataset["total_rows"], int)
            assert isinstance(dataset["user_id"], str)
            
            # Values
            assert dataset["dataset_name"] == "create_schema_test"
            assert dataset["total_tables"] == 0
            assert dataset["total_rows"] == 0
    
    def test_tables_list_schema(self, client, test_dataset, mock_token_payload):
        """Test the tables list endpoint returns data in the expected schema"""
        dataset_name = test_dataset["dataset_name"]
        
        # Add a few tables to the test dataset
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            for i in range(2):
                table_data = {
                    "table_name": f"schema_table_{i}",
                    "header": ["col1", "col2"],
                    "total_rows": 1,
                    "data": [{"col1": f"val1_{i}", "col2": f"val2_{i}"}]
                }
                client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            
            # Now get the tables list
            response = client.get(f"/datasets/{dataset_name}/tables")
            assert response.status_code == status.HTTP_200_OK
            
            # Validate the response schema
            response_data = response.json()
            
            # Top-level structure
            assert "dataset" in response_data
            assert "data" in response_data
            assert "pagination" in response_data
            assert isinstance(response_data["dataset"], str)
            assert isinstance(response_data["data"], list)
            assert isinstance(response_data["pagination"], dict)
            
            # Pagination structure
            pagination = response_data["pagination"]
            assert "next_cursor" in pagination
            assert "prev_cursor" in pagination
            
            # Table item structure
            if response_data["data"]:
                table = response_data["data"][0]
                assert "_id" in table
                assert "dataset_name" in table
                assert "table_name" in table
                assert "created_at" in table
                assert "user_id" in table
                
                # Optional fields depending on your implementation
                # These fields might be available depending on your schema
                if "total_rows" in table:
                    assert isinstance(table["total_rows"], int)
                if "header" in table:
                    assert isinstance(table["header"], list)
                
                # Data types
                assert isinstance(table["_id"], str)
                assert isinstance(table["dataset_name"], str)
                assert isinstance(table["table_name"], str)
                assert isinstance(table["created_at"], str)
                assert isinstance(table["user_id"], str)
    
    def test_table_data_schema(self, client, test_table_with_data, mock_token_payload):
        """Test the table data endpoint returns data in the expected schema"""
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.get(f"/datasets/{dataset_name}/tables/{table_name}")
            assert response.status_code == status.HTTP_200_OK
            
            # Validate the response schema
            response_data = response.json()
            
            # Top-level structure
            assert "data" in response_data
            assert "pagination" in response_data
            assert isinstance(response_data["data"], dict)
            assert isinstance(response_data["pagination"], dict)
            
            # Pagination structure
            pagination = response_data["pagination"]
            assert "next_cursor" in pagination
            assert "prev_cursor" in pagination
            
            # Table data structure
            table_data = response_data["data"]
            assert "datasetName" in table_data
            assert "tableName" in table_data
            assert "status" in table_data
            assert "header" in table_data
            assert "rows" in table_data
            
            assert isinstance(table_data["datasetName"], str)
            assert isinstance(table_data["tableName"], str)
            assert isinstance(table_data["status"], str)
            assert isinstance(table_data["header"], list)
            assert isinstance(table_data["rows"], list)
            
            # Row structure
            if table_data["rows"]:
                row = table_data["rows"][0]
                assert "idRow" in row
                assert "data" in row
                assert "linked_entities" in row
                
                assert isinstance(row["idRow"], int)
                assert isinstance(row["data"], list)
                assert isinstance(row["linked_entities"], list)
                
                # Linked entities structure
                if row["linked_entities"]:
                    entity = row["linked_entities"][0]
                    assert "idColumn" in entity
                    assert "candidates" in entity
                    assert isinstance(entity["idColumn"], int)
                    assert isinstance(entity["candidates"], list)
                    
                    # Candidate structure
                    if entity["candidates"]:
                        candidate = entity["candidates"][0]
                        assert "id" in candidate
                        assert "name" in candidate
                        assert "match" in candidate
                        
                        assert isinstance(candidate["id"], str)
                        assert isinstance(candidate["name"], str)
                        assert isinstance(candidate["match"], bool)
                        
                        # Score is present only for matched candidates
                        if candidate["match"]:
                            assert "score" in candidate
                            assert isinstance(candidate["score"], (int, float))
    
    def test_table_add_response_schema(self, client, test_dataset, mock_token_payload):
        """Test the table add endpoint returns data in the expected schema"""
        dataset_name = test_dataset["dataset_name"]
        table_data = {
            "table_name": "add_schema_test",
            "header": ["col1", "col2"],
            "total_rows": 1,
            "data": [{"col1": "val1", "col2": "val2"}]
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            assert response.status_code == status.HTTP_201_CREATED
            
            # Validate the response schema
            response_data = response.json()
            
            # Response structure
            assert "message" in response_data
            assert "tableName" in response_data
            assert "datasetName" in response_data
            assert "userId" in response_data
            
            assert isinstance(response_data["message"], str)
            assert isinstance(response_data["tableName"], str)
            assert isinstance(response_data["datasetName"], str)
            assert isinstance(response_data["userId"], str)
            
            # Values
            assert response_data["tableName"] == "add_schema_test"
            assert response_data["datasetName"] == dataset_name
    
    def test_annotation_update_schema(self, client, test_table_with_data, mock_token_payload):
        """Test the annotation update endpoint returns data in the expected schema"""
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        
        annotation_data = {
            "entity_id": "new_entity",
            "match": True,
            "score": 0.9,
            "notes": "Test annotation",
            "candidate_info": {
                "id": "new_entity",
                "name": "New Entity",
                "description": "Test entity",
                "types": [
                    {
                        "id": "type1",
                        "name": "Type 1"
                    }
                ]
            }
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.put(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/0/columns/0",
                json=annotation_data
            )
            assert response.status_code == status.HTTP_200_OK
            
            # Validate the response schema
            response_data = response.json()
            
            # Response structure
            assert "message" in response_data
            assert "dataset_name" in response_data
            assert "table_name" in response_data
            assert "row_id" in response_data
            assert "column_id" in response_data
            assert "entity" in response_data
            assert "manually_annotated" in response_data
            
            assert isinstance(response_data["message"], str)
            assert isinstance(response_data["dataset_name"], str)
            assert isinstance(response_data["table_name"], str)
            assert isinstance(response_data["row_id"], int)
            assert isinstance(response_data["column_id"], int)
            assert isinstance(response_data["entity"], dict)
            assert isinstance(response_data["manually_annotated"], bool)
            
            # Entity structure
            entity = response_data["entity"]
            assert "id" in entity
            assert "name" in entity
            assert "match" in entity
            assert "score" in entity
            
            assert isinstance(entity["id"], str)
            assert isinstance(entity["name"], str)
            assert isinstance(entity["match"], bool)
            assert isinstance(entity["score"], (int, float, type(None)))
            
            # Values
            assert entity["id"] == "new_entity"
            assert entity["match"] is True
            assert entity["score"] == 0.9
    
    def test_delete_candidate_schema(self, client, test_table_with_data, mock_token_payload, mock_mongodb):
        """Test the delete candidate endpoint returns data in the expected schema"""
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        
        # First get a valid entity ID from the database
        row = mock_mongodb.input_data.find_one({
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": 0
        })
        
        if row and row.get("el_results") and row["el_results"].get("0"):
            entity_id = row["el_results"]["0"][0]["id"]
            
            with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
                response = client.delete(
                    f"/datasets/{dataset_name}/tables/{table_name}/rows/0/columns/0/candidates/{entity_id}"
                )
                assert response.status_code == status.HTTP_200_OK
                
                # Validate the response schema
                response_data = response.json()
                
                # Response structure
                assert "message" in response_data
                assert "dataset_name" in response_data
                assert "table_name" in response_data
                assert "row_id" in response_data
                assert "column_id" in response_data
                assert "entity_id" in response_data
                assert "remaining_candidates" in response_data
                
                assert isinstance(response_data["message"], str)
                assert isinstance(response_data["dataset_name"], str)
                assert isinstance(response_data["table_name"], str)
                assert isinstance(response_data["row_id"], int)
                assert isinstance(response_data["column_id"], int)
                assert isinstance(response_data["entity_id"], str)
                assert isinstance(response_data["remaining_candidates"], int)