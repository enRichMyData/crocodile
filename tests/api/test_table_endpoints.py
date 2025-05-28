# tests/api/test_table_endpoints.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import tests.patch_imports as patch_imports

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import the app and dependencies
from backend.app.main import app
from backend.app.dependencies import get_db, get_crocodile_db, verify_token


class TestTableEndpoints:
    """Tests for table-related endpoints"""
    
    @pytest.fixture
    def mock_mongodb(self, test_db):
        """Provide the test database for MongoDB tests"""
        return test_db

    @pytest.fixture
    def mock_token_payload(self):
        """Mock token payload for authentication."""
        return {"email": "test@example.com"}

    @pytest.fixture
    def mock_crocodile_db(self, test_db):
        """Provide the test database as if it were the crocodile DB"""
        return test_db

    @pytest.fixture
    def test_dataset(self, client, mock_mongodb, mock_token_payload):
        """Create a test dataset for table tests."""
        dataset_data = {"dataset_name": "test_dataset", "description": "Test dataset"}
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post("/datasets", json=dataset_data)
            assert response.status_code == status.HTTP_201_CREATED
            return response.json()["dataset"]
    
    @pytest.fixture
    def test_table_with_data(self, client, mock_mongodb, test_dataset, mock_token_payload):
        """Create a test table with sample data"""
        dataset_name = test_dataset["dataset_name"]
        table_name = "test_table_with_data"
        
        # Create the table
        table_data = {
            "table_name": table_name,
            "header": ["col1", "col2", "col3"],
            "total_rows": 2,
            "data": [
                {"col1": "value1", "col2": "value2", "col3": "value3"},
                {"col1": "value4", "col2": "value5", "col3": "value6"}
            ]
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(
                f"/datasets/{dataset_name}/tables/json", 
                json=table_data
            )
            assert response.status_code == status.HTTP_201_CREATED
            
            # Add input data records
            for i in range(2):
                mock_mongodb.input_data.insert_one({
                    "user_id": "test@example.com",
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "row_id": i,
                    "data": [f"value{i*3+1}", f"value{i*3+2}", f"value{i*3+3}"],
                    "el_results": {
                        "0": [
                            {
                                "id": f"entity_{i}_1",
                                "name": f"Entity {i} Primary",
                                "description": "Test entity",
                                "score": 0.9,
                                "match": True
                            },
                            {
                                "id": f"entity_{i}_2",
                                "name": f"Entity {i} Secondary",
                                "description": "Another test entity",
                                "score": 0.5,
                                "match": False
                            }
                        ]
                    }
                })
            
            return {"dataset_name": dataset_name, "table_name": table_name}
        
    def test_add_table_json(self, client, mock_mongodb, test_dataset, mock_token_payload):
        """Test adding a table with JSON data"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]  # Extract the dataset name
        
        # Create a minimal test payload to pass validation
        table_data = {
            "table_name": "test_json_table",
            "header": ["col1", "col2", "col3"],
            "total_rows": 2,
            "data": [
                {"col1": "value1", "col2": "value2", "col3": "value3"},
                {"col1": "value4", "col2": "value5", "col3": "value6"}
            ]
        }
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(
                f"/datasets/{dataset_name}/tables/json", 
                json=table_data
            )
            
            # Assert
            assert response.status_code == status.HTTP_201_CREATED
            assert response.json()["message"] == "Table added successfully."
            assert response.json()["tableName"] == "test_json_table"
            assert response.json()["datasetName"] == dataset_name

            # Verify that the table exists in the database
            table = mock_mongodb.tables.find_one({
                "dataset_name": dataset_name,
                "table_name": "test_json_table"
            })
            assert table is not None
    
    def test_add_table_json_empty_data(self, client, mock_mongodb, test_dataset, mock_token_payload):
        """Test adding a table with JSON data but empty data array"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]
        table_data = {
            "table_name": "test_empty_table",
            "header": ["col1", "col2"],
            "total_rows": 0,
            "data": []
        }
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            assert response.status_code == status.HTTP_201_CREATED
            assert response.json()["message"] == "Table added successfully."

            # Verify that the table exists in the database
            table = mock_mongodb.tables.find_one({
                "dataset_name": dataset_name,
                "table_name": "test_empty_table"
            })
            assert table is not None

    def test_add_table_json_missing_field(self, client, mock_mongodb, test_dataset, mock_token_payload):
        """Test adding a table with JSON data but missing a required field"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]
        table_data = {
            "table_name": "test_missing_field_table",
            "header": ["col1", "col2"],
            "data": [{"col1": "val1", "col2": "val2"}]
        }
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            assert "detail" in response.json()

    def test_delete_table(self, client, mock_mongodb, test_dataset, mock_token_payload):
        """Test deleting a table"""
        # Arrange - Create a table to delete
        dataset_name = test_dataset["dataset_name"]  # Extract the dataset name
        table_name = "table_to_delete"
        table = {
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "header": ["col1", "col2"],
            "total_rows": 2,
            "created_at": datetime.now()
        }
        mock_mongodb.tables.insert_one(table)
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Act
            response = client.delete(f"/datasets/{dataset_name}/tables/{table_name}")
            
            # Assert
            assert response.status_code == status.HTTP_204_NO_CONTENT
            
            # Verify it was deleted
            table = mock_mongodb.tables.find_one({
                "dataset_name": dataset_name,
                "table_name": table_name
            })
            assert table is None

    def test_get_tables(self, client, mock_mongodb, test_dataset, mock_token_payload):
        """Test retrieving tables for a dataset"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]
        
        # Create some tables
        for i in range(3):
            mock_mongodb.tables.insert_one({
                "user_id": "test@example.com",
                "dataset_name": dataset_name,
                "table_name": f"table_{i}",
                "header": ["col1", "col2"],
                "total_rows": i + 1,
                "created_at": datetime.now()
            })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Act
            response = client.get(f"/datasets/{dataset_name}/tables")
            
            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "data" in data
            assert len(data["data"]) == 3
            
            # Check table names
            table_names = [t["table_name"] for t in data["data"]]
            for i in range(3):
                assert f"table_{i}" in table_names

    def test_get_tables_empty(self, client, mock_mongodb, test_dataset, mock_token_payload):
        """Test retrieving tables when none exist"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]
        
        # Ensure no tables exist for the dataset
        mock_mongodb.tables.delete_many({"dataset_name": dataset_name})
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Act
            response = client.get(f"/datasets/{dataset_name}/tables")
            
            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "data" in data
            assert len(data["data"]) == 0
            assert data["pagination"]["next_cursor"] is None
            assert data["pagination"]["prev_cursor"] is None

    def test_get_table(self, client, mock_mongodb, test_table_with_data, mock_token_payload, mock_crocodile_db):
        """Test retrieving a specific table with its data"""
        # Arrange
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload),\
             patch("backend.app.dependencies.get_crocodile_db", return_value=mock_crocodile_db):
            # Act
            response = client.get(f"/datasets/{dataset_name}/tables/{table_name}")
            
            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "data" in data
            assert data["data"]["datasetName"] == dataset_name
            assert data["data"]["tableName"] == table_name
            
            # Check for unique row IDs
            row_ids = [row["idRow"] for row in data["data"]["rows"]]
            unique_row_ids = set(row_ids)
            assert len(unique_row_ids) == 2  # We expect 2 distinct row IDs
            
            # Check that linked entities are present in at least one row
            assert any(len(row.get("linked_entities", [])) > 0 for row in data["data"]["rows"])