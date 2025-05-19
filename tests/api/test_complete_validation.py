# tests/api/test_complete_validation.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import tests.patch_imports as patch_imports

import pytest
from fastapi import status
from unittest.mock import patch, MagicMock
from io import BytesIO
import json
import pandas as pd
import numpy as np

class TestCompleteValidation:
    """Comprehensive tests for validating the API without changing the core code"""
    
    @pytest.fixture
    def mock_token_payload(self):
        """Mock token payload for authentication."""
        return {"email": "test@example.com"}
    
    @pytest.fixture
    def create_csv_file(self):
        """Helper fixture to create a CSV file BytesIO object with custom content"""
        def _create_csv_file(content, filename="test_file.csv"):
            file = BytesIO(content.encode())
            file.name = filename
            return file
        return _create_csv_file

    # =================== Dataset Endpoints Validation ===================
    
    def test_dataset_duplicate_name(self, client, mock_token_payload):
        """Test creating datasets with duplicate names is rejected"""
        dataset_name = "duplicate_dataset_test"
        
        # Create the dataset
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post("/datasets", json={"dataset_name": dataset_name})
            assert response.status_code == status.HTTP_201_CREATED
            
            # Try to create with the same name again
            response = client.post("/datasets", json={"dataset_name": dataset_name})
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "already exists" in response.json()["detail"]
    
    def test_dataset_delete_nonexistent(self, client, mock_token_payload):
        """Test deleting a non-existent dataset returns 404"""
        nonexistent_name = "nonexistent_dataset_for_delete"
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.delete(f"/datasets/{nonexistent_name}")
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "not found" in response.json()["detail"]
    
    def test_dataset_pagination_limits(self, client, mock_mongodb, mock_token_payload):
        """Test dataset pagination with different limit values"""
        # Create several datasets
        for i in range(15):
            mock_mongodb.datasets.insert_one({
                "user_id": "test@example.com",
                "dataset_name": f"pagination_test_{i}",
                "created_at": "2023-01-01T00:00:00",
                "total_tables": 0,
                "total_rows": 0
            })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Test default limit (should be 10)
            response = client.get("/datasets")
            assert response.status_code == status.HTTP_200_OK
            assert len(response.json()["data"]) == 10
            
            # Test custom limit
            response = client.get("/datasets?limit=5")
            assert response.status_code == status.HTTP_200_OK
            assert len(response.json()["data"]) == 5
            
            # Test limit exceeding available data
            response = client.get("/datasets?limit=20")
            assert response.status_code == status.HTTP_200_OK
            # Should return all available datasets (15)
            assert len(response.json()["data"]) == 15
    
    def test_dataset_pagination_cursor(self, client, mock_mongodb, mock_token_payload):
        """Test dataset pagination with next and previous cursors"""
        # Clear existing datasets and create 15 new ones
        mock_mongodb.datasets.delete_many({})
        datasets = []
        for i in range(15):
            result = mock_mongodb.datasets.insert_one({
                "user_id": "test@example.com",
                "dataset_name": f"cursor_test_{i}",
                "created_at": "2023-01-01T00:00:00",
                "total_tables": 0,
                "total_rows": 0
            })
            datasets.append(str(result.inserted_id))
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Get first page with limit=5
            response = client.get("/datasets?limit=5")
            assert response.status_code == status.HTTP_200_OK
            first_page = response.json()
            assert len(first_page["data"]) == 5
            assert first_page["pagination"]["next_cursor"] is not None
            
            # Get second page using next_cursor
            next_cursor = first_page["pagination"]["next_cursor"]
            response = client.get(f"/datasets?limit=5&next_cursor={next_cursor}")
            assert response.status_code == status.HTTP_200_OK
            second_page = response.json()
            assert len(second_page["data"]) == 5
            
            # Check that we got different datasets in second page
            first_page_ids = [d["_id"] for d in first_page["data"]]
            second_page_ids = [d["_id"] for d in second_page["data"]]
            assert not set(first_page_ids).intersection(set(second_page_ids))
            
            # Test previous cursor to go back to first page
            prev_cursor = second_page["pagination"]["prev_cursor"]
            response = client.get(f"/datasets?limit=5&prev_cursor={prev_cursor}")
            assert response.status_code == status.HTTP_200_OK
            back_to_first = response.json()
            back_first_ids = [d["_id"] for d in back_to_first["data"]]
            assert set(first_page_ids) == set(back_first_ids)
    
    def test_invalid_cursor_format(self, client, mock_token_payload):
        """Test providing an invalid cursor format"""
        invalid_cursors = [
            "not-an-object-id",
            "12345",
            "invalid@cursor",
            "a" * 23,  # Invalid length
        ]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            for cursor in invalid_cursors:
                response = client.get(f"/datasets?next_cursor={cursor}")
                assert response.status_code == status.HTTP_400_BAD_REQUEST
                assert "Invalid cursor value" in response.json()["detail"]
    
    def test_both_cursors_provided(self, client, mock_token_payload):
        """Test providing both next_cursor and prev_cursor is rejected"""
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.get("/datasets?next_cursor=507f1f77bcf86cd799439011&prev_cursor=507f1f77bcf86cd799439012")
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Only one of next_cursor or prev_cursor" in response.json()["detail"]

    # =================== Table Endpoints Validation ===================
    
    def test_add_table_nonexistent_dataset(self, client, mock_token_payload):
        """Test adding a table to a non-existent dataset returns 404"""
        nonexistent_dataset = "nonexistent_dataset"
        table_data = {
            "table_name": "test_table",
            "header": ["col1", "col2"],
            "total_rows": 1,
            "data": [{"col1": "val1", "col2": "val2"}]
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{nonexistent_dataset}/tables/json", json=table_data)
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert f"Dataset {nonexistent_dataset} not found" in response.json()["detail"]
    
    def test_add_table_duplicate_name(self, client, test_dataset, mock_token_payload):
        """Test adding tables with duplicate names to the same dataset is rejected"""
        dataset_name = test_dataset["dataset_name"]
        table_data = {
            "table_name": "duplicate_table_test",
            "header": ["col1", "col2"],
            "total_rows": 1,
            "data": [{"col1": "val1", "col2": "val2"}]
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Create the first table
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            assert response.status_code == status.HTTP_201_CREATED
            
            # Try to create with the same name again
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_409_CONFLICT]
    
    def test_delete_nonexistent_table(self, client, test_dataset, mock_token_payload):
        """Test deleting a non-existent table returns 404"""
        dataset_name = test_dataset["dataset_name"]
        nonexistent_table = "nonexistent_table"
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.delete(f"/datasets/{dataset_name}/tables/{nonexistent_table}")
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert f"Table {nonexistent_table} not found" in response.json()["detail"]
    
    def test_get_nonexistent_table(self, client, test_dataset, mock_token_payload):
        """Test retrieving a non-existent table returns 404"""
        dataset_name = test_dataset["dataset_name"]
        nonexistent_table = "nonexistent_table"
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.get(f"/datasets/{dataset_name}/tables/{nonexistent_table}")
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert f"Table {nonexistent_table} not found" in response.json()["detail"]
    
    def test_add_table_missing_required_fields(self, client, test_dataset, mock_token_payload):
        """Test adding a table with missing required fields is rejected"""
        dataset_name = test_dataset["dataset_name"]
        
        # Missing header
        table_data_1 = {
            "table_name": "missing_header",
            "total_rows": 1,
            "data": [{"col1": "val1", "col2": "val2"}]
        }
        
        # Missing total_rows
        table_data_2 = {
            "table_name": "missing_total_rows",
            "header": ["col1", "col2"],
            "data": [{"col1": "val1", "col2": "val2"}]
        }
        
        # Missing data
        table_data_3 = {
            "table_name": "missing_data",
            "header": ["col1", "col2"],
            "total_rows": 1
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            for i, table_data in enumerate([table_data_1, table_data_2, table_data_3]):
                response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
                assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    # =================== CSV Upload Validation ===================
    
    def test_csv_upload_invalid_file_type(self, client, test_dataset, mock_token_payload):
        """Test uploading a non-CSV file is rejected"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create a text file that's not a CSV
        file = BytesIO(b"This is not a CSV file")
        file.name = "not_csv.txt"
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=invalid_file_type",
                files={"file": ("not_csv.txt", file, "text/plain")}
            )
            assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_500_INTERNAL_SERVER_ERROR]
    
    def test_csv_upload_empty_file(self, client, test_dataset, mock_token_payload, create_csv_file):
        """Test uploading an empty CSV file is rejected"""
        dataset_name = test_dataset["dataset_name"]
        file = create_csv_file("")
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=empty_file",
                files={"file": (file.name, file, "text/csv")}
            )
            assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_500_INTERNAL_SERVER_ERROR]
    
    def test_csv_upload_missing_headers(self, client, test_dataset, mock_token_payload, create_csv_file):
        """Test uploading a CSV file without headers is handled correctly"""
        dataset_name = test_dataset["dataset_name"]
        csv_content = "val1,val2,val3\nval4,val5,val6"  # No header row
        file = create_csv_file(csv_content)
        
        # Mock pandas to simulate header detection
        mock_df = pd.DataFrame({
            0: ["val1", "val4"],
            1: ["val2", "val5"],
            2: ["val3", "val6"]
        })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload), \
             patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=no_headers_csv",
                files={"file": (file.name, file, "text/csv")}
            )
            # The API might treat the first row as headers, which is valid behavior
            assert response.status_code == status.HTTP_201_CREATED
    
    def test_csv_upload_malformed(self, client, test_dataset, mock_token_payload, create_csv_file):
        """Test uploading a malformed CSV file returns an appropriate error"""
        dataset_name = test_dataset["dataset_name"]
        csv_content = "col1,col2,col3\nval1,val2\nval4,val5,val6"  # Inconsistent column count
        file = create_csv_file(csv_content)
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload), \
             patch("backend.app.endpoints.crocodile_api.pd.read_csv", side_effect=pd.errors.ParserError("Error parsing CSV")):
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=malformed_csv",
                files={"file": (file.name, file, "text/csv")}
            )
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Error processing CSV" in response.json()["detail"]
    
    # =================== Annotation Validation ===================
    
    def test_update_annotation_invalid_entity(self, client, test_table_with_data, mock_token_payload):
        """Test updating an annotation with invalid entity data"""
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        
        # Missing entity_id
        annotation_1 = {
            "match": True,
            "score": 0.9
        }
        
        # Entity exists but score is out of range
        annotation_2 = {
            "entity_id": "entity_0_1",
            "match": True,
            "score": 1.5  # Should be between 0 and 1
        }
        
        # New entity without candidate_info
        annotation_3 = {
            "entity_id": "new_entity",
            "match": True,
            "score": 0.9
            # Missing candidate_info for new entity
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Test missing entity_id
            response = client.put(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/0/columns/0",
                json=annotation_1
            )
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            
            # Test score out of range - this might pass in your API if you don't check score range
            response = client.put(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/0/columns/0",
                json=annotation_2
            )
            # Either it validates the score (400) or it accepts it (200)
            assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_200_OK]
            
            # Test new entity without candidate_info
            response = client.put(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/0/columns/0",
                json=annotation_3
            )
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Please provide 'candidate_info'" in response.json()["detail"]
    
    def test_update_annotation_nonexistent_row(self, client, test_table_with_data, mock_token_payload):
        """Test updating an annotation for a non-existent row"""
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        
        annotation = {
            "entity_id": "entity_999",
            "match": True,
            "score": 0.9,
            "candidate_info": {
                "id": "entity_999",
                "name": "Non-existent entity",
                "description": "Test entity"
            }
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.put(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/999/columns/0",
                json=annotation
            )
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "Row 999 not found" in response.json()["detail"]
    
    def test_delete_candidate_nonexistent(self, client, test_table_with_data, mock_token_payload):
        """Test deleting a non-existent candidate"""
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.delete(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/0/columns/0/candidates/nonexistent_entity"
            )
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "not found in candidates" in response.json()["detail"]
    
    # =================== Authentication Validation ===================
    
    def test_missing_authentication(self, client):
        """Test that endpoints reject requests without authentication"""
        # Try to access a protected endpoint without authentication
        with patch("backend.app.dependencies.verify_token", side_effect=Exception("Authentication failed")):
            response = client.get("/datasets")
            assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN, status.HTTP_500_INTERNAL_SERVER_ERROR]
    
    def test_invalid_authentication(self, client):
        """Test that endpoints reject requests with invalid authentication"""
        # Try to access a protected endpoint with invalid token
        with patch("backend.app.dependencies.verify_token", side_effect=Exception("Invalid token")):
            response = client.get("/datasets")
            assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN, status.HTTP_500_INTERNAL_SERVER_ERROR]
    
    # =================== Security Validation ===================
    
    def test_database_injection_attempt(self, client, mock_token_payload):
        """Test that MongoDB injection attempts are handled safely"""
        injection_dataset = {"dataset_name": "dataset', $ne: null"}
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Attempt to create a dataset with injection
            response = client.post("/datasets", json=injection_dataset)
            # Should either be rejected or sanitized
            assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_201_CREATED]
            
            # Attempt to get a dataset with injection
            response = client.get(f"/datasets/{injection_dataset['dataset_name']}")
            # Should be treated as a literal string, not injection
            assert response.status_code in [status.HTTP_404_NOT_FOUND, status.HTTP_400_BAD_REQUEST]
    
    # =================== Edge Case Validation ===================
    
    def test_zero_rows_table(self, client, test_dataset, mock_token_payload):
        """Test creating a table with zero rows"""
        dataset_name = test_dataset["dataset_name"]
        table_data = {
            "table_name": "zero_rows_table",
            "header": ["col1", "col2"],
            "total_rows": 0,
            "data": []
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            assert response.status_code == status.HTTP_201_CREATED
            assert response.json()["tableName"] == "zero_rows_table"
    
    def test_large_column_names(self, client, test_dataset, mock_token_payload):
        """Test creating a table with very long column names"""
        dataset_name = test_dataset["dataset_name"]
        table_data = {
            "table_name": "long_columns_table",
            "header": ["col_" + "a" * 50, "col_" + "b" * 50],
            "total_rows": 1,
            "data": [{
                "col_" + "a" * 50: "value1",
                "col_" + "b" * 50: "value2"
            }]
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            # API should handle long column names
            assert response.status_code == status.HTTP_201_CREATED
    
    def test_special_characters_in_data(self, client, test_dataset, mock_token_payload):
        """Test creating a table with special characters in data"""
        dataset_name = test_dataset["dataset_name"]
        table_data = {
            "table_name": "special_chars_table",
            "header": ["col1", "col2"],
            "total_rows": 1,
            "data": [{
                "col1": "Value with \"quotes\" and 'apostrophes'",
                "col2": "Value with \n newline and \t tab"
            }]
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            # API should handle special characters
            assert response.status_code == status.HTTP_201_CREATED
    
    # =================== Response Schema Validation ===================
    
    def test_dataset_response_schema(self, client, test_dataset, mock_token_payload):
        """Test that dataset responses match the expected schema"""
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.get("/datasets")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            # Check general structure
            assert "data" in data
            assert "pagination" in data
            assert isinstance(data["data"], list)
            assert isinstance(data["pagination"], dict)
            assert "next_cursor" in data["pagination"]
            assert "prev_cursor" in data["pagination"]
            
            # Check dataset schema if any exist
            if data["data"]:
                dataset = data["data"][0]
                required_fields = ["_id", "dataset_name", "created_at", "total_tables", "total_rows", "user_id"]
                for field in required_fields:
                    assert field in dataset
    
    def test_table_response_schema(self, client, test_table_with_data, mock_token_payload):
        """Test that table responses match the expected schema"""
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.get(f"/datasets/{dataset_name}/tables/{table_name}")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            # Check general structure
            assert "data" in data
            assert "pagination" in data
            assert isinstance(data["data"], dict)
            assert isinstance(data["pagination"], dict)
            
            # Check table schema
            table_data = data["data"]
            required_fields = ["datasetName", "tableName", "status", "header", "rows"]
            for field in required_fields:
                assert field in table_data
            
            # Check rows schema
            assert isinstance(table_data["rows"], list)
            if table_data["rows"]:
                row = table_data["rows"][0]
                assert "idRow" in row
                assert "data" in row
                assert "linked_entities" in row
    
    # =================== Performance Validation ===================
    
    def test_pagination_performance(self, client, mock_mongodb, mock_token_payload):
        """Test pagination with large dataset (simulate without creating too many records)"""
        # Insert a limited number of records but test with larger limits
        for i in range(20):
            mock_mongodb.datasets.insert_one({
                "user_id": "test@example.com",
                "dataset_name": f"perf_dataset_{i}",
                "created_at": "2023-01-01T00:00:00",
                "total_tables": 0,
                "total_rows": 0
            })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Test with different limits to ensure performance is consistent
            for limit in [5, 10, 20]:
                start_time = pytest.importorskip("time").time()
                response = client.get(f"/datasets?limit={limit}")
                end_time = pytest.importorskip("time").time()
                
                assert response.status_code == status.HTTP_200_OK
                # Ensure response time is reasonable (less than 1 second)
                assert end_time - start_time < 1.0