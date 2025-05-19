# tests/api/test_performance.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import tests.patch_imports as patch_imports

import pytest
from fastapi import status
from unittest.mock import patch, MagicMock
import json
import time
from datetime import datetime
from bson import ObjectId
import pandas as pd
from io import BytesIO

class TestPerformance:
    """Tests that focus on performance and boundary conditions"""
    
    @pytest.fixture
    def mock_token_payload(self):
        """Mock token payload for authentication."""
        return {"email": "test@example.com"}
    
    @pytest.fixture
    def setup_large_dataset(self, mock_mongodb):
        """Set up a large dataset with many tables and rows for performance testing"""
        user_id = "test@example.com"
        dataset_name = "performance_test_dataset"
        num_tables = 20
        
        # Create dataset
        dataset_id = mock_mongodb.datasets.insert_one({
            "user_id": user_id,
            "dataset_name": dataset_name,
            "created_at": datetime.now(),
            "total_tables": num_tables,
            "total_rows": num_tables * 10  # Each table has 10 rows
        }).inserted_id
        
        # Create tables
        for i in range(num_tables):
            table_id = mock_mongodb.tables.insert_one({
                "user_id": user_id,
                "dataset_name": dataset_name,
                "table_name": f"performance_table_{i}",
                "header": ["col1", "col2", "col3"],
                "total_rows": 10,
                "created_at": datetime.now()
            }).inserted_id
            
            # Create rows
            for j in range(10):
                mock_mongodb.input_data.insert_one({
                    "user_id": user_id,
                    "dataset_name": dataset_name,
                    "table_name": f"performance_table_{i}",
                    "row_id": j,
                    "data": [f"val{j}_1", f"val{j}_2", f"val{j}_3"],
                    "el_results": {
                        "0": [
                            {
                                "id": f"entity_{i}_{j}_1",
                                "name": f"Entity {i} {j} Primary",
                                "description": "Test entity",
                                "score": 0.9,
                                "match": True
                            },
                            {
                                "id": f"entity_{i}_{j}_2",
                                "name": f"Entity {i} {j} Secondary",
                                "description": "Another test entity",
                                "score": 0.5,
                                "match": False
                            }
                        ]
                    }
                })
        
        return {"dataset_name": dataset_name, "dataset_id": str(dataset_id)}
    
    # =================== Response Time Tests ===================
    
    def test_dataset_list_performance(self, client, mock_token_payload, setup_large_dataset):
        """Test the response time of listing datasets with various pagination parameters"""
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Test default pagination
            start_time = time.time()
            response = client.get("/datasets")
            end_time = time.time()
            
            assert response.status_code == status.HTTP_200_OK
            # Response time should be under a reasonable threshold (e.g., 500ms)
            assert end_time - start_time < 0.5
            
            # Test with limit parameter
            start_time = time.time()
            response = client.get("/datasets?limit=5")
            end_time = time.time()
            
            assert response.status_code == status.HTTP_200_OK
            assert len(response.json()["data"]) <= 5
            assert end_time - start_time < 0.5
            
            # Test with pagination
            if response.json()["pagination"]["next_cursor"]:
                next_cursor = response.json()["pagination"]["next_cursor"]
                
                start_time = time.time()
                response = client.get(f"/datasets?limit=5&next_cursor={next_cursor}")
                end_time = time.time()
                
                assert response.status_code == status.HTTP_200_OK
                assert end_time - start_time < 0.5
    
    def test_table_list_performance(self, client, mock_token_payload, setup_large_dataset):
        """Test the response time of listing tables for a dataset"""
        dataset_name = setup_large_dataset["dataset_name"]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Test with different limits
            for limit in [5, 10, 20]:
                start_time = time.time()
                response = client.get(f"/datasets/{dataset_name}/tables?limit={limit}")
                end_time = time.time()
                
                assert response.status_code == status.HTTP_200_OK
                assert len(response.json()["data"]) <= limit
                # Response time should be under a reasonable threshold
                assert end_time - start_time < 0.5
    
    def test_table_data_performance(self, client, mock_token_payload, setup_large_dataset, mock_mongodb):
        """Test the response time of getting table data with various parameters"""
        dataset_name = setup_large_dataset["dataset_name"]
        
        # Get a table name
        table = mock_mongodb.tables.find_one({"dataset_name": dataset_name})
        if not table:
            pytest.skip("No tables found in the dataset")
        
        table_name = table["table_name"]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Test with different limits
            for limit in [5, 10, 20]:
                start_time = time.time()
                response = client.get(f"/datasets/{dataset_name}/tables/{table_name}?limit={limit}")
                end_time = time.time()
                
                assert response.status_code == status.HTTP_200_OK
                assert len(response.json()["data"]["rows"]) <= limit
                # Response time should be under a reasonable threshold
                assert end_time - start_time < 0.5
            
            # Test with pagination
            response = client.get(f"/datasets/{dataset_name}/tables/{table_name}?limit=5")
            if response.json()["pagination"]["next_cursor"]:
                next_cursor = response.json()["pagination"]["next_cursor"]
                
                start_time = time.time()
                response = client.get(f"/datasets/{dataset_name}/tables/{table_name}?limit=5&next_cursor={next_cursor}")
                end_time = time.time()
                
                assert response.status_code == status.HTTP_200_OK
                assert end_time - start_time < 0.5
    
    # =================== Boundary Tests ===================
    
    def test_maximum_rows_limit(self, client, test_dataset, mock_token_payload):
        """Test creating a table with the maximum reasonable number of rows"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create a table with a reasonably large number of rows (e.g., 1,000)
        # Note: In a real environment, you might want to test with even larger datasets
        num_rows = 1000
        
        # Create data
        header = ["col1", "col2", "col3"]
        data = []
        
        for i in range(num_rows):
            data.append({
                "col1": f"val{i}_1",
                "col2": f"val{i}_2",
                "col3": f"val{i}_3"
            })
        
        table_data = {
            "table_name": "large_row_table",
            "header": header,
            "total_rows": num_rows,
            "data": data
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            start_time = time.time()
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            end_time = time.time()
            
            # The API should either handle this or reject it with an appropriate error
            assert response.status_code in [
                status.HTTP_201_CREATED,  # Handled successfully
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE  # Rejected due to size limits
            ]
            
            # If successful, check that the response time is reasonable
            if response.status_code == status.HTTP_201_CREATED:
                assert end_time - start_time < 5.0  # Allow more time for large data
    
    def test_maximum_columns_limit(self, client, test_dataset, mock_token_payload):
        """Test creating a table with the maximum reasonable number of columns"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create a table with a reasonably large number of columns (e.g., 100)
        num_cols = 100
        
        # Create header and data
        header = [f"col{i}" for i in range(num_cols)]
        data = []
        
        # Just add a few rows to keep the total size reasonable
        for i in range(5):
            row = {}
            for j in range(num_cols):
                row[f"col{j}"] = f"val{i}_{j}"
            data.append(row)
        
        table_data = {
            "table_name": "large_column_table",
            "header": header,
            "total_rows": 5,
            "data": data
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            start_time = time.time()
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            end_time = time.time()
            
            # The API should either handle this or reject it with an appropriate error
            assert response.status_code in [
                status.HTTP_201_CREATED,  # Handled successfully
                status.HTTP_400_BAD_REQUEST,  # Rejected due to validation
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE  # Rejected due to size limits
            ]
            
            # If successful, check that the response time is reasonable
            if response.status_code == status.HTTP_201_CREATED:
                assert end_time - start_time < 5.0  # Allow more time for large data
    
    def test_maximum_cell_size_limit(self, client, test_dataset, mock_token_payload):
        """Test creating a table with a very large cell value"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create a table with a very large cell value (e.g., 100KB)
        large_value = "a" * 100000  # 100KB
        
        table_data = {
            "table_name": "large_cell_table",
            "header": ["normal_col", "large_col"],
            "total_rows": 1,
            "data": [{
                "normal_col": "normal value",
                "large_col": large_value
            }]
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            
            # The API should either handle this or reject it with an appropriate error
            assert response.status_code in [
                status.HTTP_201_CREATED,  # Handled successfully
                status.HTTP_400_BAD_REQUEST,  # Rejected due to validation
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE  # Rejected due to size limits
            ]
    
    def test_repeated_operations(self, client, test_dataset, mock_token_payload):
        """Test repeated operations on the same resources"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create a table
        table_data = {
            "table_name": "repeated_ops_table",
            "header": ["col1", "col2"],
            "total_rows": 1,
            "data": [{"col1": "val1", "col2": "val2"}]
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Create the table
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            assert response.status_code == status.HTTP_201_CREATED
            
            # Get the table data repeatedly
            for i in range(10):
                response = client.get(f"/datasets/{dataset_name}/tables/repeated_ops_table")
                assert response.status_code == status.HTTP_200_OK
            
            # Try to delete and recreate multiple times
            for i in range(3):
                # Delete
                response = client.delete(f"/datasets/{dataset_name}/tables/repeated_ops_table")
                assert response.status_code in [status.HTTP_204_NO_CONTENT, status.HTTP_404_NOT_FOUND]
                
                # Recreate
                table_data["table_name"] = "repeated_ops_table"
                response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
                assert response.status_code == status.HTTP_201_CREATED
    
    # =================== Large CSV Tests ===================
    
    @pytest.fixture
    def create_large_csv(self):
        """Create a large CSV file for testing"""
        def _create_large_csv(rows=1000, cols=10):
            # Create header
            header = ",".join([f"col{i}" for i in range(cols)])
            
            # Create data rows
            data_rows = []
            for i in range(rows):
                row = ",".join([f"val{i}_{j}" for j in range(cols)])
                data_rows.append(row)
            
            # Combine into CSV content
            csv_content = header + "\n" + "\n".join(data_rows)
            
            # Create file
            file = BytesIO(csv_content.encode())
            file.name = "large_file.csv"
            
            return file
        
        return _create_large_csv
    
    def test_large_csv_upload(self, client, test_dataset, mock_token_payload, create_large_csv):
        """Test uploading and processing a large CSV file"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create a large CSV (e.g., 1,000 rows x 10 columns)
        file = create_large_csv(rows=1000, cols=10)
        
        # Mock pandas read_csv to avoid actually parsing the large file
        mock_df = pd.DataFrame({
            f"col{j}": [f"val{i}_{j}" for i in range(1000)]
            for j in range(10)
        })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload), \
             patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            
            start_time = time.time()
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=large_csv_test",
                files={"file": (file.name, file, "text/csv")}
            )
            end_time = time.time()
            
            # The API should either handle this or reject it with an appropriate error
            assert response.status_code in [
                status.HTTP_201_CREATED,  # Handled successfully
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE  # Rejected due to size limits
            ]
            
            # If successful, check that the response time is reasonable
            if response.status_code == status.HTTP_201_CREATED:
                assert end_time - start_time < 10.0  # Allow more time for large upload

    # =================== Streaming Endpoint Tests ===================
    
    def test_table_status_streaming(self, client, test_dataset, mock_token_payload, mock_mongodb):
        """Test the table status streaming endpoint"""
        dataset_name = test_dataset["dataset_name"]
        table_name = "stream_test_table"
        
        # Create a table with pending documents
        mock_mongodb.tables.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "header": ["col1", "col2"],
            "total_rows": 5,
            "created_at": datetime.now()
        })
        
        # Add some rows with varying status
        statuses = ["DONE", "DONE", "TODO", "TODO", "DOING"]
        for i, status in enumerate(statuses):
            mock_mongodb.input_data.insert_one({
                "user_id": "test@example.com",
                "dataset_name": dataset_name,
                "table_name": table_name,
                "row_id": i,
                "data": [f"val{i}_1", f"val{i}_2"],
                "status": status,
                "ml_status": status
            })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # The streaming endpoint requires special handling in tests
            # We'll just verify it returns the correct content type and status code
            response = client.get(f"/datasets/{dataset_name}/tables/{table_name}/status")
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "text/event-stream"
            
            # Check that the response contains data
            assert "data:" in response.content.decode()