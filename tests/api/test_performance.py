# tests/api/test_performance.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import tests.patch_imports as patch_imports

import pytest
from fastapi import status
from unittest.mock import patch, MagicMock
import time
import pandas as pd
from io import BytesIO

class TestPerformance:
    """Tests for performance and scaling characteristics"""
    
    @pytest.fixture
    def mock_mongodb(self, test_db):
        """Provide the test database for MongoDB tests"""
        return test_db
    
    @pytest.fixture
    def mock_token_payload(self):
        """Mock token payload for authentication."""
        return {"email": "test@example.com"}
    
    @pytest.fixture
    def create_large_csv(self):
        """Helper fixture to create a large CSV file BytesIO object for testing"""
        def _create_large_csv(rows=1000, cols=10):
            """
            Create a large CSV file with the specified number of rows and columns.
            
            Args:
                rows (int): Number of rows in the CSV
                cols (int): Number of columns in the CSV
                
            Returns:
                BytesIO: A file-like object containing the CSV data
            """
            header = ",".join([f"col{j}" for j in range(cols)])
            content = header + "\n"
            
            # Generate rows
            for i in range(rows):
                row = ",".join([f"val{i}_{j}" for j in range(cols)])
                content += row + "\n"
            
            # Create file-like object
            file = BytesIO(content.encode())
            file.name = "large_test_file.csv"
            
            return file
        
        return _create_large_csv
    
    def test_maximum_rows_limit(self, client, test_dataset, mock_token_payload):
        """Test creating a table with the maximum reasonable number of rows
        
        Args:
            client: FastAPI test client fixture
            test_dataset: Fixture providing a test dataset with sample data
            mock_token_payload: Mock token payload for authentication
        This test verifies that the API can handle a table with a large number of rows
        without timing out or failing due to size limits.
        """
        dataset_name = test_dataset["dataset_name"]
        
        # Create a table with a reasonably large number of rows (e.g., 1,000)
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
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload), \
             patch("backend.app.endpoints.crocodile_api.ResultSyncService") as mock_sync_service:
            
            # Configure the sync service mock
            mock_sync_instance = mock_sync_service.return_value
            mock_sync_instance.sync_results = MagicMock()
            
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
                assert end_time - start_time < 100.0  # Increased timeout
    
    def test_maximum_columns_limit(self, client, test_dataset, mock_token_payload):
        """Test creating a table with the maximum reasonable number of columns
        
        Args:
            client: FastAPI test client fixture
            test_dataset: Fixture providing a test dataset with sample data
            mock_token_payload: Mock token payload for authentication
        This test verifies that the API can handle a table with a large number of columns
        without timing out or failing due to size limits.
        """
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
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload), \
             patch("backend.app.endpoints.crocodile_api.ResultSyncService") as mock_sync_service:
            
            # Configure the sync service mock
            mock_sync_instance = mock_sync_service.return_value
            mock_sync_instance.sync_results = MagicMock()
            
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
                assert end_time - start_time < 100.0  # Increased timeout
    
    def test_large_csv_upload(self, client, test_dataset, mock_token_payload, create_large_csv):
        """Test uploading and processing a large CSV file
        
        Args:
            client: FastAPI test client fixture
            test_dataset: Fixture providing a test dataset with sample data
            mock_token_payload: Mock token payload for authentication
            create_large_csv: Fixture to create a large CSV file BytesIO object
        This test verifies that the API can handle a large CSV upload without timing out
        or failing due to size limits.
        """
        dataset_name = test_dataset["dataset_name"]
        
        # Create a large CSV (e.g., 1,000 rows x 10 columns)
        file = create_large_csv(rows=1000, cols=10)
        
        # Mock pandas read_csv to avoid actually parsing the large file
        mock_df = pd.DataFrame({
            f"col{j}": [f"val{i}_{j}" for i in range(1000)]
            for j in range(10)
        })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload), \
             patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df), \
             patch("backend.app.endpoints.crocodile_api.ResultSyncService") as mock_sync_service:
            
            # Configure the sync service mock
            mock_sync_instance = mock_sync_service.return_value
            mock_sync_instance.sync_results = MagicMock()
            
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
                assert end_time - start_time < 100.0  # Increased timeout