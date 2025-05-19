# tests/api/test_csv_upload.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import tests.patch_imports as patch_imports

import pytest
from fastapi import status
from io import BytesIO
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# Test the CSV upload functionality
class TestCSVUpload:
    """Tests for CSV upload functionality"""

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

    def test_add_table_csv(self, client, test_db, test_dataset, mock_background_tasks, create_csv_file):
        """Test adding a table with CSV file data"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]
        csv_content = "col1,col2,col3\nvalue1,value2,value3\nvalue4,value5,value6"
        
        # Create a file-like object with the CSV content
        file = create_csv_file(csv_content)
        
        # Mock pandas read_csv
        mock_df = pd.DataFrame({
            "col1": ["value1", "value4"],
            "col2": ["value2", "value5"],
            "col3": ["value3", "value6"]
        })
        
        with patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            # Act - Using query parameter for table_name
            def test_add_table_csv_duplicate_column_names(self, client, test_db, test_dataset, mock_background_tasks, create_csv_file):
                """Test adding a CSV file with duplicate column names"""
                # Arrange
                dataset_name = test_dataset["dataset_name"]
                csv_content = "col1,col1,col2\nvalue1,value2,value3\nvalue4,value5,value6"
                
                # Create file
                file = create_csv_file(csv_content)
                
                mock_df = pd.DataFrame({
                    "col1": ["value1", "value4"],
                    "col1.1": ["value2", "value5"],
                    "col2": ["value3", "value6"]
                })
                
                with patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
                    # Act
                    response = client.post(
                        f"/datasets/{dataset_name}/tables/csv?table_name=duplicate_columns_table",
                        files={"file": ("duplicate_columns.csv", file, "text/csv")}
                    )
                    
                    # Assert
                    assert response.status_code == status.HTTP_201_CREATED
                    
                    # Verify the table was created
                    table = test_db.tables.find_one({
                        "dataset_name": dataset_name,
                        "table_name": "duplicate_columns_table"
                    })
                    assert table is not None
                    assert table["header"] == ["col1", "col1.1", "col2"]
                    assert table["total_rows"] == 2

            def test_add_table_csv_empty_file(self, client, test_dataset, create_csv_file):
                """Test adding an empty CSV file"""
                # Arrange
                dataset_name = test_dataset["dataset_name"]
                csv_content = ""
                
                # Create file
                file = create_csv_file(csv_content)
                
                # Act
                response = client.post(
                    f"/datasets/{dataset_name}/tables/csv?table_name=empty_table",
                    files={"file": ("empty.csv", file, "text/csv")}
                )
                
                # Assert
                assert response.status_code == status.HTTP_400_BAD_REQUEST
                assert "Empty CSV file" in response.json()["detail"]
        file = create_csv_file(csv_content)
        
        # Mock pandas read_csv with different data types
        mock_df = pd.DataFrame({
            "string_col": ["text1", "text2"],
            "int_col": [10, 20],
            "float_col": [10.5, 20.5],
            "date_col": pd.to_datetime(["2023-01-01", "2023-02-01"])
        })
        
        with patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            # Act - Using query parameter for table_name
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=datatypes_table",
                files={"file": ("datatypes.csv", file, "text/csv")}
            )
            
            # Assert
            assert response.status_code == status.HTTP_201_CREATED
            assert response.json()["tableName"] == "datatypes_table"
            
            # Verify the table was created with correct headers
            table = test_db.tables.find_one({
                "dataset_name": dataset_name,
                "table_name": "datatypes_table"
            })
            assert table is not None
            assert "string_col" in table["header"]
            assert "int_col" in table["header"]
            assert "float_col" in table["header"]
            assert "date_col" in table["header"]

    def test_add_table_csv_with_nulls(self, client, test_db, test_dataset, mock_background_tasks, create_csv_file):
        """Test adding a CSV with missing/null values"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]
        csv_content = "col1,col2,col3\nvalue1,,value3\n,value5,value6"
        
        # Create file
        file = create_csv_file(csv_content)
        
        # Mock pandas read_csv with NaN values
        mock_df = pd.DataFrame({
            "col1": ["value1", np.nan],
            "col2": [np.nan, "value5"],
            "col3": ["value3", "value6"]
        })
        # Mock the replace method - we need to verify this is called
        mock_df.replace = MagicMock(return_value=mock_df)
        
        with patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            # Act - Using query parameter for table_name
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=null_values_table",
                files={"file": ("nulls.csv", file, "text/csv")}
            )
            
            # Assert
            assert response.status_code == status.HTTP_201_CREATED
            
            # Verify the NaN replacement was called (checking the API is handling NaN values)
            assert mock_df.replace.called
            # Often called with np.nan: None to convert NaN to None for JSON serialization
            call_args = mock_df.replace.call_args[0][0]
            assert np.nan in call_args or np.nan in call_args.values()

    def test_add_table_csv_column_classification(self, client, test_db, test_dataset, mock_background_tasks, create_csv_file):
        """Test adding a CSV with explicit column classification"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]
        csv_content = "name,age,email\nJohn Doe,30,john@example.com\nJane Smith,25,jane@example.com"
        
        # Create file
        file = create_csv_file(csv_content)
        
        # Explicitly classify columns
        column_classification = {
            "name": {"type": "string", "subtype": "person_name"},
            "age": {"type": "number", "subtype": "integer"},
            "email": {"type": "string", "subtype": "email"}
        }
        
        # Convert to JSON string as expected by Form parameter
        import json
        column_classification_json = json.dumps(column_classification)
        
        # Mock pandas read_csv
        mock_df = pd.DataFrame({
            "name": ["John Doe", "Jane Smith"],
            "age": [30, 25],
            "email": ["john@example.com", "jane@example.com"]
        })
        
        with patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df), \
             patch("backend.app.endpoints.crocodile_api.parse_json_column_classification", return_value=column_classification):
            # Act - Using query parameter for table_name
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=classified_table",
                data={"column_classification": column_classification_json},
                files={"file": ("classified.csv", file, "text/csv")}
            )
            
            # Assert
            assert response.status_code == status.HTTP_201_CREATED
            
            # Verify classification was stored
            # This might be stored in a different way in your system, adjust as needed
            table = test_db.tables.find_one({
                "dataset_name": dataset_name,
                "table_name": "classified_table"
            })
            assert table is not None
            assert "classification" in table

    def test_add_table_csv_invalid_format(self, client, test_dataset, create_csv_file):
        """Test adding a table with invalid CSV file"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]
        invalid_csv = "This is not a valid CSV file"
        
        file = create_csv_file(invalid_csv)
        
        # Mock pandas read_csv to raise an exception
        with patch("backend.app.endpoints.crocodile_api.pd.read_csv", side_effect=Exception("CSV parsing error")):
            # Act - Using query parameter for table_name
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=invalid_csv_table",
                files={"file": ("invalid.csv", file, "text/csv")}
            )
            
            # Assert
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Error processing CSV" in response.json()["detail"]
    
    def test_add_table_csv_missing_table_name(self, client, test_dataset, create_csv_file):
        """Test adding a CSV without specifying a table name"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]
        csv_content = "col1,col2\nvalue1,value2"
        
        file = create_csv_file(csv_content)
        
        # Act - Don't include table_name in the query parameter
        response = client.post(
            f"/datasets/{dataset_name}/tables/csv",
            files={"file": ("missing_table_name.csv", file, "text/csv")}
        )
        
        # Assert - Should return 422 Unprocessable Entity
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
    def test_add_table_csv_missing_file(self, client, test_dataset):
        """Test adding a CSV without providing a file"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]
        
        # Act - Don't include the file
        response = client.post(
            f"/datasets/{dataset_name}/tables/csv?table_name=missing_file_table"
        )
        
        # Assert - Should return 422 Unprocessable Entity
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
    def test_add_table_csv_nonexistent_dataset(self, client, create_csv_file, mock_token_payload):
        """Test adding a CSV to a nonexistent dataset"""
        # Arrange
        nonexistent_dataset = "nonexistent_dataset"
        csv_content = "col1,col2\nvalue1,value2"
        
        file = create_csv_file(csv_content)
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Act - Using query parameter for table_name
            response = client.post(
                f"/datasets/{nonexistent_dataset}/tables/csv?table_name=nonexistent_dataset_table",
                files={"file": ("nonexistent.csv", file, "text/csv")}
            )
            
            # Assert - Should return 404 Not Found
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert f"Dataset {nonexistent_dataset} not found" in response.json()["detail"]
            
    def test_add_large_csv(self, client, test_db, test_dataset, mock_background_tasks, create_csv_file):
        """Test adding a large CSV file"""
        # Arrange
        dataset_name = test_dataset["dataset_name"]
        
        # Create a large CSV with 1000 rows
        header = "id,name,value\n"
        rows = "\n".join([f"{i},name_{i},{i*10}" for i in range(1000)])
        csv_content = header + rows
        
        file = create_csv_file(csv_content)
        
        # Create a mock DataFrame with 1000 rows
        mock_df = pd.DataFrame({
            "id": list(range(1000)),
            "name": [f"name_{i}" for i in range(1000)],
            "value": [i*10 for i in range(1000)]
        })
        
        with patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            # Act - Using query parameter for table_name
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=large_csv_table",
                files={"file": ("large_file.csv", file, "text/csv")}
            )
            
            # Assert
            assert response.status_code == status.HTTP_201_CREATED
            
            # Verify the table has 1000 rows
            table = test_db.tables.find_one({
                "dataset_name": dataset_name,
                "table_name": "large_csv_table"
            })
            assert table is not None
            assert table["total_rows"] == 1000
    
    def test_troubleshoot_csv_upload(self, client, test_dataset, create_csv_file, mock_token_payload):
        """Troubleshoot the CSV upload to find out why it's failing"""
        dataset_name = test_dataset["dataset_name"]
        csv_content = "col1,col2,col3\nvalue1,valu2,value3\nvalue4,value5,value6"
        
        file = create_csv_file(csv_content)
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Try with table_name as query parameter
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=troubleshoot_table",
                files={"file": (file.name, file, "text/csv")}
            )
            
            print(f"Response with query param: {response.status_code}")
            print(f"Response JSON: {response.json() if response.status_code != 204 else 'No content'}")