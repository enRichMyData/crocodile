# tests/api/test_csv_upload.py

# Standard library imports for path manipulation
import sys
import os
# Add project root to Python path to enable imports from project modules
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import test configuration and patch utilities
from tests.conftest import test_db
import tests.patch_imports as patch_imports

# Testing framework and web testing imports
import pytest
from fastapi import status
from io import BytesIO  # For creating in-memory file objects
from unittest.mock import patch, MagicMock

# Data processing and manipulation libraries
import pandas as pd
import numpy as np
import json
from datetime import datetime


class TestCSVUpload:
    """Tests for CSV upload functionality
    
    This test class covers various scenarios for uploading CSV files:
    - Basic CSV upload with different data types
    - Handling null/missing values
    - Column classification and metadata
    - Edge cases like duplicate column names
    - Error handling for missing parameters
    - performance with large CSV files
    """

    @pytest.fixture
    def mock_token_payload(self):
        """Mock token payload for authentication testing.
        
        Returns:
            dict: Mock authentication payload with test email
        """
        return {"email": "test@example.com"}

    @pytest.fixture
    def create_csv_file(self):
        """Helper fixture to create CSV file BytesIO objects with custom content.
        
        This fixture returns a function that can create in-memory CSV files
        for testing without needing actual file I/O operations.
        
        Returns:
            function: Factory function that creates BytesIO objects with CSV content
        """
        def _create_csv_file(content, filename="test_file.csv"):
            """Create a BytesIO object representing a CSV file.
            
            Args:
                content (str): The CSV content as a string
                filename (str): The filename to assign to the file object
                
            Returns:
                BytesIO: In-memory file object with CSV content
            """
            file = BytesIO(content.encode())
            file.name = filename
            return file
        return _create_csv_file

    def test_add_table_csv(self, client, test_db, test_dataset, mock_background_tasks, create_csv_file):
        """Test adding a table with CSV file containing different data types.
        
        This test verifies that the API can handle CSV files with various data types
        including strings, integers, floats, and dates.
        
        Args:
            client: FastAPI test client fixture
            test_db: Database connection fixture
            test_dataset: Test dataset fixture
            mock_background_tasks: Mock for background task processing
            create_csv_file: Helper function to create test CSV files
        """
        # Arrange - Set up test data
        dataset_name = test_dataset["dataset_name"]
        # Simple CSV content with basic structure
        csv_content = "col1,col2,col3\nvalue1,value2,value3\nvalue4,value5,value6"
        
        # Create an in-memory file object
        file = create_csv_file(csv_content)
        
        # Create a mock DataFrame with different data types to test type handling
        mock_df = pd.DataFrame({
            "string_col": ["text1", "text2"], # String data
            "int_col": [10, 20], 
            "float_col": [10.5, 20.5],
            "date_col": pd.to_datetime(["2023-01-01", "2023-02-01"])
        })
        # Mock pandas read_csv to return our controlled test data
        with patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            # Act - Make API call to upload CSV
            # Using query parameter for table_name (instead of form data)
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=datatypes_table",
                files={"file": ("datatypes.csv", file, "text/csv")}
            )
            
            # Assert - Verify successful upload and correct data storage
            assert response.status_code == status.HTTP_201_CREATED
            assert response.json()["tableName"] == "datatypes_table"
            
            # Verify the table was created in database with correct structure
            table = test_db.tables.find_one({
                "dataset_name": dataset_name,
                "table_name": "datatypes_table"
            })
            # Check that all column types are preserved in the header
            assert table is not None
            assert "string_col" in table["header"]
            assert "int_col" in table["header"]
            assert "float_col" in table["header"]
            assert "date_col" in table["header"]


    def test_add_table_csv_with_nulls(self, client, test_db, test_dataset, mock_background_tasks, create_csv_file):
        """Test adding a CSV with missing/null values.
        
        This test ensures the API properly handles CSV files with empty cells
        and null values, which are common in real-world data.
        
        Args:
            client: FastAPI test client fixture
            test_db: Database connection fixture
            test_dataset: Test dataset fixture
            mock_background_tasks: Mock for background task processing
            create_csv_file: Helper function to create test CSV files
        """
        # Arrange - Set up test data with missing values
        dataset_name = test_dataset["dataset_name"]
        # CSV with empty cells (represented by adjacent commas)
        csv_content = "col1,col2,col3\nvalue1,,value3\n,value5,value6"
        
        # Create file
        file = create_csv_file(csv_content)
        
        # Mock pandas DataFrame with NaN values (how pandas represents missing data)
        mock_df = pd.DataFrame({
            "col1": ["value1", np.nan],
            "col2": [np.nan, "value5"],
            "col3": ["value3", "value6"]
        })
        # Mock the replace method to verify NaN handling
        # This is important for JSON serialization (NaN isn't valid JSON)
        mock_df.replace = MagicMock(return_value=mock_df)
        
        # Mock pandas read_csv to return our test DataFrame
        with patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
             # Act - Upload CSV with null values
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=null_values_table",
                files={"file": ("nulls.csv", file, "text/csv")}
            )
            
            # Assert - Verify successful processing of null values
            assert response.status_code == status.HTTP_201_CREATED
            
            # Verify that NaN replacement was called
            # This ensures the API is handling null values properly for JSON serialization
            assert mock_df.replace.called
            # Check that np.nan was part of the replacement call
            # This is typically replaced with None for JSON compatibility
            call_args = mock_df.replace.call_args[0][0]
            assert np.nan in call_args or np.nan in call_args.values()

    def test_add_table_csv_column_classification(self, client, test_db, test_dataset, mock_background_tasks, create_csv_file):
        """Test adding a CSV with explicit column classification metadata.
        
        This test verifies that users can provide semantic information about columns
        (e.g., marking a column as containing person names, email addresses, etc.)
        which can be used for enhanced processing and validation.
        
        Args:
            client: FastAPI test client fixture
            test_db: Database connection fixture
            test_dataset: Test dataset fixture
            mock_background_tasks: Mock for background task processing
            create_csv_file: Helper function to create test CSV files
        """
        # Arrange - Set up test data with semantic meaning
        dataset_name = test_dataset["dataset_name"]
        # CSV with columns that have clear semantic types   
        csv_content = "name,age,email\nJohn Doe,30,john@example.com\nJane Smith,25,jane@example.com"
        
        # Create file object
        file = create_csv_file(csv_content)
        
        # Define explicit column classifications with semantic types
        column_classification = {
            "name": {"type": "string", "subtype": "person_name"},
            "age": {"type": "number", "subtype": "integer"},
            "email": {"type": "string", "subtype": "email"}
        }
        
        # Convert classification to JSON string (as expected by the API form parameter)
        column_classification_json = json.dumps(column_classification)
        
        # Mock pandas DataFrame with the test data
        mock_df = pd.DataFrame({
            "name": ["John Doe", "Jane Smith"],
            "age": [30, 25],
            "email": ["john@example.com", "jane@example.com"]
        })
        
        # Mock both pandas read_csv and the classification parsing function
        with patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df), \
            patch("backend.app.endpoints.crocodile_api.parse_json_column_classification", return_value=column_classification):

            # Act - Upload CSV with column classification metadata
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=classified_table",
                data={"column_classification": column_classification_json},
                files={"file": ("classified.csv", file, "text/csv")}
            )
            
            # Assert - Verify classification was stored correctly
            assert response.status_code == status.HTTP_201_CREATED
            
            # Verify classification metadata was persisted to database
            table = test_db.tables.find_one({
                "dataset_name": dataset_name,
                "table_name": "classified_table"
            })
            assert table is not None
            # Check for the presence of classification metadata
            assert "classified_columns" in table
            # Verify the content matches what we provided
            assert table["classified_columns"] == column_classification

    def test_add_table_csv_duplicate_column_names(self, client, test_db, test_dataset, mock_background_tasks, create_csv_file):
        """Test adding a CSV file with duplicate column names.
        
        This test ensures the API handles the common issue of CSV files having
        duplicate column headers, which pandas typically resolves by adding suffixes.
        
        Args:
            client: FastAPI test client fixture
            test_db: Database connection fixture
            test_dataset: Test dataset fixture
            mock_background_tasks: Mock for background task processing
            create_csv_file: Helper function to create test CSV files
        """

        # Arrange - Set up CSV with duplicate column names
        dataset_name = test_dataset["dataset_name"]
        # CSV with "col1" appearing twice in the header
        csv_content = "col1,col1,col2\nvalue1,value2,value3\nvalue4,value5,value6"
        
        # Create file object
        file = create_csv_file(csv_content)
        
        # Mock DataFrame showing how pandas handles duplicate column names
        # Pandas typically adds .1, .2, etc. to make column names unique
        mock_df = pd.DataFrame({
            "col1": ["value1", "value4"],
            "col1.1": ["value2", "value5"],
            "col2": ["value3", "value6"]
        })
        
        # Mock pandas read_csv
        with patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            # Act - Upload CSV with duplicate column names
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=duplicate_columns_table",
                files={"file": ("duplicate_columns.csv", file, "text/csv")}
            )
            
            # Assert - Verify successful handling of duplicate column names
            assert response.status_code == status.HTTP_201_CREATED
            
            # Verify the table was created with properly renamed columns 
            table = test_db.tables.find_one({
                "dataset_name": dataset_name,
                "table_name": "duplicate_columns_table"
            })
            assert table is not None
            # Check that pandas' column renaming was preserved
            assert table["header"] == ["col1", "col1.1", "col2"]
            assert table["total_rows"] == 2

    def test_add_table_csv_missing_parameters(self, client, test_dataset, create_csv_file):
        """Test CSV upload with missing required parameters.
        
        This test verifies proper error handling when required parameters
        are missing from the API request.
        
        Args:
            client: FastAPI test client fixture
            test_dataset: Test dataset fixture
            create_csv_file: Helper function to create test CSV files
        """
        # Arrange - Set up basic test data
        dataset_name = test_dataset["dataset_name"]
        csv_content = "col1,col2\nvalue1,value2"
        
        file = create_csv_file(csv_content)
        
        # Test Case 1: Missing table_name parameter
        response = client.post(
            f"/datasets/{dataset_name}/tables/csv",
            files={"file": ("missing_table_name.csv", file, "text/csv")}
        )
        
        # Should return 422 Unprocessable Entity due to missing required parameter
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test Case 2: Missing file parameter
        response = client.post(
            f"/datasets/{dataset_name}/tables/csv?table_name=missing_file_table"
        )
        
        # Should return 422 Unprocessable Entity due to missing required file
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_add_large_csv(self, client, test_db, test_dataset, mock_background_tasks, create_csv_file):
        """Test adding a large CSV file to verify performance and memory handling.
        
        This test ensures the API can handle larger datasets without performance
        issues or memory problems. Uses 1000 rows as a reasonable test size.
        
        Args:
            client: FastAPI test client fixture
            test_db: Database connection fixture
            test_dataset: Test dataset fixture
            mock_background_tasks: Mock for background task processing
            create_csv_file: Helper function to create test CSV files
        """    
        # Arrange - Create a large CSV dataset
        dataset_name = test_dataset["dataset_name"]
        
        # Generate CSV content with 1000 rows programmatically
        header = "id,name,value\n"
        # Create 1000 rows of test data
        rows = "\n".join([f"{i},name_{i},{i*10}" for i in range(1000)])
        csv_content = header + rows
        # Create file object with large content
        file = create_csv_file(csv_content)
        
        # Create a corresponding mock DataFrame with 1000 rows
        mock_df = pd.DataFrame({
            "id": list(range(1000)),     # Sequential IDs 0-999
            "name": [f"name_{i}" for i in range(1000)], # Generated names
            "value": [i*10 for i in range(1000)] # Calculated values
        })
        # Mock pandas read_csv to return our large test DataFrame
        with patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            # Act - Upload large CSV file
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=large_csv_table",
                files={"file": ("large_file.csv", file, "text/csv")}
            )
            
            # Assert - Verify successful processing of large file
            assert response.status_code == status.HTTP_201_CREATED
            
            # Verify the table was created with correct row count
            table = test_db.tables.find_one({
                "dataset_name": dataset_name,
                "table_name": "large_csv_table"
            })
            assert table is not None
            # Confirm all 1000 rows were processed and stored
            assert table["total_rows"] == 1000