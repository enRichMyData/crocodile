# tests/api/test_dataset_endpoints.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import tests.patch_imports as patch_imports

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from datetime import datetime
from bson import ObjectId
from unittest.mock import patch

# Import the app and dependencies
from backend.app.main import app
from backend.app.dependencies import get_db, get_crocodile_db, verify_token

class TestDatasetEndpoints:
    """Tests for dataset-related endpoints

    this test class covers the following endpoints:
    - POST /datasets: Create a new dataset
    - GET /datasets: Retrieve datasets with pagination
    - DELETE /datasets/{dataset_name}: Delete a dataset
    - GET /datasets/{dataset_name}/tables: Retrieve tables in a dataset
    - POST /datasets/{dataset_name}/tables: Add a table to a dataset
    - DELETE /datasets/{dataset_name}/tables/{table_name}: Delete a table from a dataset
    """
    
    @pytest.fixture
    def mock_mongodb(self, test_db):
        """Provide the test database for MongoDB tests
        
        This fixture sets up a mock MongoDB database for testing purposes.
        Args:
            test_db: The test database fixture that provides a MongoDB instance.
        Returns:
            MongoDB client: A mock MongoDB client connected to the test database.
        This allows tests to interact with a controlled database environment.
        """
        return test_db
    
    @pytest.fixture
    def mock_token_payload(self):
        """Mock token payload for authentication.
        
        This fixture simulates a user authentication payload for testing purposes.
        Returns:
            dict: A mock payload containing user information, such as email.
        This allows tests to verify authentication without needing a real token.

        Args:
            None
        Returns:
            dict: A mock token payload with user information.
        This simulates a user being authenticated in the system.
        """
        return {"email": "test@example.com"}
    
    def test_create_dataset(self, client, mock_mongodb):
        """Test creating a new dataset
        
        This test verifies that a dataset can be created successfully.
        Args:
            client: FastAPI test client fixture
            mock_mongodb: Mock MongoDB database fixture
        Returns:
            None
        This test checks that the dataset is created with the correct data and stored in the database.
        """
        # Arrange - Prepare dataset data
        dataset_data = {
            "dataset_name": "test_create_dataset", # Name of the dataset to create
            "description": "Dataset for testing" # Description of the dataset
        }
        
        # Act - Make the API call to create the dataset
        response = client.post("/datasets", json=dataset_data) 
        
        # Assert - Check the response status and content
        assert response.status_code == 201 # Expect 201 Created status
        assert response.json()["message"] == "Dataset created successfully" # Check success message
        assert response.json()["dataset"]["dataset_name"] == "test_create_dataset" # Verify dataset name in response
        
        # Verify it was created in the database
        dataset = mock_mongodb.datasets.find_one({"dataset_name": "test_create_dataset"})
        assert dataset is not None # Ensure dataset exists in the database
        assert dataset["user_id"] == "test@example.com" # Check that the user ID matches the mock token payload
    
    def test_get_datasets_empty(self, client, mock_mongodb):
        """Test retrieving datasets when none exist
        
        This test verifies that the endpoint returns an empty list when no datasets are present.
        Args:
            client: FastAPI test client fixture
            mock_mongodb: Mock MongoDB database fixture
        Returns:
            None
        This test checks that the response contains an empty data list and correct pagination info.
        """
        # Ensure database is empty
        mock_mongodb.datasets.delete_many({})
        
        # Act - Make the API call to retrieve datasets
        response = client.get("/datasets")
        
        # Assert - Check the response status and content
        assert response.status_code == 200 # Expect 200 OK status
        data = response.json() # Parse the JSON response
        assert "data" in data # Check that 'data' key exists in response
        assert len(data["data"]) == 0 # Ensure data list is empty
        assert data["pagination"]["next_cursor"] is None # Check pagination info
        assert data["pagination"]["prev_cursor"] is None # Ensure no previous cursor exists
    
    def test_get_datasets(self, client, mock_mongodb):
        """Test retrieving datasets

        This test verifies that the endpoint returns a list of datasets.
        Args:
            client: FastAPI test client fixture
            mock_mongodb: Mock MongoDB database fixture
        Returns:
            None
        This test checks that the response contains the expected datasets and pagination info.
        """
        # Clear existing datasets
        mock_mongodb.datasets.delete_many({})
        
        # Arrange - Create some datasets
        for i in range(3):
            mock_mongodb.datasets.insert_one({
                "user_id": "test@example.com",
                "dataset_name": f"dataset_{i}",
                "created_at": datetime.now(),
                "total_tables": 0,
                "total_rows": 0
            })
        
        # Act - Make the API call to retrieve datasets
        response = client.get("/datasets")
        
        # Assert - Check the response status and content
        assert response.status_code == 200 # Expect 200 OK status
        data = response.json() # Parse the JSON response
        assert "data" in data # Check that 'data' key exists in response
        assert len(data["data"]) == 3 # Ensure we got 3 datasets back
        
        # Check dataset names
        dataset_names = [d["dataset_name"] for d in data["data"]] 
        for i in range(3):
            assert f"dataset_{i}" in dataset_names
    
    def test_delete_dataset(self, client, mock_mongodb):
        """Test deleting a dataset
        
        This test verifies that a dataset can be deleted successfully.
        Args:
            client: FastAPI test client fixture
            mock_mongodb: Mock MongoDB database fixture
        Returns:
            None
        """
        # Clear existing datasets
        mock_mongodb.datasets.delete_many({})
        
        # Arrange - Create a dataset to delete
        mock_mongodb.datasets.insert_one({
            "user_id": "test@example.com",
            "dataset_name": "dataset_to_delete",
            "created_at": datetime.now(),
            "total_tables": 0,
            "total_rows": 0
        })
        
        # Act - Make the API call to delete the dataset
        response = client.delete("/datasets/dataset_to_delete")
        
        # Assert - Check the response status and content
        assert response.status_code == 200 # Expect 200 OK status
        
        # Verify it was deleted from the database
        dataset = mock_mongodb.datasets.find_one({"dataset_name": "dataset_to_delete"})
        assert dataset is None
    
    def test_dataset_pagination(self, client, mock_mongodb, mock_token_payload):
        """Test basic dataset pagination functionality
        This test verifies that the pagination works correctly for datasets.
        Args:
            client: FastAPI test client fixture
            mock_mongodb: Mock MongoDB database fixture
            mock_token_payload: Mock token payload for authentication
        Returns:
            None
        This test checks that the endpoint returns the correct number of datasets per page,
        and that pagination works with next_cursor.
        """
        # Create several datasets
        for i in range(15):
            mock_mongodb.datasets.insert_one({
                "user_id": "test@example.com",
                "dataset_name": f"pagination_test_{i}",
                "created_at": datetime.now(),
                "total_tables": 0,
                "total_rows": 0
            })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Test default pagination (should return 10 datasets)
            response = client.get("/datasets")
            assert response.status_code == status.HTTP_200_OK
            first_page = response.json()
            assert len(first_page["data"]) == 10
            assert first_page["pagination"]["next_cursor"] is not None
            
            # Test with custom limit
            response = client.get("/datasets?limit=5")
            assert response.status_code == status.HTTP_200_OK
            assert len(response.json()["data"]) == 5
            
            # Test pagination using next_cursor
            next_cursor = first_page["pagination"]["next_cursor"]
            response = client.get(f"/datasets?next_cursor={next_cursor}")
            assert response.status_code == status.HTTP_200_OK
            second_page = response.json()
            assert len(second_page["data"]) == 5  # Should return remaining 5 datasets
            
            # Check that datasets on second page are different from first page
            first_page_ids = [d["_id"] for d in first_page["data"]]
            second_page_ids = [d["_id"] for d in second_page["data"]]
            assert not set(first_page_ids).intersection(set(second_page_ids))