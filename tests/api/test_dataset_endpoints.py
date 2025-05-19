# tests/api/test_dataset_endpoints.py
# Import the patch before anything else
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import tests.patch_imports as patch_imports

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from bson import ObjectId

# Import the app and dependencies
from backend.app.main import app
from backend.app.dependencies import get_db, get_crocodile_db, verify_token

class TestDatasetEndpoints:
    """Tests for dataset-related endpoints"""
    
    @pytest.fixture
    def mock_mongodb(self, test_db):
        """Provide the test database for MongoDB tests"""
        return test_db
    
    def test_create_dataset(self, client, mock_mongodb):
        """Test creating a new dataset"""
        # Arrange
        dataset_data = {
            "dataset_name": "test_create_dataset",
            "description": "Dataset for testing"
        }
        
        # Act
        response = client.post("/datasets", json=dataset_data)
        
        # Assert
        assert response.status_code == 201
        assert response.json()["message"] == "Dataset created successfully"
        assert response.json()["dataset"]["dataset_name"] == "test_create_dataset"
        
        # Verify it was created in the database
        dataset = mock_mongodb.datasets.find_one({"dataset_name": "test_create_dataset"})
        assert dataset is not None
        assert dataset["user_id"] == "test@example.com"
    
    def test_get_datasets_empty(self, client, mock_mongodb):
        """Test retrieving datasets when none exist"""
        # Ensure database is empty
        mock_mongodb.datasets.delete_many({})
        
        # Act
        response = client.get("/datasets")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 0
        assert data["pagination"]["next_cursor"] is None
        assert data["pagination"]["prev_cursor"] is None
    
    def test_get_datasets(self, client, mock_mongodb):
        """Test retrieving datasets"""
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
        
        # Act
        response = client.get("/datasets")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 3
        
        # Check dataset names
        dataset_names = [d["dataset_name"] for d in data["data"]]
        for i in range(3):
            assert f"dataset_{i}" in dataset_names
    
    def test_delete_dataset(self, client, mock_mongodb):
        """Test deleting a dataset"""
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
        
        # Act
        response = client.delete("/datasets/dataset_to_delete")
        
        # Assert
        assert response.status_code == 204
        
        # Verify it was deleted
        dataset = mock_mongodb.datasets.find_one({"dataset_name": "dataset_to_delete"})
        assert dataset is None
    
    def test_delete_nonexistent_dataset(self, client, mock_mongodb):
        """Test deleting a dataset that doesn't exist"""
        # Ensure the dataset doesn't exist
        mock_mongodb.datasets.delete_many({"dataset_name": "nonexistent_dataset"})
        
        # Act
        response = client.delete("/datasets/nonexistent_dataset")
        
        # Assert
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]