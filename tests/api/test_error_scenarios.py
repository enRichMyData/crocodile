# tests/api/test_error_scenarios.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import tests.patch_imports as patch_imports

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from datetime import datetime
from io import BytesIO
from unittest.mock import patch, MagicMock

# Import the app and dependencies
from backend.app.main import app
from backend.app.dependencies import get_db, get_crocodile_db, verify_token

class TestErrorScenarios:
    """Tests for various error scenarios"""
    
    @pytest.fixture
    def mock_mongodb(self, test_db):
        """Provide the test database for MongoDB tests"""
        return test_db

    @pytest.fixture
    def mock_token_payload(self):
        """Mock token payload for authentication."""
        return {"email": "test@example.com"}

    @pytest.fixture
    def invalid_token_payload(self):
        """Mock token payload with insufficient permissions."""
        return {"email": "unauthorized@example.com", "role": "guest"}

    def test_authentication_failure(self, client):
        """Test endpoint access without authentication token"""
        # Act - Leave out the auth token
        with patch("backend.app.dependencies.verify_token", side_effect=Exception("Not authenticated")):
            response = client.get("/datasets")
            
            # Assert
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_dataset_not_found(self, client, mock_mongodb, mock_token_payload):
        """Test accessing a dataset that doesn't exist"""
        # Arrange
        nonexistent_dataset = "nonexistent_dataset"
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Act
            response = client.get(f"/datasets/{nonexistent_dataset}/tables")
            
            # Assert
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert f"Dataset {nonexistent_dataset} not found" in response.json()["detail"]
    
    def test_dataset_creation_duplicate(self, client, mock_mongodb, mock_token_payload):
        """Test creating a dataset with a name that already exists"""
        # Arrange - Create a dataset
        dataset_data = {"dataset_name": "duplicate_dataset", "description": "Test dataset"}
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # First creation should succeed
            response = client.post("/datasets", json=dataset_data)
            assert response.status_code == status.HTTP_201_CREATED
            
            # Act - Try to create with the same name
            response = client.post("/datasets", json=dataset_data)
            
            # Assert
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "already exists" in response.json()["detail"]
    
    def test_invalid_pagination_parameters(self, client, mock_mongodb, mock_token_payload):
        """Test using invalid pagination parameters"""
        # Arrange
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Act - Provide both next_cursor and prev_cursor
            response = client.get("/datasets?next_cursor=123&prev_cursor=456")
            
            # Assert
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Only one of next_cursor or prev_cursor" in response.json()["detail"]
    
    def test_invalid_cursor_format(self, client, mock_mongodb, mock_token_payload):
        """Test using an invalid cursor format"""
        # Arrange
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Act - Provide an invalid cursor
            response = client.get("/datasets?next_cursor=invalid_cursor")
            
            # Assert
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Invalid cursor format" in response.json()["detail"]

    def test_invalid_cursor_format(self, client, mock_mongodb, mock_token_payload):
        """Test using an invalid cursor format"""
        # Arrange
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Act - Provide an invalid cursor
            response = client.get("/datasets?next_cursor=not-a-valid-object-id")
            
            # Assert
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Invalid cursor value" in response.json()["detail"]
    
    def test_update_annotation_row_not_found(self, client, mock_mongodb, mock_token_payload):
        """Test updating an annotation for a row that doesn't exist"""
        # Arrange
        dataset_name = "test_dataset"
        table_name = "test_table"
        
        # Create the dataset and table but no rows
        mock_mongodb.datasets.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "created_at": datetime.now()
        })
        
        mock_mongodb.tables.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "created_at": datetime.now()
        })
        
        annotation_data = {
            "entity_id": "entity_1",
            "match": True,
            "score": 0.9
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Act
            response = client.put(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/999/columns/0",
                json=annotation_data
            )
            
            # Assert
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "Row 999 not found" in response.json()["detail"]
    
    def test_update_annotation_new_entity_no_info(self, client, mock_mongodb, mock_token_payload):
        """Test updating an annotation with a new entity ID but no candidate info"""
        # Arrange
        dataset_name = "test_dataset"
        table_name = "test_table"
        row_id = 0
        
        # Create dataset, table and row
        mock_mongodb.datasets.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "created_at": datetime.now()
        })
        
        mock_mongodb.tables.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "created_at": datetime.now()
        })
        
        mock_mongodb.input_data.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id,
            "data": ["test_value"],
            "el_results": {
                "0": [
                    {
                        "id": "existing_entity",
                        "name": "Existing Entity",
                        "match": True
                    }
                ]
            }
        })
        
        # Try to add a new entity without providing candidate_info
        annotation_data = {
            "entity_id": "new_entity",  # New ID not in current results
            "match": True,
            "score": 0.9
            # Missing candidate_info
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Act
            response = client.put(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/0",
                json=annotation_data
            )
            
            # Assert
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Please provide 'candidate_info'" in response.json()["detail"]
    
    def test_delete_candidate_no_candidates(self, client, mock_mongodb, mock_token_payload):
        """Test deleting a candidate when no candidates exist"""
        # Arrange
        dataset_name = "test_dataset"
        table_name = "test_table"
        row_id = 0
        
        # Create dataset, table and row with no candidates
        mock_mongodb.datasets.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "created_at": datetime.now()
        })
        
        mock_mongodb.tables.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "created_at": datetime.now()
        })
        
        mock_mongodb.input_data.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id,
            "data": ["test_value"],
            "el_results": {}  # No candidates
        })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Act
            response = client.delete(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/0/candidates/some_entity"
            )
            
            # Assert
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "No entity linking candidates found" in response.json()["detail"]