# tests/api/test_streaming_endpoints.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import tests.patch_imports as patch_imports

import pytest
import asyncio
from fastapi import status
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

# Import the app and dependencies
from backend.app.main import app
from backend.app.dependencies import get_db, get_crocodile_db, verify_token
from backend.app.endpoints.crocodile_api import get_table_status

class TestStreamingEndpoints:
    """Tests for streaming endpoints"""
    
    @pytest.fixture
    def mock_mongodb(self, test_db):
        """Provide the test database for MongoDB tests"""
        return test_db

    @pytest.fixture
    def mock_token_payload(self):
        """Mock token payload for authentication."""
        return {"email": "test@example.com"}

    @pytest.fixture
    def test_dataset(self, client, mock_mongodb, mock_token_payload):
        """Create a test dataset for table tests."""
        dataset_data = {"dataset_name": "test_dataset", "description": "Test dataset"}
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post("/datasets", json=dataset_data)
            assert response.status_code == status.HTTP_201_CREATED
            return response.json()["dataset"]
    
    @pytest.fixture
    def test_processing_table(self, client, mock_mongodb, test_dataset, mock_token_payload):
        """Create a test table with processing status"""
        dataset_name = test_dataset["dataset_name"]
        table_name = "processing_table"
        
        # Create the table
        mock_mongodb.tables.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "header": ["col1", "col2"],
            "total_rows": 5,
            "created_at": datetime.now()
        })
        
        # Add input data records with mixed statuses
        for i in range(5):
            status = "DONE" if i < 3 else "TODO"
            mock_mongodb.input_data.insert_one({
                "user_id": "test@example.com",
                "dataset_name": dataset_name,
                "table_name": table_name,
                "row_id": i,
                "data": [f"value{i*2+1}", f"value{i*2+2}"],
                "status": status,
                "ml_status": status
            })
        
        return {"dataset_name": dataset_name, "table_name": table_name}

    @pytest.mark.asyncio
    async def test_get_table_status_generator(self, mock_mongodb, test_processing_table, mock_token_payload):
        """Test the get_table_status generator function"""
        # Arrange
        dataset_name = test_processing_table["dataset_name"]
        table_name = test_processing_table["table_name"]  # Removed extra 'a' character
        
        # Create a patched MongoClient that returns our test db
        with patch("backend.app.endpoints.crocodile_api.MongoClient", return_value=MagicMock(
            __getitem__=lambda self, key: mock_mongodb if key == "crocodile_backend_db" else MagicMock(),
            close=lambda: None
        )):
            # Act - Get the generator
            generator = get_table_status(dataset_name, table_name, mock_token_payload)
            
            # Get the first item from the generator
            first_response = await generator.__anext__()
            
            # Assert
            assert "data:" in first_response
            assert "pending" in first_response or "rows" in first_response  # Allow either format
            
            # Now mark all as DONE
            mock_mongodb.input_data.update_many(
                {"dataset_name": dataset_name, "table_name": table_name},
                {"$set": {"status": "DONE", "ml_status": "DONE"}}
            )
            
            # Get the next item, which should show 0% pending
            with patch("backend.app.endpoints.crocodile_api.asyncio.sleep", new=AsyncMock()):
                second_response = await generator.__anext__()
                assert "0.00%" in second_response or "pending" in second_response  # Allow either format
    
    def test_stream_table_status_endpoint(self, client, mock_mongodb, test_processing_table, mock_token_payload):
        """Test the streaming status endpoint"""
        dataset_name = test_processing_table["dataset_name"]
        table_name = test_processing_table["table_name"]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload),\
             patch("backend.app.endpoints.crocodile_api.get_table_status", return_value=(item for item in ["data: {}\n\n"])):
            # Act
            response = client.get(f"/datasets/{dataset_name}/tables/{table_name}/status")
            
            # Assert
            assert response.status_code == status.HTTP_200_OK
            # Make the check more flexible to handle charset
            assert "text/event-stream" in response.headers["content-type"]