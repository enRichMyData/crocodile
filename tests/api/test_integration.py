# tests/api/test_integration.py
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
from crocodile import Crocodile


class TestIntegrationWorkflow:
    """Integration tests for the full API workflow"""
    
    @pytest.fixture
    def mock_mongodb(self, test_db):
        """Provide the test database for MongoDB tests"""
        return test_db

    @pytest.fixture
    def mock_crocodile_db(self, test_db):
        """Provide the test database as if it were the crocodile DB"""
        return test_db

    @pytest.fixture
    def mock_token_payload(self):
        """Mock token payload for authentication."""
        return {"email": "test@example.com"}

    def test_full_workflow(self, client, mock_mongodb, mock_token_payload, mock_crocodile_db):
        """Test the full workflow from dataset creation to result retrieval"""
        
        # Step 1: Create a dataset
        dataset_data = {"dataset_name": "integration_test", "description": "Integration test dataset"}
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post("/datasets", json=dataset_data)
            assert response.status_code == status.HTTP_201_CREATED
            assert response.json()["dataset"]["dataset_name"] == "integration_test"
        
        # Step 2: Add a table with JSON data
        table_data = {
            "table_name": "integration_table",
            "header": ["col1", "col2", "col3"],
            "total_rows": 2,
            "data": [
                {"col1": "value1", "col2": "value2", "col3": "value3"},
                {"col1": "value4", "col2": "value5", "col3": "value6"}
            ]
        }
        
        # Mock the Crocodile class 
        mock_crocodile = MagicMock()
        mock_crocodile.run = MagicMock()
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload),\
             patch("backend.app.endpoints.crocodile_api.Crocodile", return_value=mock_crocodile),\
             patch("backend.app.endpoints.crocodile_api.ResultSyncService") as mock_sync_service:
            
            # Configure the sync service mock
            mock_sync_instance = mock_sync_service.return_value
            mock_sync_instance.sync_results = MagicMock()
            
            # Add the table
            response = client.post(
                "/datasets/integration_test/tables/json", 
                json=table_data
            )
            assert response.status_code == status.HTTP_201_CREATED
            assert response.json()["tableName"] == "integration_table"
            
            # Verify Crocodile was instantiated with expected params
            assert mock_crocodile.run.called
            
            # Simulate processing by adding results directly to the database
            for i in range(2):
                mock_mongodb.input_data.insert_one({
                    "user_id": "test@example.com",
                    "dataset_name": "integration_test",
                    "table_name": "integration_table",
                    "row_id": i,
                    "data": [f"value{i*3+1}", f"value{i*3+2}", f"value{i*3+3}"],
                    "status": "DONE",
                    "ml_status": "DONE",
                    "el_results": {
                        "0": [
                            {
                                "id": f"entity_{i}",
                                "name": f"Entity {i}",
                                "description": "Integration test entity",
                                "score": 0.9,
                                "match": True
                            }
                        ]
                    }
                })
        
        # Step 3: Get the tables for the dataset
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.get("/datasets/integration_test/tables")
            assert response.status_code == status.HTTP_200_OK
            assert len(response.json()["data"]) == 1
            assert response.json()["data"][0]["table_name"] == "integration_table"
        
        # Step 4: Get the table data 
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload),\
             patch("backend.app.dependencies.get_crocodile_db", return_value=mock_crocodile_db):
            response = client.get("/datasets/integration_test/tables/integration_table")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["data"]["status"] == "DONE"  # All rows processed
            assert len(data["data"]["rows"]) == 2
            
            # Verify the linked entities are present
            for row in data["data"]["rows"]:
                assert "linked_entities" in row
                assert len(row["linked_entities"]) == 1
                assert row["linked_entities"][0]["idColumn"] == 0
        
        # Step 5: Update an annotation
        annotation_data = {
            "entity_id": "new_entity",
            "match": True,
            "score": 0.95,
            "notes": "Manual annotation",
            "candidate_info": {
                "id": "new_entity",
                "name": "New Entity",
                "description": "Manually added entity",
                "uri": "http://example.org/new_entity"
            }
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.put(
                "/datasets/integration_test/tables/integration_table/rows/0/columns/0",
                json=annotation_data
            )
            assert response.status_code == status.HTTP_200_OK
            assert response.json()["entity"]["name"] == "New Entity"
            assert response.json()["entity"]["match"] is True
        
        # Step 6: Get the table data again to see the updated annotation
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload),\
             patch("backend.app.dependencies.get_crocodile_db", return_value=mock_crocodile_db):
            response = client.get("/datasets/integration_test/tables/integration_table")
            assert response.status_code == status.HTTP_200_OK
            
            # Find row 0 and check its linked entities
            row_0 = next((r for r in response.json()["data"]["rows"] if r["idRow"] == 0), None)
            assert row_0 is not None
            
            # Check the annotation has been updated
            entities = row_0["linked_entities"][0]["candidates"]
            assert entities[0]["id"] == "new_entity"
            assert entities[0]["match"] is True
        
        # Step 7: Delete the dataset (cleanup)
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload),\
             patch("backend.app.dependencies.get_crocodile_db", return_value=mock_crocodile_db):
            response = client.delete("/datasets/integration_test")
            assert response.status_code == status.HTTP_204_NO_CONTENT
            
            # Verify the dataset is gone
            response = client.get("/datasets")
            datasets = [d["dataset_name"] for d in response.json()["data"]]
            assert "integration_test" not in datasets