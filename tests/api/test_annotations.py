# tests/api/test_annotations.py

#Standard library imports for path manipulation
import sys
import os

#Add the project root directory to Python's module search path
#This allows importing modules from the project root when running tests from nested directories
# __file__ -> tests/api/test_annotations.py
# dirname(__file__) -> tests/api/
# dirname(dirname(__file__)) -> tests/
# dirname(dirname(dirname(__file__))) -> project_root/
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

#Import test utilities for pathching/mocking imports
import tests.patch_imports as patch_imports

#import frameowrks imports
import pytest
from fastapi import status

class TestAnnotations:
    """Tests for annotation and entity management
    
    This test class covers the core annotation features:
    - updating annotations for table cells 
    - Managing entity matches and scores
    - Deleting entity candidates 
    """
    
    def test_update_annotation(self, client, test_table_with_data):
        """Test updating an annotation for a cell

        Tests the scenario where a user manually adds a new entity annotation
        that wasn't in the original entity linking results.

        Args:
            client: FastAPI test client fixture
            test_table_with_data: Fixture providing test dataset with sample data
        
        """
        # Arrange
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        row_id = 0
        column_id = 0
        
        # Create annotation data for a new entity (not in existing results)
        # This simulates a user manually adding an entity annotation
        annotation_data = {
            "entity_id": "entity_0_2",  # New entity not in current results
            "match": True,  # Mark as positive match
            "score": 0.95, # High confidence score
            "notes": "Manual annotation", # User notes
            "candidate_info": { # Full entity information
                "id": "entity_0_2",
                "name": "New Entity",
                "description": "Manually added entity",
                "types": [    # This was missing in your test
                    {
                        "id": "type1",
                        "name": "Test Type"
                    }
                ]
            }
        }
        
        # Act - Make the API call to update the annotation  
        response = client.put(
            f"/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}",
            json=annotation_data
        )
        
        # Assert - Verify the response and behavior
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["message"] == "Annotation updated successfully"
        assert response.json()["entity"]["id"] == "entity_0_2"
        assert response.json()["entity"]["match"] is True
        assert response.json()["manually_annotated"] is True

    def test_update_annotation_existing_entity(self, client, test_db, test_table_with_data):
        """Test updating an annotation for an existing entity
        
        Tests the workflow of:
        1. Unmarking a curently matched entity
        2. Marking a different entity as matched
        3. Verifying database state changes

        Args:
            client: FastAPI test client fixture
            test_db: Database connection fixture for direct DB access
            test_table_with_data: Fixture providing test dataset with sample data
        """
        # Arrange - Set up test data and get existing entities
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        row_id = 0
        column_id = 0
        
        # Query the database directly to get existing entity linking results
        row = test_db.input_data.find_one({
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id
        })
        
        #get the first entity (presumably currently matched)
        existing_entity_id = row["el_results"]["0"][0]["id"]
        
        # Find a second entity to mark as matched later
        second_entity_id = None
        for candidate in row["el_results"]["0"]:
            if candidate["id"] != existing_entity_id:
                second_entity_id = candidate["id"]
                break
        
        # First, unmark the currently matched entity
        annotation_data = {
            "entity_id": existing_entity_id,
            "match": False,  # Change the match status
            "score": None,   # Remove score since not matched
            "notes": "Changed annotation"
        }
        
        # Act - Update the first entity to unmatched
        response = client.put(
            f"/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}",
            json=annotation_data
        )
        
        # Assert - Verify the unmatch operation
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["entity"]["id"] == existing_entity_id
        assert response.json()["entity"]["match"] is False
        assert response.json()["entity"]["score"] is None
        
        # Verify the change was persisted to the database
        updated_row = test_db.input_data.find_one({
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id
        })
        assert updated_row["manually_annotated"] is True
        
        # Now test marking a different entity as matched
        if second_entity_id:
            # Create annotation data for the second entity
            second_annotation_data = {
                "entity_id": second_entity_id,
                "match": True,
                "score": 0.8
            }
            # Act - Mark the second entity as matched
            second_response = client.put(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}",
                json=second_annotation_data
            )
            # Assert - Verify the match operation
            assert second_response.status_code == status.HTTP_200_OK
            assert second_response.json()["entity"]["id"] == second_entity_id
            assert second_response.json()["entity"]["match"] is True
            
            # Verify that at least one entity is marked as matched in the final state
            final_row = test_db.input_data.find_one({
                "dataset_name": dataset_name,
                "table_name": table_name,
                "row_id": row_id
            })
            # Check that there's at least one matched entity
            found_match = False
            for candidate in final_row["el_results"]["0"]:
                if candidate["match"]:
                    found_match = True
                    break
            assert found_match, "No entity is marked as matched after explicit update"

    def test_delete_candidate(self, client, test_db, test_table_with_data):
        """Test deleting a candidate from entity linking results
        
        Tests the ability to completely remove an entity candidate from the results, 
        which might be used when an entitiy is clearly irrelevant  or incorrect.

        Args:
            client: FastAPI test client fixture
            test_db: Database connection fixture for direct DB access
            test_table_with_data: Fixture providing test dataset with sample data
        
        """
        # Arrange - Set up test data and identify entity to delete
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        row_id = 0
        column_id = 0
        
        # Get the current entity linking results from the database
        row = test_db.input_data.find_one({
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id
        })
        
        # Select the second entity candidate for deletion
        # (avoiding the first one in case it's the matched entity)
        entity_id = row["el_results"]["0"][1]["id"]  # Get the second entity
        
        # Act - Delete the candidate via API
        response = client.delete(
            f"/datasets/{dataset_name}/tables/{table_name}/rows/{row_id}/columns/{column_id}/candidates/{entity_id}"
        )
        
        # Assert - Verify successful deletion
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["message"] == "Candidate deleted successfully"
        
        # Verify the entity was actually removed from the database
        updated_row = test_db.input_data.find_one({
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": row_id
        })
        
        # Check that the deleted entity is no longer in the candidates list
        candidate_ids = [c["id"] for c in updated_row["el_results"]["0"]]
        assert entity_id not in candidate_ids

   