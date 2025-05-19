# tests/api/test_edge_cases.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import tests.patch_imports as patch_imports

import pytest
from fastapi import status
from unittest.mock import patch, MagicMock
from io import BytesIO
import json
import pandas as pd
import numpy as np
from datetime import datetime
from bson import ObjectId

class TestEdgeCases:
    """Tests for edge cases and exceptional situations"""
    
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

    # =================== Unusual Data Types ===================
    
    def test_special_chars_in_names(self, client, mock_token_payload):
        """Test creating datasets and tables with special characters in names"""
        # List of names with special characters
        special_names = [
            "name with spaces",
            "name-with-hyphens",
            "name_with_underscores",
            "name.with.dots",
            "name123WithNumbers",
            "UPPERCASE_name",
            "mixedCASE_name",
            # More extreme cases
            "ñáéíóú",  # Non-ASCII characters
            "非常に長い名前",  # Non-Latin characters
            "empty⟫↯⟪space",  # Symbols
        ]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Test dataset names
            for i, name in enumerate(special_names):
                dataset_name = f"test_{i}_{name}"
                response = client.post("/datasets", json={"dataset_name": dataset_name})
                
                # Some of these might be rejected by your validation
                if response.status_code == status.HTTP_201_CREATED:
                    assert response.json()["dataset"]["dataset_name"] == dataset_name
                else:
                    # If rejected, make sure it's for a valid reason
                    assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    def test_extreme_pagination_values(self, client, mock_mongodb, mock_token_payload):
        """Test pagination with extreme values"""
        # Seed database with some datasets
        for i in range(5):
            mock_mongodb.datasets.insert_one({
                "user_id": "test@example.com",
                "dataset_name": f"pagination_test_{i}",
                "created_at": datetime.now(),
                "total_tables": 0,
                "total_rows": 0
            })
        
        extreme_limits = [
            "0",  # Zero limit
            "-1",  # Negative limit
            "999999",  # Very large limit
            "not_a_number",  # Non-numeric limit
        ]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            for limit in extreme_limits:
                response = client.get(f"/datasets?limit={limit}")
                
                # The API should either handle these gracefully or reject them with proper error
                assert response.status_code in [
                    status.HTTP_200_OK,  # Handled gracefully
                    status.HTTP_400_BAD_REQUEST,  # Rejected with validation error
                    status.HTTP_422_UNPROCESSABLE_ENTITY  # Rejected by FastAPI validation
                ]
    
    def test_invalid_cursor_formats(self, client, mock_token_payload):
        """Test various invalid cursor formats"""
        invalid_cursors = [
            "not_an_objectid",
            "123",
            "a" * 23,  # Invalid length
            "g" * 24,  # Invalid characters
            "!@#$%^&*()",  # Special characters
            "null",  # String "null"
            "",  # Empty string
            " ",  # Space
        ]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            for cursor in invalid_cursors:
                response = client.get(f"/datasets?next_cursor={cursor}")
                
                # Should be rejected with validation error
                assert response.status_code in [
                    status.HTTP_400_BAD_REQUEST,
                    status.HTTP_422_UNPROCESSABLE_ENTITY
                ]
    
    # =================== Unusual Data Values ===================
    
    def test_empty_or_whitespace_values(self, client, test_dataset, mock_token_payload):
        """Test handling of empty or whitespace-only values"""
        dataset_name = test_dataset["dataset_name"]
        
        empty_values = [
            "",  # Empty string
            " ",  # Single space
            "   ",  # Multiple spaces
            "\t",  # Tab
            "\n",  # Newline
            "\r\n",  # Carriage return and newline
        ]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Test table names
            for value in empty_values:
                table_data = {
                    "table_name": value,
                    "header": ["col1", "col2"],
                    "total_rows": 1,
                    "data": [{"col1": "val1", "col2": "val2"}]
                }
                
                response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
                
                # Should be rejected with validation error
                assert response.status_code in [
                    status.HTTP_400_BAD_REQUEST,
                    status.HTTP_422_UNPROCESSABLE_ENTITY
                ]
            
            # Test header values
            table_data = {
                "table_name": "empty_header_test",
                "header": [""],  # Empty header
                "total_rows": 1,
                "data": [{"": "val1"}]
            }
            
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            
            # Should either be rejected or handled gracefully
            assert response.status_code in [
                status.HTTP_201_CREATED,  # Handled gracefully
                status.HTTP_400_BAD_REQUEST,  # Rejected with validation error
                status.HTTP_422_UNPROCESSABLE_ENTITY  # Rejected by FastAPI validation
            ]
    
    def test_duplicate_data_elements(self, client, test_dataset, mock_token_payload):
        """Test handling of duplicate elements in data structures"""
        dataset_name = test_dataset["dataset_name"]
        
        # Duplicate headers
        table_data = {
            "table_name": "duplicate_headers_test",
            "header": ["col1", "col2", "col1"],  # Duplicate header
            "total_rows": 1,
            "data": [{"col1": "val1", "col2": "val2"}]  # Note: JSON doesn't allow duplicate keys
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            
            # Should be rejected or headers should be made unique automatically
            assert response.status_code in [
                status.HTTP_201_CREATED,  # Handled gracefully
                status.HTTP_400_BAD_REQUEST,  # Rejected with validation error
                status.HTTP_422_UNPROCESSABLE_ENTITY  # Rejected by FastAPI validation
            ]
    
    def test_unusual_data_types_in_table(self, client, test_dataset, mock_token_payload):
        """Test handling of unusual data types in table data"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create a table with various data types
        table_data = {
            "table_name": "unusual_types_test",
            "header": ["string_col", "int_col", "float_col", "bool_col", "null_col", "nested_col"],
            "total_rows": 1,
            "data": [{
                "string_col": "regular string",
                "int_col": 12345,
                "float_col": 123.45,
                "bool_col": True,
                "null_col": None,
                "nested_col": {"nested": "value"}  # Nested object
            }]
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            
            # Should be accepted or rejected based on your API's validation
            assert response.status_code in [
                status.HTTP_201_CREATED,  # Accepted
                status.HTTP_400_BAD_REQUEST,  # Rejected with validation error
                status.HTTP_422_UNPROCESSABLE_ENTITY  # Rejected by FastAPI validation
            ]
    
    def test_extremely_long_values(self, client, test_dataset, mock_token_payload):
        """Test handling of extremely long values"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create a table with extremely long values
        long_string = "a" * 10000  # 10,000 characters
        table_data = {
            "table_name": "long_values_test",
            "header": ["long_col"],
            "total_rows": 1,
            "data": [{
                "long_col": long_string
            }]
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            
            # Should be accepted or rejected based on your API's validation
            assert response.status_code in [
                status.HTTP_201_CREATED,  # Accepted
                status.HTTP_400_BAD_REQUEST,  # Rejected with validation error
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,  # Rejected due to size
                status.HTTP_422_UNPROCESSABLE_ENTITY  # Rejected by FastAPI validation
            ]
    
    # =================== Unusual Request Patterns ===================
    
    def test_repeated_entity_annotations(self, client, test_table_with_data, mock_token_payload):
        """Test repeatedly annotating the same entity"""
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        
        # First, add a new entity
        annotation_data = {
            "entity_id": "repeated_entity",
            "match": True,
            "score": 0.9,
            "candidate_info": {
                "id": "repeated_entity",
                "name": "Repeated Entity",
                "description": "Test entity",
                "types": [{"id": "type1", "name": "Test Type"}]
            }
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # First annotation
            response = client.put(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/0/columns/0",
                json=annotation_data
            )
            assert response.status_code == status.HTTP_200_OK
            
            # Update the same entity multiple times
            for i in range(5):
                # Toggle match status and change score
                annotation_data["match"] = not annotation_data["match"]
                annotation_data["score"] = (i + 1) * 0.1
                
                response = client.put(
                    f"/datasets/{dataset_name}/tables/{table_name}/rows/0/columns/0",
                    json=annotation_data
                )
                assert response.status_code == status.HTTP_200_OK
                
                # Verify the updated match status and score
                entity = response.json()["entity"]
                assert entity["match"] == annotation_data["match"]
                if annotation_data["match"]:
                    assert entity["score"] == annotation_data["score"]
                else:
                    assert entity["score"] is None
    
    def test_rapid_request_sequences(self, client, test_dataset, mock_token_payload):
        """Test sending many requests in quick succession"""
        dataset_name = test_dataset["dataset_name"]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Send multiple GET requests quickly
            for i in range(10):
                response = client.get("/datasets")
                assert response.status_code == status.HTTP_200_OK
            
            # Create multiple tables quickly
            for i in range(5):
                table_data = {
                    "table_name": f"rapid_table_{i}",
                    "header": ["col1", "col2"],
                    "total_rows": 1,
                    "data": [{"col1": f"val1_{i}", "col2": f"val2_{i}"}]
                }
                
                response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
                assert response.status_code == status.HTTP_201_CREATED
    
    def test_concurrent_modifications(self, client, test_table_with_data, mock_token_payload):
        """Simulate concurrent modifications to the same resource"""
        dataset_name = test_table_with_data["dataset_name"]
        table_name = test_table_with_data["table_name"]
        
        # Create two different annotation updates for the same cell
        annotation_1 = {
            "entity_id": "concurrent_entity_1",
            "match": True,
            "score": 0.9,
            "candidate_info": {
                "id": "concurrent_entity_1",
                "name": "Concurrent Entity 1",
                "description": "Test entity 1",
                "types": [{"id": "type1", "name": "Test Type"}]
            }
        }
        
        annotation_2 = {
            "entity_id": "concurrent_entity_2",
            "match": True,
            "score": 0.8,
            "candidate_info": {
                "id": "concurrent_entity_2",
                "name": "Concurrent Entity 2",
                "description": "Test entity 2",
                "types": [{"id": "type1", "name": "Test Type"}]
            }
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Send both requests without waiting for the first to complete
            response_1 = client.put(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/0/columns/0",
                json=annotation_1
            )
            
            response_2 = client.put(
                f"/datasets/{dataset_name}/tables/{table_name}/rows/0/columns/0",
                json=annotation_2
            )
            
            # Both should succeed (the second will overwrite the first)
            assert response_1.status_code == status.HTTP_200_OK
            assert response_2.status_code == status.HTTP_200_OK
            
            # Verify the final state (should be the second entity)
            response = client.get(f"/datasets/{dataset_name}/tables/{table_name}")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            row = next((r for r in data["data"]["rows"] if r["idRow"] == 0), None)
            assert row is not None
            
            if row["linked_entities"]:
                column_0 = next((e for e in row["linked_entities"] if e["idColumn"] == 0), None)
                if column_0 and column_0["candidates"]:
                    matched_entity = next((c for c in column_0["candidates"] if c["match"]), None)
                    if matched_entity:
                        # The second entity should be the one that's matched
                        assert matched_entity["id"] == "concurrent_entity_2"
    
    # =================== Security Tests ===================
    
    def test_injection_attempts(self, client, mock_token_payload):
        """Test protection against injection attempts"""
        # MongoDB injection attempts
        injection_attempts = [
            {"dataset_name": '{"$gt": ""}'},  # NoSQL injection
            {"dataset_name": "'; DROP TABLE datasets; --"},  # SQL injection
            {"dataset_name": '<script>alert("XSS")</script>'},  # XSS attempt
        ]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            for attempt in injection_attempts:
                response = client.post("/datasets", json=attempt)
                
                # Should be either rejected or sanitized
                if response.status_code == status.HTTP_201_CREATED:
                    # If accepted, the injection should be treated as a literal string
                    dataset_name = response.json()["dataset"]["dataset_name"]
                    assert dataset_name == attempt["dataset_name"]
                else:
                    # Or rejected with validation error
                    assert response.status_code in [
                        status.HTTP_400_BAD_REQUEST,
                        status.HTTP_422_UNPROCESSABLE_ENTITY
                    ]
    
    def test_large_request_payloads(self, client, test_dataset, mock_token_payload):
        """Test handling of unusually large request payloads"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create a large table with many rows
        rows = 1000
        columns = 20
        
        header = [f"col{i}" for i in range(columns)]
        data = []
        
        for i in range(rows):
            row = {}
            for j in range(columns):
                row[f"col{j}"] = f"value_{i}_{j}"
            data.append(row)
        
        table_data = {
            "table_name": "large_payload_test",
            "header": header,
            "total_rows": rows,
            "data": data
        }
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            response = client.post(f"/datasets/{dataset_name}/tables/json", json=table_data)
            
            # Should either be accepted or rejected with a size limit error
            assert response.status_code in [
                status.HTTP_201_CREATED,  # Accepted
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE  # Rejected due to size
            ]
    
    # =================== CSV Specific Edge Cases ===================
    
    def test_csv_with_mixed_line_endings(self, client, test_dataset, mock_token_payload, create_csv_file):
        """Test uploading CSV with mixed line endings"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create CSV with mixed line endings (CR, LF, CRLF)
        csv_content = "col1,col2,col3\r"  # CR
        csv_content += "val1,val2,val3\n"  # LF
        csv_content += "val4,val5,val6\r\n"  # CRLF
        
        file = create_csv_file(csv_content)
        
        # Mock pandas read_csv to handle the mixed line endings
        mock_df = pd.DataFrame({
            "col1": ["val1", "val4"],
            "col2": ["val2", "val5"],
            "col3": ["val3", "val6"]
        })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload), \
             patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=mixed_endings_csv",
                files={"file": (file.name, file, "text/csv")}
            )
            
            # Should be accepted
            assert response.status_code == status.HTTP_201_CREATED
    
    def test_csv_with_quoted_fields(self, client, test_dataset, mock_token_payload, create_csv_file):
        """Test uploading CSV with quoted fields containing delimiters"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create CSV with quoted fields containing commas
        csv_content = 'col1,col2,col3\n'
        csv_content += '"value with, comma",val2,val3\n'
        csv_content += 'val4,"value with ""quotes""","value with \n newline"\n'
        
        file = create_csv_file(csv_content)
        
        # Mock pandas read_csv to handle the quoted fields
        mock_df = pd.DataFrame({
            "col1": ["value with, comma", "val4"],
            "col2": ["val2", 'value with "quotes"'],
            "col3": ["val3", "value with \n newline"]
        })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload), \
             patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=quoted_fields_csv",
                files={"file": (file.name, file, "text/csv")}
            )
            
            # Should be accepted
            assert response.status_code == status.HTTP_201_CREATED
    
    def test_csv_with_bom(self, client, test_dataset, mock_token_payload, create_csv_file):
        """Test uploading CSV with BOM (Byte Order Mark)"""
        dataset_name = test_dataset["dataset_name"]
        
        # Create CSV with UTF-8 BOM
        bom = b'\xef\xbb\xbf'  # UTF-8 BOM
        csv_content = bom + b'col1,col2,col3\nval1,val2,val3\nval4,val5,val6'
        
        # Create file with bytes directly
        file = BytesIO(csv_content)
        file.name = "bom_file.csv"
        
        # Mock pandas read_csv to handle the BOM
        mock_df = pd.DataFrame({
            "col1": ["val1", "val4"],
            "col2": ["val2", "val5"],
            "col3": ["val3", "val6"]
        })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload), \
             patch("backend.app.endpoints.crocodile_api.pd.read_csv", return_value=mock_df):
            response = client.post(
                f"/datasets/{dataset_name}/tables/csv?table_name=bom_csv",
                files={"file": (file.name, file, "text/csv")}
            )
            
            # Should be accepted
            assert response.status_code == status.HTTP_201_CREATED
    
    # =================== MongoDB Edge Cases ===================
    
    def test_mongodb_reserved_characters(self, client, mock_token_payload):
        """Test handling of MongoDB reserved characters in IDs"""
        # MongoDB has some restrictions on field names
        problematic_names = [
            "field.with.dots",
            "field$with$dollars",
            "field_with_$leading_dollar",
        ]
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            for name in problematic_names:
                response = client.post("/datasets", json={"dataset_name": name})
                
                # Should be either rejected or sanitized
                assert response.status_code in [
                    status.HTTP_201_CREATED,  # Accepted (MongoDB handles automatically)
                    status.HTTP_400_BAD_REQUEST,  # Rejected with validation error
                    status.HTTP_422_UNPROCESSABLE_ENTITY  # Rejected by FastAPI validation
                ]
    
    def test_string_object_id(self, client, mock_mongodb, mock_token_payload):
        """Test handling endpoints with string versions of ObjectId"""
        # Create a dataset with known ObjectId
        object_id = ObjectId()
        str_id = str(object_id)
        
        mock_mongodb.datasets.insert_one({
            "_id": object_id,
            "user_id": "test@example.com",
            "dataset_name": "object_id_test",
            "created_at": datetime.now(),
            "total_tables": 0,
            "total_rows": 0
        })
        
        with patch("backend.app.dependencies.verify_token", return_value=mock_token_payload):
            # Test pagination with specific ObjectId
            response = client.get(f"/datasets?next_cursor={str_id}")
            
            # Should be accepted
            assert response.status_code == status.HTTP_200_OK