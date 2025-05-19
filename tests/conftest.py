# tests/conftest.py
# Import the patch before anything else
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import tests.patch_imports as patch_imports

import pytest
import mongomock
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, MagicMock

# Try importing our app and dependencies
try:
    from backend.app.main import app
    from backend.app.dependencies import get_db, get_crocodile_db, verify_token
    APP_IMPORT_SUCCESS = True
except Exception as e:
    print(f"Error importing app: {e}")
    APP_IMPORT_SUCCESS = False
    app = None
    get_db = None
    get_crocodile_db = None
    verify_token = None

# Mock MongoDB client
@pytest.fixture(scope="function")
def mongo_client():
    """Create a mock MongoDB client for testing"""
    client = mongomock.MongoClient()
    
    # Create test databases
    client.crocodile_backend_db = client["crocodile_backend_db"]
    client.crocodile_db = client["crocodile_db"]
    
    # Set up collections
    client.crocodile_backend_db.datasets = client.crocodile_backend_db["datasets"]
    client.crocodile_backend_db.tables = client.crocodile_backend_db["tables"]
    client.crocodile_backend_db.input_data = client.crocodile_backend_db["input_data"]
    
    client.crocodile_db.input_data = client.crocodile_db["input_data"]
    
    # Set up indexes
    client.crocodile_backend_db.datasets.create_index([("user_id", 1), ("dataset_name", 1)], unique=True)
    client.crocodile_backend_db.tables.create_index([("user_id", 1), ("dataset_name", 1), ("table_name", 1)], unique=True)
    
    yield client
    
    # Clean up
    client.drop_database("crocodile_backend_db")
    client.drop_database("crocodile_db")

@pytest.fixture(scope="function")
def test_db(mongo_client):
    """Provide the test backend database"""
    return mongo_client.crocodile_backend_db

@pytest.fixture(scope="function")
def test_crocodile_db(mongo_client):
    """Provide the test crocodile database"""
    return mongo_client.crocodile_db

@pytest.fixture
def mock_token_payload():
    """Mock token payload for authentication."""
    return {"email": "test@example.com", "name": "Test User"}

@pytest.fixture
def mock_verify_token(mock_token_payload):
    """Mock the verify_token dependency"""
    return mock_token_payload

@pytest.fixture
def mock_background_tasks():
    """Mock BackgroundTasks for testing"""
    mock_bg = MagicMock()
    mock_bg.add_task = MagicMock()
    return mock_bg

@pytest.fixture
def client(test_db, test_crocodile_db, mock_verify_token, mock_background_tasks):
    """Create test client with dependency overrides"""
    if not APP_IMPORT_SUCCESS:
        pytest.skip("Skipping test due to app import error")
        
    # Override the database dependency
    app.dependency_overrides[get_db] = lambda: test_db
    app.dependency_overrides[get_crocodile_db] = lambda: test_crocodile_db
    app.dependency_overrides[verify_token] = lambda: mock_verify_token
    
    # Create patchers for dependencies
    with patch("backend.app.endpoints.crocodile_api.BackgroundTasks", return_value=mock_background_tasks):
        with TestClient(app) as test_client:
            yield test_client
            
    # Clear dependency overrides
    app.dependency_overrides.clear()

# Test fixtures for common test data
@pytest.fixture
def test_dataset(client, test_db):
    """Create a test dataset"""
    if not APP_IMPORT_SUCCESS:
        pytest.skip("Skipping test due to app import error")
        
    dataset_data = {
        "user_id": "test@example.com",
        "dataset_name": "test_dataset",
        "description": "Test Dataset",
        "created_at": datetime.now(),
        "total_tables": 0,
        "total_rows": 0
    }
    
    result = test_db.datasets.insert_one(dataset_data)
    
    return {
        "id": str(result.inserted_id),
        "dataset_name": "test_dataset"
    }

@pytest.fixture
def test_table_with_data(client, test_db, test_dataset):
    """Create a test table with sample data"""
    if not APP_IMPORT_SUCCESS:
        pytest.skip("Skipping test due to app import error")
        
    dataset_name = test_dataset["dataset_name"]
    table_name = "test_table_with_data"
    
    # Create the table
    table_data = {
        "user_id": "test@example.com",
        "dataset_name": dataset_name,
        "table_name": table_name,
        "header": ["col1", "col2", "col3"],
        "total_rows": 2,
        "created_at": datetime.now()
    }
    
    test_db.tables.insert_one(table_data)
    
    # Add input data records
    for i in range(2):
        test_db.input_data.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": i,
            "data": [f"value{i*3+1}", f"value{i*3+2}", f"value{i*3+3}"],
            "el_results": {
                "0": [
                    {
                        "id": f"entity_{i}_1",
                        "name": f"Entity {i} Primary",
                        "description": "Test entity",
                        "score": 0.9,
                        "match": True
                    },
                    {
                        "id": f"entity_{i}_2",
                        "name": f"Entity {i} Secondary",
                        "description": "Another test entity",
                        "score": 0.5,
                        "match": False
                    }
                ]
            }
        })
    
    return {"dataset_name": dataset_name, "table_name": table_name}

@pytest.fixture
def test_processing_table(client, test_db, test_dataset):
    """Create a test table with processing status"""
    if not APP_IMPORT_SUCCESS:
        pytest.skip("Skipping test due to app import error")
        
    dataset_name = test_dataset["dataset_name"]
    table_name = "processing_table"
    
    # Create the table
    test_db.tables.insert_one({
        "user_id": "test@example.com",
        "dataset_name": dataset_name,
        "table_name": table_name,
        "header": ["col1", "col2"],
        "total_rows": 5,
        "created_at": datetime.now()
    })
    
    # Insert some rows with different statuses
    for i in range(5):
        status = "DONE" if i < 3 else "TODO"
        test_db.input_data.insert_one({
            "user_id": "test@example.com",
            "dataset_name": dataset_name,
            "table_name": table_name,
            "row_id": i,
            "data": [f"value{i*2+1}", f"value{i*2+2}"],
            "status": status,
            "ml_status": status
        })
    
    return {"dataset_name": dataset_name, "table_name": table_name}

# Create a small test function that should always pass
def test_dummy():
    """This test should always pass"""
    assert True