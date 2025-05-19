# patch_imports.py
"""
This script patches imports for testing by adding mock modules to sys.modules.
Import this at the beginning of your test files.

Usage:
    import patch_imports  # This should be the first import
"""

import sys
from unittest.mock import MagicMock
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Create mock for crocodile module with CrocodileResultFetcher
class CrocodileResultFetcher:
    """Mock implementation of CrocodileResultFetcher"""
    
    def __init__(self, *args, **kwargs):
        """Initialize with any arguments"""
        self.args = args
        self.kwargs = kwargs
    
    def fetch_results(self, *args, **kwargs):
        """Mock implementation of fetch_results"""
        return []
    
    def get_results(self, *args, **kwargs):
        """Mock implementation of get_results"""
        return []

class Crocodile:
    """Mock implementation of Crocodile"""
    
    def __init__(self, *args, **kwargs):
        """Initialize with any arguments"""
        self.input_csv = kwargs.get('input_csv')
        self.client_id = kwargs.get('client_id')
        self.dataset_name = kwargs.get('dataset_name')
        self.table_name = kwargs.get('table_name')
        self.columns_type = kwargs.get('columns_type', {})
        self.save_output_to_csv = kwargs.get('save_output_to_csv', False)
    
    def run(self):
        """Mock implementation of run method"""
        # Just return without doing anything
        return

try:
    # Try to import the actual crocodile module
    import crocodile
    
    # Check if CrocodileResultFetcher exists
    if not hasattr(crocodile, 'CrocodileResultFetcher'):
        # If not, add it to the module
        crocodile.CrocodileResultFetcher = CrocodileResultFetcher
        
    # Make sure Crocodile is available
    if not hasattr(crocodile, 'Crocodile'):
        crocodile.Crocodile = Crocodile
except ImportError:
    # If crocodile module can't be imported, create a mock
    # Create a mock crocodile module
    mock_crocodile = MagicMock()
    mock_crocodile.CrocodileResultFetcher = CrocodileResultFetcher
    mock_crocodile.Crocodile = Crocodile
    
    # Add the mock module to sys.modules
    sys.modules['crocodile'] = mock_crocodile

# Set required environment variables for testing
os.environ.setdefault("JWT_SECRET_KEY", "test_secret_key_for_testing")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("ENTITY_RETRIEVAL_ENDPOINT", "http://mock-endpoint/retrieve")
os.environ.setdefault("ENTITY_BOW_ENDPOINT", "http://mock-endpoint/bow")
os.environ.setdefault("ENTITY_RETRIEVAL_TOKEN", "mock-token")