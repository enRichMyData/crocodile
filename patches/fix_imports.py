# patches/fix_imports.py
"""
This script fixes relative imports in the crocodile project.
It creates backup files with .bak extension and modifies the original files.

Usage:
    python patches/fix_imports.py
"""

import os
import re
import shutil
from pathlib import Path

# Define the project root
PROJECT_ROOT = Path(__file__).parent.parent

def backup_file(file_path):
    """Create a backup of the file"""
    backup_path = f"{file_path}.bak"
    print(f"Creating backup: {backup_path}")
    shutil.copy2(file_path, backup_path)

def fix_main_py():
    """Fix imports in main.py"""
    file_path = PROJECT_ROOT / "backend" / "app" / "main.py"
    
    print(f"Processing {file_path}")
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the endpoints import
    new_content = re.sub(
        r'from endpoints\.crocodile_api import router',
        'from backend.app.endpoints.crocodile_api import router',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Fixed imports in {file_path}")

def fix_dependencies_py():
    """Fix imports in dependencies.py"""
    file_path = PROJECT_ROOT / "backend" / "app" / "dependencies.py"
    
    print(f"Processing {file_path}")
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the config import
    new_content = re.sub(
        r'from config import settings',
        'from backend.app.config import settings',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Fixed imports in {file_path}")

def fix_crocodile_api_py():
    """Fix imports in crocodile_api.py"""
    file_path = PROJECT_ROOT / "backend" / "app" / "endpoints" / "crocodile_api.py"
    
    print(f"Processing {file_path}")
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix multiple imports
    replacements = [
        (r'from dependencies import get_crocodile_db, get_db, verify_token', 
         'from backend.app.dependencies import get_crocodile_db, get_db, verify_token'),
        
        (r'from endpoints\.imdb_example import IMDB_EXAMPLE', 
         'from backend.app.endpoints.imdb_example import IMDB_EXAMPLE'),
        
        (r'from schemas import AnnotationUpdate, TableUpload, TableAddResponse', 
         'from backend.app.schemas import AnnotationUpdate, TableUpload, TableAddResponse'),
        
        (r'from services\.data_service import DataService', 
         'from backend.app.services.data_service import DataService'),
        
        (r'from services\.result_sync import ResultSyncService', 
         'from backend.app.services.result_sync import ResultSyncService'),
        
        (r'from services\.utils import sanitize_for_json', 
         'from backend.app.services.utils import sanitize_for_json')
    ]
    
    new_content = content
    for pattern, replacement in replacements:
        new_content = re.sub(pattern, replacement, new_content)
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Fixed imports in {file_path}")

def fix_services_py():
    """Fix imports in data_service.py and result_sync.py"""
    service_files = [
        PROJECT_ROOT / "backend" / "app" / "services" / "data_service.py",
        PROJECT_ROOT / "backend" / "app" / "services" / "result_sync.py"
    ]
    
    for file_path in service_files:
        print(f"Processing {file_path}")
        backup_file(file_path)
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the services.utils import
        new_content = re.sub(
            r'from services\.utils import',
            'from backend.app.services.utils import',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Fixed imports in {file_path}")

def main():
    print("=== Fixing Imports in Crocodile Project ===")
    
    # Create patches directory if it doesn't exist
    os.makedirs(PROJECT_ROOT / "patches", exist_ok=True)
    
    try:
        fix_main_py()
        fix_dependencies_py()
        fix_crocodile_api_py()
        fix_services_py()
        
        print("\n✅ All imports fixed successfully!")
        print("Original files have been backed up with .bak extension")
        print("Run your tests again to see if the imports are working")
    except Exception as e:
        print(f"\n❌ Error fixing imports: {e}")
        print("Some files may not have been fixed properly")

if __name__ == "__main__":
    main()