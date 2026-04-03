"""
Database package - re-exports from database.py to avoid import conflicts
"""
import sys
import os
import importlib.util

# Get parent directory (birsakisan-backend)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
database_py_path = os.path.join(parent_dir, "database.py")

# Load database.py as a module
if os.path.exists(database_py_path):
    spec = importlib.util.spec_from_file_location("_database_file", database_py_path)
    _database_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_database_file)
    
    # Re-export the functions and db object
    init_db = _database_file.init_db
    close_db = _database_file.close_db
    db = _database_file.db
else:
    raise ImportError(f"Could not find database.py at {database_py_path}")

