"""
Root conftest.py — adds the project root to sys.path so tests can import
modules directly (e.g. `from core.classification import ...`) without
requiring the package to be installed as 'dronecmd'.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
