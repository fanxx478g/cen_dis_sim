import sys
from pathlib import Path


# Ensure tests can import the local package when the project is not installed.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
