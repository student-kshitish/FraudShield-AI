import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
AIML_DIR = PROJECT_ROOT / "AIML-01"

sys.path.insert(0, str(AIML_DIR))

from phase09_fastapi_server import app

