import runpy
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "Twitter-Sentiment-Analysis"
if not PROJECT_DIR.exists():
    raise FileNotFoundError("Project folder not found: " + str(PROJECT_DIR))

sys.path.insert(0, str(PROJECT_DIR))
original_argv = list(sys.argv)
sys.argv = [str(PROJECT_DIR / "predict.py"), *original_argv[1:]]
try:
    runpy.run_path(str(PROJECT_DIR / "predict.py"), run_name="__main__")
finally:
    sys.argv = original_argv
