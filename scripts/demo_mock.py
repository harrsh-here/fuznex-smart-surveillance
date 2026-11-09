"""Show mock template loading. No face recognition is performed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.pipeline import RecognitionSystem

if __name__ == "__main__":
    gallery = RecognitionSystem().load_templates()
    print(f"Loaded {len(gallery)} synthetic templates (mock only).")
    print("Real face recognition has not been implemented in this scaffold.")
