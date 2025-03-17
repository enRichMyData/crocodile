# imdb_example.py
import json
from pathlib import Path

# Option B: Dynamically load from a JSON file at runtime
# (Uncomment if you prefer reading from the actual .json file)
data_path = Path(__file__).parent / "imdb_data.json"
with data_path.open("r", encoding="utf-8") as f:
     IMDB_EXAMPLE = json.load(f)