from dotenv import load_dotenv
load_dotenv()
import os
import requests
from pathlib import Path
import json

NASA_API_KEY = os.getenv("NASA_API_KEY")

r = requests.get(f"https://api.nasa.gov/neo/rest/v1/neo/browse?page=0&size=20&api_key={NASA_API_KEY}")

data = dict(r.json())
data_dir = Path.cwd() / "data"
data_dir.mkdir(exist_ok=True)
file_path = data_dir / "asteroids.json"

with open(file_path, "w") as f:
    json.dump(data, f)