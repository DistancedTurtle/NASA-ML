from dotenv import load_dotenv
load_dotenv()
import os
import requests
import time
from pathlib import Path
import json

NASA_API_KEY = os.getenv("NASA_API_KEY")

data_dir = Path.cwd() / "data"
data_dir.mkdir(exist_ok=True)
file_path = data_dir / "near_earth_objects.json"

if file_path.exists():
    with open(file_path) as f:
        master_neo_dict = json.load(f)
else:
    master_neo_dict = {}

start_page = len(master_neo_dict) // 20
url = f"https://api.nasa.gov/neo/rest/v1/neo/browse?page={start_page}&size=20&api_key={NASA_API_KEY}"

while url:
    try:
        response = requests.get(url, timeout=30)
    except requests.exceptions.ConnectionError:
        print("Connection error. Waiting 30 seconds and retrying...")
        time.sleep(30)
        continue

    if response.status_code == 429:
        print("Rate limit reached. Waiting 5 minutes...")
        time.sleep(300)
        continue

    elif response.status_code != 200:
        print(f"Error {response.status_code}. Stopping.")
        break

    data = response.json()
    page_neos = data.get("near_earth_objects", [])

    for neo in page_neos:
        neo_id = neo.get("id")
        master_neo_dict[neo_id] = neo

    url = data.get("links", {}).get("next")

    with open(file_path, "w") as f:
        json.dump(master_neo_dict, f)
