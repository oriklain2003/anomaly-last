import sqlite3
import json
import sys
from pathlib import Path

# Add root to path to import existing modules
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
from flight_fetcher import deserialize_flight
from new_service.glitch_analyzer import analyze_glitches

CACHE_DB_PATH = Path("flight_cache.db")

def inspect_flight(flight_id):
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print(f"Flight {flight_id} not found in cache.")
        return

    flight = deserialize_flight(row[0])
    
    print(f"Running NEW glitch analyzer on {flight_id}...")
    report = analyze_glitches(flight)
    
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    inspect_flight("3ac7b198")
