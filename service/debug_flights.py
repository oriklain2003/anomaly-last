import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from core.db import DbConfig, FlightRepository

def check_flights():
    db_path = Path("last.db")
    repo = FlightRepository(DbConfig(path=db_path, table="anomalous_tracks"))
    flights = list(repo.iter_flights(limit=5, min_points=50))
    
    print(f"Loaded {len(flights)} anomalous flights.")
    for f in flights:
        print(f"Flight {f.flight_id}: {len(f.points)} points.")

if __name__ == "__main__":
    check_flights()

