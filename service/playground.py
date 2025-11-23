from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path

from core.db import DbConfig, FlightRepository
from core.models import FlightTrack, TrackPoint
from rules.rule_engine import AnomalyRuleEngine

from fr24sdk.client import Client
from fr24sdk.models.geographic import Boundary

# Boundaries
MIN_LAT = 29.53523
MAX_LAT = 33.614619
MIN_LON = 34.145508
MAX_LON = 36.386719

# FlightRadar24 API Token

# Track active flights
active_flights = {}
# Boundary format: north, south, west, east
boundary = Boundary(
    north=MAX_LAT,
    south=MIN_LAT,
    west=MIN_LON,
    east=MAX_LON
)
client = Client(api_token="019a9d54-678a-73da-a927-1f7a60b27a8f|xouEuixtJFLIlwNKajdW0mFjV4sguJsf5eByvIYTde7f4b26")

response = client.live.flight_positions.get_full(bounds=boundary, altitude_ranges=["1000-50000"])


def get():
    from fr24sdk.client import Client

    client = Client(api_token="019a9d54-678a-73da-a927-1f7a60b27a8f|xouEuixtJFLIlwNKajdW0mFjV4sguJsf5eByvIYTde7f4b26")
    response = client.live.flight_positions.get_full(bounds=boundary, altitude_ranges=["1000-50000"])

    tracks = client.flight_tracks.get(flight_id=["3bc6854c"])
    # Get first FlightTracks object
    flight_data = tracks.model_dump()["data"][0]
    flight_id = flight_data["fr24_id"]
    track_points = flight_data["tracks"]
    
    # Convert to TrackPoint objects
    points = []
    for tp in track_points:
        # Parse ISO timestamp to unix timestamp
        ts_str = tp["timestamp"].replace("Z", "+00:00")
        ts = int(datetime.fromisoformat(ts_str).timestamp())
        
        points.append(TrackPoint(
            flight_id=flight_id,
            timestamp=ts,
            lat=float(tp["lat"]),
            lon=float(tp["lon"]),
            alt=float(tp["alt"]),
            gspeed=float(tp["gspeed"]) if tp.get("gspeed") is not None else None,
            vspeed=float(tp["vspeed"]) if tp.get("vspeed") is not None else None,
            track=float(tp["track"]) if tp.get("track") is not None else None,
            squawk=str(tp["squawk"]) if tp.get("squawk") else None,
            callsign=tp.get("callsign"),
            source=tp.get("source"),
        ))
    
    return FlightTrack(flight_id=flight_id, points=points)


if __name__ == '__main__':
    # Get flight track
    track = get()
    
    # Create temp DB
    db_path = Path("temp_playground.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE flight_tracks (
            flight_id TEXT, timestamp INTEGER, lat REAL, lon REAL, alt REAL,
            gspeed REAL, vspeed REAL, track REAL, squawk TEXT, callsign TEXT, source TEXT
        )
    """)
    
    for p in track.points:
        cursor.execute("INSERT INTO flight_tracks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (p.flight_id, p.timestamp, p.lat, p.lon, p.alt,
                       p.gspeed, p.vspeed, p.track, p.squawk, p.callsign, p.source))
    conn.commit()
    conn.close()
    
    # Run rules
    rules_path = Path("anomaly_rule.json")
    repository = FlightRepository(DbConfig(path=db_path))
    engine = AnomalyRuleEngine(repository, rules_path)
    report = engine.evaluate_flight(track.flight_id)
    
    print(json.dumps(report["matched_rules"], indent=2, ensure_ascii=False))
    
    # Cleanup
    db_path.unlink()
