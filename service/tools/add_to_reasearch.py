import sqlite3
import json
from datetime import datetime
import time

CACHE_DB = "../flight_cache.db"
RESEARCH_DB = "../realtime/research.db"
FLIGHT_ID = "3ad71404"


def copy_flight():
    # 1. Get from Cache
    conn_cache = sqlite3.connect(CACHE_DB)
    cursor_cache = conn_cache.cursor()
    cursor_cache.execute("SELECT data FROM flights WHERE flight_id = ?", (FLIGHT_ID,))
    row = cursor_cache.fetchone()
    conn_cache.close()

    if not row:
        print(f"Flight {FLIGHT_ID} not found in cache.")
        return

    data = json.loads(row[0])
    points = data.get("points", [])

    if not points:
        print("No points in flight data.")
        return

    # Calculate Date
    first_ts = points[0].get("timestamp")
    flight_date = datetime.fromtimestamp(first_ts).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Flight Date: {flight_date}")

    # 2. Insert into Research DB
    conn_research = sqlite3.connect(RESEARCH_DB)
    cursor_research = conn_research.cursor()

    # Insert Tracks
    print(f"Inserting {len(points)} points into anomalies_tracks...")
    for p in points:
        cursor_research.execute("""
            INSERT OR REPLACE INTO anomalies_tracks 
            (flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p.get("flight_id"),
            p.get("timestamp"),
            p.get("lat"),
            p.get("lon"),
            p.get("alt"),
            p.get("gspeed"),
            p.get("vspeed"),
            p.get("track"),
            p.get("squawk"),
            p.get("callsign"),
            p.get("source")
        ))

    # Insert Report
    # Use the last timestamp for the report
    last_ts = points[-1].get("timestamp")

    report = {
        "summary": {
            "flight_id": FLIGHT_ID,
            "callsign": points[0].get("callsign", "UNKNOWN"),
            "is_anomaly": True,
            "reason": "Manual Test Injection"
        },
        "details": "Manually injected from cache for testing."
    }

    print("Inserting anomaly report...")
    cursor_research.execute("""
        INSERT OR REPLACE INTO anomaly_reports 
        (flight_id, timestamp, is_anomaly, severity_cnn, severity_dense, full_report)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        FLIGHT_ID,
        last_ts,
        1,  # True
        0.95,  # High severity
        0.95,
        json.dumps(report)
    ))

    conn_research.commit()
    conn_research.close()
    print("Done.")


if __name__ == "__main__":
    copy_flight()
