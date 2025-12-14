import sqlite3
import json
import time
from pathlib import Path
import sys
import dataclasses
import logging

# Fix DLL error by importing torch first if possible, or just skip it if we can't load it.
# However, AnomalyPipeline imports it at top level.
# We can try to set the dll directory or just wrap the import.
# Actually, the service/api.py does:
# try:
#    import torch
# except ImportError:
#    pass

# Let's try to emulate that structure BEFORE importing anomaly_pipeline
try:
    import torch
except Exception:
    pass

# Add parent directory to path
root_path = str(Path(__file__).resolve().parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Now import pipeline
try:
    from anomaly_pipeline import AnomalyPipeline
except OSError as e:
    print(f"Failed to import AnomalyPipeline due to DLL error: {e}")
    # Fallback: We can't run the full pipeline without torch if the detectors need it.
    # But maybe we can mock it or run a partial version?
    # The user wants to run the pipeline. If the environment is broken for torch, we might be stuck.
    # BUT, the service runs fine? Let's check service/api.py.
    # It imports torch first.
    sys.exit(1)

from core.models import FlightTrack, TrackPoint

# Configuration
CACHE_DB_PATH = "flight_cache.db"
DB_ANOMALIES_PATH = "realtime/live_anomalies.db"
DB_TRACKS_PATH = "realtime/live_tracks.db"
FLIGHT_ID = "3bc6854c"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper to deserialize
def deserialize_flight(json_str: str) -> FlightTrack:
    data = json.loads(json_str)
    flight_id = data["flight_id"]
    points = []
    for p_dict in data["points"]:
        points.append(TrackPoint(**p_dict))
    return FlightTrack(flight_id=flight_id, points=points)

def cleanup_old_entry():
    logger.info(f"Cleaning up old entry for {FLIGHT_ID}...")
    if Path(DB_ANOMALIES_PATH).exists():
        try:
            conn = sqlite3.connect(DB_ANOMALIES_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM anomaly_reports WHERE flight_id = ?", (FLIGHT_ID,))
            conn.commit()
            conn.close()
            logger.info("Cleanup complete.")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

def save_track(flight: FlightTrack):
    logger.info(f"Saving track for {flight.flight_id} to live_tracks.db...")
    
    if not Path(DB_TRACKS_PATH).exists():
        # Create DB if not exists
        conn = sqlite3.connect(DB_TRACKS_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flight_tracks (
                flight_id TEXT,
                timestamp INTEGER,
                lat REAL,
                lon REAL,
                alt REAL,
                gspeed REAL,
                track REAL,
                callsign TEXT,
                PRIMARY KEY (flight_id, timestamp)
            )
        """)
        conn.commit()
        conn.close()

    conn = sqlite3.connect(DB_TRACKS_PATH)
    cursor = conn.cursor()
    
    # First clean old track for this flight
    cursor.execute("DELETE FROM flight_tracks WHERE flight_id = ?", (flight.flight_id,))
    
    count = 0
    for p in flight.points:
        cursor.execute("""
            INSERT OR REPLACE INTO flight_tracks 
            (flight_id, timestamp, lat, lon, alt, gspeed, track, callsign)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (flight.flight_id, p.timestamp, p.lat, p.lon, p.alt, p.gspeed, p.track, p.callsign))
        count += 1
        
    conn.commit()
    conn.close()
    logger.info(f"Saved {count} points.")

def run_pipeline_and_save():
    # 1. Load Flight
    logger.info(f"Loading flight {FLIGHT_ID} from cache...")
    if not Path(CACHE_DB_PATH).exists():
        logger.error("Cache DB not found.")
        return

    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (FLIGHT_ID,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        logger.error(f"Flight {FLIGHT_ID} not found in cache.")
        return

    flight = deserialize_flight(row[0])
    logger.info(f"Loaded flight with {len(flight.points)} points.")

    # 2. Run Pipeline
    logger.info("Initializing pipeline...")
    pipeline = AnomalyPipeline()
    
    logger.info("Analyzing flight...")
    report = pipeline.analyze(flight)
    
    is_anomaly = report["summary"]["is_anomaly"]
    logger.info(f"Analysis complete. Is Anomaly? {is_anomaly}")
    
    # 3. Save Results
    now_ts = int(time.time())
    
    # Save Track Points first (needed for UI visualization)
    save_track(flight)
    
    # Save Anomaly Report
    logger.info("Saving report to live_anomalies.db...")
    conn = sqlite3.connect(DB_ANOMALIES_PATH)
    cursor = conn.cursor()
    
    # Extract severities (safely)
    sev_cnn = 0.0
    if "layer_4_deep_cnn" in report and "anomaly_score" in report["layer_4_deep_cnn"]:
         sev_cnn = float(report["layer_4_deep_cnn"]["anomaly_score"])
         
    sev_dense = 0.0
    if "layer_3_deep_dense" in report and "reconstruction_error" in report["layer_3_deep_dense"]:
        sev_dense = float(report["layer_3_deep_dense"]["reconstruction_error"])

    cursor.execute("""
        INSERT INTO anomaly_reports (flight_id, timestamp, is_anomaly, severity_cnn, severity_dense, full_report)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (FLIGHT_ID, now_ts, 1 if is_anomaly else 0, sev_cnn, sev_dense, json.dumps(report)))
    
    conn.commit()
    conn.close()
    logger.info("Done.")

if __name__ == "__main__":
    cleanup_old_entry()
    run_pipeline_and_save()
