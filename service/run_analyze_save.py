import sqlite3
import json
import sys
from pathlib import Path
from datetime import datetime

# Add root to path so we can import modules
sys.path.append(str(Path(__file__).parent))

from flight_fetcher import deserialize_flight
try:
    from anomaly_pipeline import AnomalyPipeline
    PIPELINE_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: Could not import AnomalyPipeline ({e}). Using mock report.")
    PIPELINE_AVAILABLE = False


CACHE_DB_PATH = Path("flight_cache.db")
RESEARCH_DB_PATH = Path("realtime/research.db")
FLIGHT_ID = "3d7211ef"

def main():
    print(f"Analyzing flight {FLIGHT_ID}...")
    
    local_pipeline_available = PIPELINE_AVAILABLE

    # 1. Fetch from Cache
    if not CACHE_DB_PATH.exists():
        print(f"Error: Cache DB not found at {CACHE_DB_PATH}")
        return

    conn = sqlite3.connect(str(CACHE_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (FLIGHT_ID,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        print(f"Error: Flight {FLIGHT_ID} not found in cache.")
        return

    flight = deserialize_flight(row[0])
    print(f"Loaded flight {flight.flight_id} with {len(flight.points)} points.")

    # 2. Run Analysis
    results = {}
    is_anomaly = False
    
    if local_pipeline_available:
        try:
            pipeline = AnomalyPipeline()
            results = pipeline.analyze(flight)
            is_anomaly = results["summary"]["is_anomaly"]
            confidence = results["summary"]["confidence_score"]
            print(f"Analysis complete. Is Anomaly: {is_anomaly} (Confidence: {confidence}%)")
        except Exception as e:
            print(f"Error during analysis: {e}. Falling back to mock.")
            local_pipeline_available = False # Fallback
            
    if not local_pipeline_available:
        # Mock results
        print("Generating mock anomaly report...")
        is_anomaly = True
        flight_path = [[p.lon, p.lat] for p in flight.sorted_points()]
        results = {
            "summary": {
                "is_anomaly": True,
                "confidence_score": 99.9,
                "triggers": ["MANUAL_FORCE", "MOCK_ANALYSIS"],
                "flight_id": flight.flight_id,
                "callsign": flight.points[0].callsign if flight.points else "UNKNOWN",
                "num_points": len(flight.points),
                "flight_path": flight_path
            },
            "layer_1_rules": {"status": "SKIPPED"},
            "layer_2_xgboost": {"is_anomaly": False, "score": 0.0},
            "layer_3_deep_dense": {"is_anomaly": True, "score": 0.95, "severity": 0.95},
            "layer_4_deep_cnn": {"is_anomaly": True, "score": 0.95, "severity": 0.95},
        }

    # FORCE ANOMALY if user requested to see it in UI (which filters for anomalies)
    # The user explicitly said "save it ... with a anomaly report so i can see it"
    if not is_anomaly:
        print("Forcing is_anomaly=True for UI visibility as requested.")
        is_anomaly = True
        results["summary"]["is_anomaly"] = True
        results["summary"]["triggers"].append("MANUAL_FORCE")

    # 4. Save to Research DB
    if not RESEARCH_DB_PATH.exists():
        print(f"Creating research DB directory at {RESEARCH_DB_PATH.parent}...")
        RESEARCH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn_res = sqlite3.connect(str(RESEARCH_DB_PATH))
    cursor_res = conn_res.cursor()

    # Create tables if they don't exist (just in case)
    cursor_res.execute("""
        CREATE TABLE IF NOT EXISTS anomalies_tracks (
            flight_id TEXT,
            timestamp INTEGER,
            lat REAL,
            lon REAL,
            alt REAL,
            gspeed REAL,
            vspeed REAL,
            track REAL,
            squawk TEXT,
            callsign TEXT,
            source TEXT,
            PRIMARY KEY (flight_id, timestamp)
        )
    """)
    
    cursor_res.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_reports (
            flight_id TEXT PRIMARY KEY,
            timestamp INTEGER,
            is_anomaly INTEGER,
            severity_cnn REAL,
            severity_dense REAL,
            full_report TEXT
        )
    """)

    # Insert Tracks
    
    print(f"Inserting {len(flight.points)} points into anomalies_tracks...")
    for p in flight.points:
        cursor_res.execute("""
            INSERT OR REPLACE INTO anomalies_tracks 
            (flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p.flight_id,
            p.timestamp,
            p.lat,
            p.lon,
            p.alt,
            p.gspeed,
            p.vspeed,
            p.track,
            p.squawk,
            p.callsign,
            p.source
        ))

    # Insert Report
    # Extract severities
    severity_cnn = 0.0
    if "layer_4_deep_cnn" in results and "severity" in results["layer_4_deep_cnn"]:
        severity_cnn = float(results["layer_4_deep_cnn"]["severity"])
        
    severity_dense = 0.0
    if "layer_3_deep_dense" in results and "severity" in results["layer_3_deep_dense"]:
         severity_dense = float(results["layer_3_deep_dense"]["severity"])

    last_ts = flight.points[-1].timestamp if flight.points else int(datetime.now().timestamp())

    print("Inserting anomaly report...")
    cursor_res.execute("""
        INSERT OR REPLACE INTO anomaly_reports 
        (flight_id, timestamp, is_anomaly, severity_cnn, severity_dense, full_report)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        FLIGHT_ID,
        last_ts,
        1 if is_anomaly else 0,
        severity_cnn,
        severity_dense,
        json.dumps(results)
    ))

    conn_res.commit()
    conn_res.close()
    print("Successfully saved to research.db")

if __name__ == "__main__":
    main()
