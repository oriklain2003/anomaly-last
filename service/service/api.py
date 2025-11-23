from __future__ import annotations

from datetime import datetime
import sys
import json
import logging
import sqlite3
import dataclasses
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory (project root) to path so we can import anomaly_pipeline
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Fix DLL load error on Windows by importing torch first
try:
    import torch
except ImportError:
    pass

from flight_fetcher import get, search_flight_path
from anomaly_pipeline import AnomalyPipeline
from core.models import FlightTrack, TrackPoint

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Anomaly Detection Service", description="API for Multi-Layer Flight Anomaly Detection")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files (The Web UI)
static_path = Path("web")
if not static_path.exists():
    static_path.mkdir()
app.mount("/ui", StaticFiles(directory="web"), name="ui")

# Cache DB Configuration
CACHE_DB_PATH = Path("flight_cache.db")
DB_ANOMALIES_PATH = Path("realtime/live_anomalies.db")
DB_TRACKS_PATH = Path("realtime/live_tracks.db")

def setup_cache_db():
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flights (
            flight_id TEXT PRIMARY KEY,
            fetched_at INTEGER,
            data JSON
        )
    """)
    conn.commit()
    conn.close()

# Helper: Serialize FlightTrack to JSON
def serialize_flight(flight: FlightTrack) -> str:
    data = {
        "flight_id": flight.flight_id,
        "points": [dataclasses.asdict(p) for p in flight.points]
    }
    return json.dumps(data)

# Helper: Deserialize JSON to FlightTrack
def deserialize_flight(json_str: str) -> FlightTrack:
    data = json.loads(json_str)
    flight_id = data["flight_id"]
    points = []
    for p_dict in data["points"]:
        points.append(TrackPoint(**p_dict))
    return FlightTrack(flight_id=flight_id, points=points)

# Global Pipeline Instance
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        logger.info("Initializing Global Pipeline...")
        pipeline = AnomalyPipeline()
    return pipeline

@app.on_event("startup")
async def startup_event():
    setup_cache_db()
    get_pipeline()

@app.get("/")
def root():
    # Redirect to UI
    return FileResponse('web/index.html')

class SearchRequest(BaseModel):
    callsign: str
    from_date: datetime
    to_date: datetime

@app.post("/api/search")
def search_flight_endpoint(request: SearchRequest):
    """
    Search for a flight path by callsign and date range.
    """
    pipeline = get_pipeline()
    
    try:
        # Search for flight
        logger.info(f"Searching for {request.callsign} between {request.from_date} and {request.to_date}")
        flight = search_flight_path(request.callsign, request.from_date, request.to_date)
        
        if not flight or not flight.points:
             raise HTTPException(status_code=404, detail=f"No flight found for {request.callsign} in specified range")

        # Save to Cache
        flight_id = flight.flight_id
        conn = sqlite3.connect(str(CACHE_DB_PATH))
        cursor = conn.cursor()
        import time
        cursor.execute(
            "INSERT OR REPLACE INTO flights (flight_id, fetched_at, data) VALUES (?, ?, ?)",
            (flight_id, int(time.time()), serialize_flight(flight))
        )
        conn.commit()
        conn.close()
        logger.info(f"Saved {flight_id} to cache.")
        
        # Run Pipeline
        results = pipeline.analyze(flight)
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/{flight_id}")
def analyze_flight_endpoint(flight_id: str):
    """
    Analyze a flight by ID using the AnomalyPipeline.
    Checks local cache first, then fetches from FR24.
    """
    pipeline = get_pipeline()
    
    try:
        flight = None
        
        # 1. Check Cache
        conn = sqlite3.connect(str(CACHE_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            logger.info(f"Cache HIT for {flight_id}")
            flight = deserialize_flight(row[0])
        else:
            logger.info(f"Cache MISS for {flight_id}. Fetching live...")
            # Fetch Flight Data
            flight = get(flight_id=flight_id)
            
            if flight and flight.points:
                # Save to Cache
                conn = sqlite3.connect(str(CACHE_DB_PATH))
                cursor = conn.cursor()
                import time
                cursor.execute(
                    "INSERT OR REPLACE INTO flights (flight_id, fetched_at, data) VALUES (?, ?, ?)",
                    (flight_id, int(time.time()), serialize_flight(flight))
                )
                conn.commit()
                conn.close()
                logger.info(f"Saved {flight_id} to cache.")
        
        if not flight or not flight.points:
             raise HTTPException(status_code=404, detail=f"Flight data not found for {flight_id}")

        # 2. Run Pipeline
        results = pipeline.analyze(flight)
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/live/anomalies")
def get_live_anomalies(start_ts: int, end_ts: int):
    """
    Fetch anomalies from the live realtime database within a time range.
    """
    if not DB_ANOMALIES_PATH.exists():
        return []
        
    try:
        conn = sqlite3.connect(str(DB_ANOMALIES_PATH))
        cursor = conn.cursor()
        
        query = """
            SELECT flight_id, timestamp, is_anomaly, severity_cnn, severity_dense, full_report 
            FROM anomaly_reports 
            WHERE timestamp BETWEEN ? AND ? AND is_anomaly = 1
            ORDER BY timestamp DESC
        """
        
        cursor.execute(query, (start_ts, end_ts))
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        # Create a connection to the tracks DB to fetch callsigns efficiently
        conn_tracks = None
        if DB_TRACKS_PATH.exists():
            try:
                conn_tracks = sqlite3.connect(str(DB_TRACKS_PATH))
            except:
                pass

        for row in rows:
            # Parse full report if it's a string
            report = row[5]
            if isinstance(report, str):
                try:
                    report = json.loads(report)
                except:
                    pass
            
            flight_id = row[0]
            callsign = None
            
            # Try to fetch callsign from tracks DB
            if conn_tracks:
                try:
                    cursor_tracks = conn_tracks.cursor()
                    cursor_tracks.execute("SELECT callsign FROM flight_tracks WHERE flight_id = ? AND callsign IS NOT NULL AND callsign != '' LIMIT 1", (flight_id,))
                    track_row = cursor_tracks.fetchone()
                    if track_row and track_row[0]:
                        callsign = track_row[0]
                except:
                    pass
            
            # Fallback: Try to get callsign from the report summary itself (if available)
            if not callsign and isinstance(report, dict):
                 callsign = report.get("summary", {}).get("callsign")

            results.append({
                "flight_id": flight_id,
                "timestamp": row[1],
                "is_anomaly": bool(row[2]),
                "severity_cnn": row[3],
                "severity_dense": row[4],
                "full_report": report,
                "callsign": callsign
            })
        
        if conn_tracks:
            conn_tracks.close()
            
        return results
    except Exception as e:
        logger.error(f"Failed to fetch live anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/live/track/{flight_id}")
def get_live_track(flight_id: str):
    """
    Fetch the full track for a flight from the live tracks database.
    """
    if not DB_TRACKS_PATH.exists():
        raise HTTPException(status_code=404, detail="Live tracks DB not found")
        
    try:
        conn = sqlite3.connect(str(DB_TRACKS_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM flight_tracks 
            WHERE flight_id = ? 
            ORDER BY timestamp ASC
        """
        
        cursor.execute(query, (flight_id,))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            raise HTTPException(status_code=404, detail="Track not found")
            
        points = []
        for row in rows:
            points.append({
                "lat": row["lat"],
                "lon": row["lon"],
                "alt": row["alt"],
                "timestamp": row["timestamp"],
                "gspeed": row["gspeed"],
                "track": row["track"],
                "flight_id": row["flight_id"]
            })
            
        return {
            "flight_id": flight_id,
            "points": points
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch live track: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from training_ops.db_utils import save_feedback

@app.post("/api/feedback")
def submit_feedback(feedback: dict):
    """
    Submit user feedback for a flight.
    Payload: {
        "flight_id": "...",
        "is_anomaly": true/false,
        "comments": "..."
    }
    """
    flight_id = feedback.get("flight_id")
    is_anomaly = feedback.get("is_anomaly")
    comments = feedback.get("comments", "")
    
    if not flight_id or is_anomaly is None:
        raise HTTPException(status_code=400, detail="Missing flight_id or is_anomaly")
        
    logger.info(f"Received feedback for {flight_id}: Anomaly={is_anomaly}")
    
    # 1. Find Flight Data (Check Cache first, then Live DB)
    points = []
    
    # Check Cache
    try:
        conn = sqlite3.connect(str(CACHE_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = json.loads(row[0])
            points = data.get("points", [])
    except Exception as e:
        logger.error(f"Cache lookup error: {e}")

    # Check Live DB if not in cache
    if not points and DB_TRACKS_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_TRACKS_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM flight_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()
            conn.close()
            if rows:
                # Convert rows to dicts
                points = [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Live DB lookup error: {e}")
            
    if not points:
        raise HTTPException(status_code=404, detail="Flight data not found in cache or live DB. Cannot save feedback.")

    # 2. Save to Training DB
    try:
        save_feedback(flight_id, is_anomaly, points, comments)
        
        # If user says it's NOT an anomaly, remove it from the live anomalies view (DB)
        if is_anomaly is False:
            try:
                if DB_ANOMALIES_PATH.exists():
                    conn = sqlite3.connect(str(DB_ANOMALIES_PATH))
                    cursor = conn.cursor()
                    # We set is_anomaly=0 to "soft delete" it from the live view query which filters is_anomaly=1
                    cursor.execute("UPDATE anomaly_reports SET is_anomaly = 0 WHERE flight_id = ?", (flight_id,))
                    conn.commit()
                    conn.close()
                    logger.info(f"Removed {flight_id} from live anomalies view (soft delete).")
            except Exception as db_e:
                logger.error(f"Failed to update anomaly status in live DB: {db_e}")
                
    except Exception as e:
        logger.error(f"Save feedback failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")
        
    return {"status": "success", "message": "Feedback saved and flight added to training dataset"}

if __name__ == "__main__":
    import uvicorn
    # Run from root with: python service/api.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
