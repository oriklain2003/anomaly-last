from __future__ import annotations

from datetime import datetime
import sys
import os
import json
import logging
import sqlite3
import dataclasses
import base64
import io
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterable
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests as http_requests

# Add parent directory (project root) to path so we can import anomaly_pipeline
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Fix DLL load error on Windows by importing torch first
try:
    import torch
except ImportError:
    pass

from flight_fetcher import get, search_flight_path, serialize_flight, deserialize_flight
from anomaly_pipeline import AnomalyPipeline
from core.models import FlightTrack, TrackPoint
from core.config import TRAIN_NORTH, TRAIN_SOUTH, TRAIN_WEST, TRAIN_EAST

# FR24 SDK for fetching flight details
try:
    from fr24sdk.client import Client as FR24Client

    FR24_AVAILABLE = True
    FR24_API_TOKEN = "019aca50-8288-7260-94b5-6d82fbeb351c|dC21vuw2bsf2Y43qAlrBKb7iSM9ibqSDT50x3giN763b577b"
except ImportError:
    FR24_AVAILABLE = False
    FR24_API_TOKEN = None

# OpenAI client for chat
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import staticmap for generating map images
from staticmap import StaticMap, Line, CircleMarker

STATICMAP_AVAILABLE = True

app = FastAPI(title="Anomaly Detection Service", description="API for Multi-Layer Flight Anomaly Detection")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Mount Static Files (The Web UI)
# Prefer the built 'dist' directory for production
static_path = Path("web/dist")
if not static_path.exists():
    # Fallback to 'web' if dist doesn't exist (though this might not work for React apps without Vite)
    static_path = Path("web")

if not static_path.exists():
    static_path.mkdir(parents=True, exist_ok=True)

app.mount("/ui", StaticFiles(directory=str(static_path), html=True), name="ui")

# Resolve paths relative to this file to ensure they work regardless of CWD
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Cache DB Configuration
CACHE_DB_PATH = PROJECT_ROOT / "flight_cache.db"
DB_ANOMALIES_PATH = PROJECT_ROOT / "realtime/live_anomalies.db"
DB_TRACKS_PATH = PROJECT_ROOT / "realtime/live_tracks.db"
DB_RESEARCH_PATH = PROJECT_ROOT / "realtime/research.db"
PRESENT_DB_PATH = BASE_DIR / "present_anomalies.db"


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
    # Initialize feedback and training databases
    init_dbs()
    logger.info("Feedback and training databases initialized")


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


# Rule Definitions
RULES_METADATA = [
    {"id": 1, "name": "Emergency Squawk", "description": "Transponder emergency code (7500, 7600, 7700)"},
    {"id": 2, "name": "Altitude Change", "description": "Extreme altitude change detected"},
    {"id": 3, "name": "Abrupt Turn", "description": "Sharp heading change or holding pattern"},
    {"id": 4, "name": "Proximity Alert", "description": "Dangerous proximity to another aircraft"},
    {"id": 6, "name": "Go-Around", "description": "Aborted landing and climb-out"},
    {"id": 7, "name": "Return to Field", "description": "Immediate return to origin airport"},
    {"id": 8, "name": "Diversion", "description": "Landed at unplanned destination"},
    {"id": 9, "name": "Low Altitude", "description": "Flight below minimum safe altitude"},
    {"id": 10, "name": "Signal Loss", "description": "Extended loss of signal"},
    {"id": 11, "name": "Off Course", "description": "Deviation from known flight paths"},
    {"id": 12, "name": "Unplanned Landing", "description": "Landing at incorrect airport (Israel specific)"},
]


@app.get("/api/rules")
def get_rules():
    """
    Return list of available anomaly rules.
    """
    return RULES_METADATA


@app.get("/api/rules/{rule_id}/flights")
def get_flights_by_rule(rule_id: int):
    """
    Get all flights that triggered a specific rule from Research DB.
    """
    if not DB_RESEARCH_PATH.exists():
        return []

    try:
        conn = sqlite3.connect(str(DB_RESEARCH_PATH))
        cursor = conn.cursor()

        # Use JSON extract to find flights where this rule was triggered
        # Structure: full_report -> layer_1_rules -> report -> matched_rules -> [{id: X, ...}]
        # Note: We use json_each to search the array

        query = """
            SELECT DISTINCT 
                t1.flight_id, 
                t1.timestamp, 
                t1.is_anomaly, 
                t1.severity_cnn, 
                t1.severity_dense, 
                t1.full_report
            FROM anomaly_reports t1, 
                 json_each(t1.full_report, '$.layer_1_rules.report.matched_rules') as rule
            WHERE json_extract(rule.value, '$.id') = ?
            ORDER BY t1.timestamp DESC
            LIMIT 200
        """

        cursor.execute(query, (rule_id,))
        rows = cursor.fetchall()

        # Fetch callsigns for these flights
        flight_ids = [r[0] for r in rows]
        callsigns = {}

        if flight_ids:
            placeholders = ",".join(["?"] * len(flight_ids))
            try:
                cursor.execute(
                    f"SELECT flight_id, callsign FROM anomalies_tracks WHERE flight_id IN ({placeholders}) AND callsign IS NOT NULL",
                    flight_ids)
                for fid, cs in cursor.fetchall():
                    if cs: callsigns[fid] = cs
            except:
                pass

            # Try normal_tracks as fallback
            try:
                cursor.execute(
                    f"SELECT flight_id, callsign FROM normal_tracks WHERE flight_id IN ({placeholders}) AND callsign IS NOT NULL",
                    flight_ids)
                for fid, cs in cursor.fetchall():
                    if cs and fid not in callsigns: callsigns[fid] = cs
            except:
                pass

        conn.close()

        results = []
        for row in rows:
            report = row[5]
            if isinstance(report, str):
                try:
                    report = json.loads(report)
                except:
                    pass

            flight_id = row[0]
            callsign = callsigns.get(flight_id)

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

        return results

    except Exception as e:
        logger.error(f"Failed to fetch flights by rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/track/unified/{flight_id}")
def get_unified_track(flight_id: str):
    """
    Get flight track from any available source (Live DB, Research DB, Cache),
    or fall back to fetching and analyzing if not found.
    """
    points = []

    # 1. Check Live DB
    if DB_TRACKS_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_TRACKS_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM flight_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()
            conn.close()
            if rows:
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
        except Exception as e:
            logger.error(f"Unified track - Live DB error: {e}")

    # 2. Check Research DB if not found
    if not points and DB_RESEARCH_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_RESEARCH_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Try anomalies_tracks
            cursor.execute("SELECT * FROM anomalies_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()
            if not rows:
                # Try normal_tracks
                cursor.execute("SELECT * FROM normal_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
                rows = cursor.fetchall()

            conn.close()

            if rows:
                points = [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Unified track - Research DB error: {e}")

    # 3. Check Cache DB if not found
    if not points:
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
            logger.error(f"Unified track - Cache DB error: {e}")

    # Return if found
    if points:
        return {
            "flight_id": flight_id,
            "points": points
        }

    # 4. Fallback: Fetch and Analyze
    logger.info(f"Unified track - Not found in DBs, analyzing {flight_id}")
    try:
        # Call analyze_flight_endpoint logic to reuse fetch/analyze code
        analysis_result = analyze_flight_endpoint(flight_id)

        if "track" in analysis_result:
            return analysis_result["track"]
        else:
            raise HTTPException(status_code=404, detail="Track not found after analysis")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unified track - Analysis failed: {e}")
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
            import time
            t_fetch = time.time()
            flight = get(flight_id=flight_id)
            logger.info(f"  [Timer] Fetch: {time.time() - t_fetch:.4f}s")

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

        # Inject full points for UI visualization
        results["track"] = {
            "flight_id": flight.flight_id,
            "points": [dataclasses.asdict(p) for p in flight.points]
        }

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
                    cursor_tracks.execute(
                        "SELECT callsign FROM flight_tracks WHERE flight_id = ? AND callsign IS NOT NULL AND callsign != '' LIMIT 1",
                        (flight_id,))
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


@app.get("/api/research/anomalies")
def get_research_anomalies(start_ts: int, end_ts: int):
    """
    Fetch anomalies from the research database within a time range.
    """
    if not DB_RESEARCH_PATH.exists():
        return []

    try:
        conn = sqlite3.connect(str(DB_RESEARCH_PATH))
        cursor = conn.cursor()

        query = """
            SELECT flight_id, timestamp, is_anomaly, severity_cnn, severity_dense, full_report 
            FROM anomaly_reports 
            WHERE timestamp BETWEEN ? AND ? AND is_anomaly = 1
            ORDER BY timestamp DESC
        """

        cursor.execute(query, (start_ts, end_ts))
        rows = cursor.fetchall()

        # Gather all flight_ids
        flight_ids = [r[0] for r in rows]

        # Fetch callsigns from tracks tables (normal or anomalies)
        callsigns = {}
        if flight_ids:
            placeholders = ",".join(["?"] * len(flight_ids))
            # Try anomalies_tracks
            try:
                cursor.execute(
                    f"SELECT flight_id, callsign FROM anomalies_tracks WHERE flight_id IN ({placeholders}) AND callsign IS NOT NULL",
                    flight_ids)
                for fid, cs in cursor.fetchall():
                    if cs: callsigns[fid] = cs
            except:
                pass

            # Try normal_tracks (in case it was flagged anomaly later or vice versa)
            try:
                cursor.execute(
                    f"SELECT flight_id, callsign FROM normal_tracks WHERE flight_id IN ({placeholders}) AND callsign IS NOT NULL",
                    flight_ids)
                for fid, cs in cursor.fetchall():
                    if cs and fid not in callsigns: callsigns[fid] = cs
            except:
                pass

        conn.close()

        results = []
        for row in rows:
            report = row[5]
            if isinstance(report, str):
                try:
                    report = json.loads(report)
                except:
                    pass

            flight_id = row[0]
            callsign = callsigns.get(flight_id)

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

        return results
    except Exception as e:
        logger.error(f"Failed to fetch research anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/research/track/{flight_id}")
def get_research_track(flight_id: str):
    """
    Fetch the full track for a flight from Research DB.
    Checks both anomalies_tracks and normal_tracks.
    """
    if not DB_RESEARCH_PATH.exists():
        raise HTTPException(status_code=404, detail="Research DB not found")

    points = []
    try:
        conn = sqlite3.connect(str(DB_RESEARCH_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Try anomalies_tracks
        try:
            cursor.execute("SELECT * FROM anomalies_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()
            if rows:
                points = [dict(r) for r in rows]
        except:
            pass

        # Try normal_tracks if empty
        if not points:
            try:
                cursor.execute("SELECT * FROM normal_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
                rows = cursor.fetchall()
                if rows:
                    points = [dict(r) for r in rows]
            except:
                pass

        conn.close()
    except Exception as e:
        logger.error(f"Research track fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not points:
        raise HTTPException(status_code=404, detail="Track not found in Research DB")

    return {
        "flight_id": flight_id,
        "points": points
    }


@app.get("/api/research/callsign/{flight_id}")
def get_research_callsign(flight_id: str):
    """
    Fetch a callsign for a research flight ID.
    Tries anomalies_tracks -> normal_tracks -> anomaly_reports summary.
    """
    if not DB_RESEARCH_PATH.exists():
        raise HTTPException(status_code=404, detail="Research DB not found")

    conn = None
    try:
        conn = sqlite3.connect(str(DB_RESEARCH_PATH))
        cursor = conn.cursor()

        callsign = None

        # Try anomalies_tracks first
        try:
            cursor.execute(
                "SELECT callsign FROM anomalies_tracks WHERE flight_id = ? AND callsign IS NOT NULL AND callsign != '' LIMIT 1",
                (flight_id,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                callsign = row[0]
        except Exception:
            pass

        # Fallback to normal_tracks
        if not callsign:
            try:
                cursor.execute(
                    "SELECT callsign FROM normal_tracks WHERE flight_id = ? AND callsign IS NOT NULL AND callsign != '' LIMIT 1",
                    (flight_id,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    callsign = row[0]
            except Exception:
                pass

        # Final fallback: summary in anomaly_reports
        if not callsign:
            try:
                cursor.execute(
                    "SELECT full_report FROM anomaly_reports WHERE flight_id = ? LIMIT 1",
                    (flight_id,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    report = row[0]
                    if isinstance(report, str):
                        try:
                            report = json.loads(report)
                        except Exception:
                            report = None
                    if isinstance(report, dict):
                        callsign = report.get("summary", {}).get("callsign")
            except Exception:
                pass

        return {"callsign": callsign}
    except Exception as e:
        logger.error(f"Failed to fetch research callsign for {flight_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/data/flights")
def get_data_flights(start_ts: int, end_ts: int):
    """
    Get a list of flights within a time range from all available databases.
    Aggregates results from Live DB and Research DB.
    """
    results = {}  # flight_id -> dict

    def process_rows(rows, source_name):
        for r in rows:
            fid = r[0]
            start_t = r[1]
            end_t = r[2]
            cs = r[3]
            count = r[4]

            if fid not in results:
                results[fid] = {
                    "flight_id": fid,
                    "callsign": cs,
                    "start_time": start_t,
                    "end_time": end_t,
                    "point_count": count,
                    "source": source_name
                }
            else:
                # Merge info
                curr = results[fid]
                curr["start_time"] = min(curr["start_time"], start_t) if curr["start_time"] else start_t
                curr["end_time"] = max(curr["end_time"], end_t) if curr["end_time"] else end_t
                curr["point_count"] += count
                # Prefer source that is not "live" if we have research data, or just keep first found?
                # Actually, if we found it in research, it's probably better to label it as such or combined.
                if "research" in source_name:
                    curr["source"] = source_name

                if not curr["callsign"] and cs:
                    curr["callsign"] = cs

    # 1. Live Tracks
    if DB_TRACKS_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_TRACKS_PATH))
            cursor = conn.cursor()
            # Check if index exists on timestamp to make this fast, otherwise it might be slow
            cursor.execute("""
                SELECT flight_id, MIN(timestamp), MAX(timestamp), MAX(callsign), COUNT(*)
                FROM flight_tracks
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY flight_id
            """, (start_ts, end_ts))
            process_rows(cursor.fetchall(), "live")
            conn.close()
        except Exception as e:
            logger.error(f"Error querying live tracks: {e}")

    # 2. Research DB
    if DB_RESEARCH_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_RESEARCH_PATH))
            cursor = conn.cursor()

            # Normal Tracks
            try:
                cursor.execute("""
                    SELECT flight_id, MIN(timestamp), MAX(timestamp), MAX(callsign), COUNT(*)
                    FROM normal_tracks
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY flight_id
                """, (start_ts, end_ts))
                process_rows(cursor.fetchall(), "research_normal")
            except Exception as e:
                logger.warning(f"Error querying normal_tracks: {e}")

            # Anomalies Tracks
            try:
                cursor.execute("""
                    SELECT flight_id, MIN(timestamp), MAX(timestamp), MAX(callsign), COUNT(*)
                    FROM anomalies_tracks
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY flight_id
                """, (start_ts, end_ts))
                process_rows(cursor.fetchall(), "research_anomaly")
            except Exception as e:
                logger.warning(f"Error querying anomalies_tracks: {e}")

            conn.close()
        except Exception as e:
            logger.error(f"Error querying research tracks: {e}")

    return list(results.values())


@app.get("/api/paths")
def get_learned_paths():
    path_file = Path("rules/learned_paths.json")
    if not path_file.exists():
        return {"layers": {"strict": [], "loose": []}}
    with path_file.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/learned-layers")
def get_learned_layers():
    """
    Return all learned layers: paths, turns, SIDs, and STARs.
    These are generated by the learning module and stored in rules/ folder.
    """
    result = {"paths": [], "turns": [], "sids": [], "stars": []}
    
    rules_dir = Path("rules")
    
    # Load paths
    paths_file = rules_dir / "learned_paths.json"
    if paths_file.exists():
        try:
            with paths_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                result["paths"] = data.get("paths", [])
        except Exception as e:
            logger.error(f"Error loading learned_paths.json: {e}")
    
    # Load turns
    turns_file = rules_dir / "learned_turns.json"
    if turns_file.exists():
        try:
            with turns_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                result["turns"] = data.get("zones", [])
        except Exception as e:
            logger.error(f"Error loading learned_turns.json: {e}")
    
    # Load SIDs
    sids_file = rules_dir / "learned_sid.json"
    if sids_file.exists():
        try:
            with sids_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                result["sids"] = data.get("procedures", [])
        except Exception as e:
            logger.error(f"Error loading learned_sid.json: {e}")
    
    # Load STARs
    stars_file = rules_dir / "learned_star.json"
    if stars_file.exists():
        try:
            with stars_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                result["stars"] = data.get("procedures", [])
        except Exception as e:
            logger.error(f"Error loading learned_star.json: {e}")
    
    return result


@app.get("/api/live/track/{flight_id}")
def get_live_track(flight_id: str):
    """
    Fetch the full track for a flight.
    1. Try Live Tracks DB
    2. Fallback to Cache DB
    """
    points = []

    # 1. Try Live DB
    if DB_TRACKS_PATH.exists():
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
        except Exception as e:
            logger.error(f"Failed to fetch from live tracks: {e}")

    # 2. Fallback to Cache
    if not points:
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
            logger.error(f"Failed to fetch from cache: {e}")

    if not points:
        raise HTTPException(status_code=404, detail="Track not found in live or cache DB")

    return {
        "flight_id": flight_id,
        "points": points
    }


from training_ops.db_utils import save_feedback, init_dbs

# Feedback DB Path
FEEDBACK_DB_PATH = PROJECT_ROOT / "training_ops/feedback.db"


def _get_feedback_trigger_replacement(flight_id: str) -> Optional[str]:
    """
    If the user provided anomaly feedback for this flight, return a short string to replace
    the generic trigger label 'User Feedback' (typically the user's comments / other_details).
    """
    if not flight_id:
        return None
    if not FEEDBACK_DB_PATH.exists():
        return None

    try:
        conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Prefer most recent anomaly feedback entries
        rows = cur.execute(
            """
            SELECT comments, other_details
            FROM user_feedback
            WHERE flight_id = ? AND user_label = 1
            ORDER BY timestamp DESC
            LIMIT 3
            """,
            (flight_id,),
        ).fetchall()
        conn.close()

        parts: List[str] = []
        for r in rows:
            c = (r["comments"] or "").strip()
            od = (r["other_details"] or "").strip()
            if c and c not in parts:
                parts.append(c)
            if od and od not in parts:
                parts.append(od)

        if not parts:
            return None

        # Join multiple notes, keep it compact
        s = " | ".join(parts)
        s = " ".join(s.split())  # normalize whitespace/newlines
        if len(s) > 220:
            s = s[:217].rstrip() + "..."
        return s
    except Exception as e:
        logger.warning(f"Failed to fetch feedback comments for flight {flight_id}: {e}")
        return None


def _rewrite_triggers_with_feedback(triggers: Any, flight_id: str) -> Any:
    """
    If triggers is a list and contains 'User Feedback', replace that entry with the user's
    feedback comment string (when available).
    """
    if not isinstance(triggers, list):
        return triggers
    if "User Feedback" not in triggers:
        return triggers

    replacement = _get_feedback_trigger_replacement(flight_id)
    if not replacement:
        return triggers

    return [replacement if t == "User Feedback" else t for t in triggers]


# DEPRECATED/REMOVED - duplicate
# @app.post("/api/feedback/reanalyze/{flight_id}")
def _deprecated_reanalyze_feedback_flight(flight_id: str):
    """
    Re-run the anomaly pipeline for a flight already in the feedback system.
    Updates the record in present_anomalies.db and feedback.db with the new report.
    Returns the new full report structure.
    """
    pipeline = get_pipeline()

    try:
        # 1. Fetch flight data
        # Try Present DB first (since it's a feedback flight)
        flight = None

        if PRESENT_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(PRESENT_DB_PATH))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT flight_id, timestamp, lat, lon, alt, heading, gspeed, vspeed, track, squawk, callsign, source
                    FROM flight_tracks
                    WHERE flight_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (flight_id,),
                )
                rows = cursor.fetchall()
                conn.close()

                if rows:
                    # Convert to FlightTrack object
                    points = []
                    for r in rows:
                        points.append(TrackPoint(
                            lat=r["lat"],
                            lon=r["lon"],
                            alt=r["alt"],
                            timestamp=r["timestamp"],
                            heading=r["heading"],
                            gspeed=r["gspeed"],
                            vspeed=r["vspeed"],
                            squawk=r["squawk"],
                            callsign=r["callsign"],
                            source=r["source"]
                        ))
                    if points:
                        flight = FlightTrack(flight_id=flight_id, points=points)
            except Exception as e:
                logger.warning(f"Failed to fetch from present_anomalies.db: {e}")

        # Fallback to standard fetch (cache/live) if not found (e.g. data lost but cache remains)
        if not flight:
            try:
                # Reuse analyze logic helper or copy
                conn = sqlite3.connect(str(CACHE_DB_PATH))
                cursor = conn.cursor()
                cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    flight = deserialize_flight(row[0])
            except Exception:
                pass

        if not flight or not flight.points:
            # Last resort: fetch live
            flight = get(flight_id=flight_id)

        if not flight or not flight.points:
            raise HTTPException(status_code=404, detail="Flight data not found for re-analysis")

        # 2. Run Pipeline
        results = pipeline.analyze(flight)
        full_report = results  # The pipeline returns the full report dict

        # Extract summary fields
        summary = full_report.get("summary", {})
        is_anomaly = summary.get("is_anomaly", False)
        severity_cnn = full_report.get("severity_cnn", 0.0)
        severity_dense = full_report.get("severity_dense", 0.0)
        confidence_score = summary.get("confidence_score", 0.0)
        triggers = summary.get("triggers", [])
        pipeline_is_anomaly = 1 if is_anomaly else 0

        # Serialize report
        full_report_json = json.dumps(full_report)

        # 3. Update present_anomalies.db
        if PRESENT_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(PRESENT_DB_PATH))
                cursor = conn.cursor()

                # Check if report exists
                cursor.execute("SELECT id FROM anomaly_reports WHERE flight_id = ?", (flight_id,))
                row = cursor.fetchone()

                if row:
                    # Update
                    cursor.execute(
                        """
                        UPDATE anomaly_reports
                        SET full_report_json = ?,
                            severity_cnn = ?,
                            severity_dense = ?,
                            confidence_score = ?,
                            pipeline_is_anomaly = ?,
                            summary_triggers = ?
                        WHERE flight_id = ?
                        """,
                        (
                            full_report_json,
                            severity_cnn,
                            severity_dense,
                            confidence_score,
                            pipeline_is_anomaly,
                            ", ".join([str(t) for t in triggers]),
                            flight_id
                        )
                    )
                else:
                    # Insert new (edge case where track exists but report missing)
                    # We need feedback details to insert correctly, might be messy.
                    # For now, assume it exists if it's in feedback history.
                    pass

                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to update present_anomalies.db: {e}")

        # 4. Update feedback.db
        if FEEDBACK_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
                cursor = conn.cursor()

                # Update full_report_json
                cursor.execute(
                    """
                    UPDATE user_feedback
                    SET full_report_json = ?
                    WHERE flight_id = ?
                    """,
                    (full_report_json, flight_id)
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to update feedback.db: {e}")

        # 5. Return result with formatted structure for UI
        # UI expects AnomalyReport structure
        return {
            "flight_id": flight_id,
            "timestamp": flight.points[0].timestamp,
            "is_anomaly": is_anomaly,
            "severity_cnn": severity_cnn,
            "severity_dense": severity_dense,
            "full_report": full_report,
            "confidence_score": confidence_score
        }

    except Exception as e:
        logger.error(f"Re-analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _safe_join(values: Iterable[Any]) -> str:
    return ", ".join(str(v) for v in values if v is not None and str(v) != "")


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def flatten_rules(full_report: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract rule-related columns plus a per-rule row list.
    Returns (aggregate_columns, rule_rows).
    """
    if not full_report:
        return {}, []

    rules_layer = full_report.get("layer_1_rules") or {}
    rule_report = rules_layer.get("report") or {}
    matched_rules: List[Dict[str, Any]] = rule_report.get("matched_rules") or []
    evaluations: List[Dict[str, Any]] = rule_report.get("evaluations") or []

    aggregate = {
        "rules_status": rules_layer.get("status"),
        "rules_triggers": _safe_join(rules_layer.get("triggers") or []),
        "matched_rule_ids": _safe_join([r.get("id") for r in matched_rules]),
        "matched_rule_names": _safe_join([r.get("name") for r in matched_rules]),
        "matched_rule_categories": _safe_join([r.get("category") for r in matched_rules]),
    }

    rule_rows: List[Dict[str, Any]] = []
    for r in evaluations:
        rule_rows.append(
            {
                "rule_id": r.get("id"),
                "rule_name": r.get("name"),
                "category": r.get("category"),
                "severity": r.get("severity"),
                "matched": 1 if r.get("matched") else 0,
                "summary": _coerce_text(r.get("summary")),
                "details": _coerce_text(r.get("details")),
            }
        )

    return aggregate, rule_rows


@app.get("/api/feedback/track/{flight_id}")
def get_feedback_track(flight_id: str):
    """
    Return track points for a feedback flight from present_anomalies.db.
    """
    if not PRESENT_DB_PATH.exists():
        raise HTTPException(status_code=404, detail="present_anomalies.db not found")

    try:
        conn = sqlite3.connect(str(PRESENT_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT flight_id, timestamp, lat, lon, alt, heading, gspeed, vspeed, track, squawk, callsign, source
            FROM flight_tracks
            WHERE flight_id = ?
            ORDER BY timestamp ASC
            """,
            (flight_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        points = []
        for r in rows:
            points.append({
                "flight_id": r["flight_id"],
                "timestamp": r["timestamp"],
                "lat": r["lat"],
                "lon": r["lon"],
                "alt": r["alt"],
                "heading": r["heading"],
                "gspeed": r["gspeed"],
                "vspeed": r["vspeed"],
                "track": r["track"],
                "squawk": r["squawk"],
                "callsign": r["callsign"],
                "source": r["source"],
            })

        return {"flight_id": flight_id, "points": points}
    except Exception as e:
        logger.error(f"Failed to fetch feedback track {flight_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback/reanalyze/{flight_id}")
def reanalyze_feedback_flight(flight_id: str):
    """
    Re-run analysis for a flight, update feedback DB and present_anomalies DB,
    and return the new report.
    """
    pipeline = get_pipeline()

    try:
        # 1. Fetch flight data (try feedback track first, then others)
        points = []
        if PRESENT_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(PRESENT_DB_PATH))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT lat, lon, alt, heading, gspeed, vspeed, track, squawk, callsign, timestamp, source
                    FROM flight_tracks
                    WHERE flight_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (flight_id,),
                )
                rows = cursor.fetchall()
                conn.close()
                for r in rows:
                    points.append(dict(r))
            except Exception as e:
                logger.error(f"Reanalyze: Failed to fetch from present_anomalies: {e}")

        if not points:
            # Try unified track logic
            track_data = get_unified_track(flight_id)
            points = track_data.get("points", [])

        if not points:
            raise HTTPException(status_code=404, detail="Flight track not found")

        # Reconstruct Flight object
        # Note: FlightTrack/TrackPoint models might be needed
        from core.models import FlightTrack, TrackPoint

        track_points = []
        for p in points:
            # Handle potential key mismatch
            track_points.append(TrackPoint(
                flight_id=flight_id,
                lat=p.get("lat"),
                lon=p.get("lon"),
                alt=p.get("alt"),
                timestamp=p.get("timestamp"),
                gspeed=p.get("gspeed"),
                vspeed=p.get("vspeed"),
                track=p.get("heading") or p.get("track"),
                squawk=p.get("squawk"),
                callsign=p.get("callsign"),
                source=p.get("source")
            ))

        flight = FlightTrack(flight_id=flight_id, points=track_points)

        # 2. Run Analysis
        results = pipeline.analyze(flight)
        full_report = results

        # Serialize report
        report_json = json.dumps(full_report)

        # 3. Update Feedback DB (Source of Truth)
        if FEEDBACK_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
                cursor = conn.cursor()
                # Update all entries for this flight? Or just the latest?
                # Usually we want to update the record associated with the feedback.
                # Let's update all records for this flight_id to be safe/consistent.
                cursor.execute(
                    "UPDATE user_feedback SET full_report_json = ? WHERE flight_id = ?",
                    (report_json, flight_id)
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Reanalyze: Failed to update feedback DB: {e}")

        # 4. Update present_anomalies.db (View for UI)
        if PRESENT_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(PRESENT_DB_PATH))
                cursor = conn.cursor()

                # Flatten rules for columns
                flat_rules, rule_rows = flatten_rules(full_report)
                summary = full_report.get("summary", {})

                # Update anomaly_reports
                # We update the existing row for this flight
                cursor.execute(
                    """
                    UPDATE anomaly_reports
                    SET 
                        anomaly_timestamp = ?,
                        pipeline_is_anomaly = ?,
                        severity_cnn = ?,
                        severity_dense = ?,
                        confidence_score = ?,
                        summary_triggers = ?,
                        rules_status = ?,
                        rules_triggers = ?,
                        matched_rule_ids = ?,
                        matched_rule_names = ?,
                        matched_rule_categories = ?,
                        full_report_json = ?
                    WHERE flight_id = ?
                    """,
                    (
                        full_report.get("timestamp"),
                        summary.get("is_anomaly"),
                        full_report.get("severity_cnn"),
                        full_report.get("severity_dense"),
                        summary.get("confidence_score"),
                        _safe_join(summary.get("triggers") or []),
                        flat_rules.get("rules_status"),
                        flat_rules.get("rules_triggers"),
                        flat_rules.get("matched_rule_ids"),
                        flat_rules.get("matched_rule_names"),
                        flat_rules.get("matched_rule_categories"),
                        report_json,
                        flight_id
                    )
                )

                # Update rule_matches
                # First delete old matches
                # Need report_id(s)
                cursor.execute("SELECT id FROM anomaly_reports WHERE flight_id = ?", (flight_id,))
                report_ids = [r[0] for r in cursor.fetchall()]

                if report_ids:
                    placeholders = ",".join(["?"] * len(report_ids))
                    cursor.execute(f"DELETE FROM rule_matches WHERE report_id IN ({placeholders})", report_ids)

                    # Insert new matches for each report_id (usually just one per flight in this view)
                    for rid in report_ids:
                        if rule_rows:
                            cursor.executemany(
                                """
                                INSERT INTO rule_matches (
                                    report_id, rule_id, rule_name, category, severity, matched, summary, details
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                [
                                    (
                                        rid,
                                        r.get("rule_id"),
                                        r.get("rule_name"),
                                        r.get("category"),
                                        r.get("severity"),
                                        r.get("matched"),
                                        r.get("summary"),
                                        r.get("details"),
                                    )
                                    for r in rule_rows
                                ],
                            )

                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Reanalyze: Failed to update present_anomalies DB: {e}")
                # Don't fail the request if just this DB update fails, return result anyway

        return full_report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Re-analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/history")
def get_feedback_history(start_ts: int = 0, end_ts: int = None, limit: int = 100):
    """
    Fetch feedback flights from `present_anomalies.db`, filtering by actual
    track timestamps (not tag timestamps). For each flight that has at least
    one point in the requested window, return its anomaly report.
    """
    if not PRESENT_DB_PATH.exists():
        logger.warning("present_anomalies.db not found; returning empty history")
        return []

    if end_ts is None:
        end_ts = int(datetime.now().timestamp())

    try:
        conn = sqlite3.connect(str(PRESENT_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 1) Flights that have track points in the requested window
        cursor.execute(
            """
            SELECT flight_id, MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts
            FROM flight_tracks
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY flight_id
            """,
            (start_ts, end_ts),
        )
        track_rows = cursor.fetchall()
        if not track_rows:
            conn.close()
            return []

        flight_ids = [row["flight_id"] for row in track_rows]
        first_ts_map = {row["flight_id"]: row["first_ts"] for row in track_rows}

        # 2) Pull reports for those flights
        placeholders = ",".join(["?"] * len(flight_ids))
        cursor.execute(
            f"""
            SELECT
                id,
                flight_id,
                feedback_id,
                feedback_timestamp,
                user_label,
                comments,
                rule_id,
                other_details,
                COALESCE(anomaly_timestamp, feedback_timestamp, 0) AS ts_report,
                pipeline_is_anomaly,
                severity_cnn,
                severity_dense,
                confidence_score,
                summary_triggers,
                rules_status,
                rules_triggers,
                matched_rule_ids,
                matched_rule_names,
                matched_rule_categories,
                full_report_json
            FROM anomaly_reports
            WHERE flight_id IN ({placeholders})
            """,
            flight_ids,
        )
        report_rows = cursor.fetchall()
        report_by_fid = {}
        for row in report_rows:
            report_by_fid[row["flight_id"]] = row

        history = []
        for fid in flight_ids:
            row = report_by_fid.get(fid)
            full_report = None
            callsign = None
            confidence_score = None
            severity_cnn = None
            severity_dense = None
            pipeline_is_anomaly = None
            rule_id = None
            comments = None
            other_details = None
            matched_rule_ids = None
            matched_rule_names = None
            matched_rule_categories = None
            feedback_id = None

            if row:
                full_report = row["full_report_json"]
                if isinstance(full_report, (str, bytes)):
                    try:
                        full_report = json.loads(full_report)
                    except Exception:
                        pass

                if isinstance(full_report, dict):
                    callsign = full_report.get("summary", {}).get("callsign")
                    triggers = full_report.get("summary", {}).get("triggers")
                    if triggers:
                        full_report["summary"]["triggers"] = _rewrite_triggers_with_feedback(
                            triggers, fid
                        )

                confidence_score = row["confidence_score"]
                severity_cnn = row["severity_cnn"]
                severity_dense = row["severity_dense"]
                pipeline_is_anomaly = row["pipeline_is_anomaly"]
                rule_id = row["rule_id"]
                comments = row["comments"]
                other_details = row["other_details"]
                matched_rule_ids = row["matched_rule_ids"]
                matched_rule_names = row["matched_rule_names"]
                matched_rule_categories = row["matched_rule_categories"]
                feedback_id = row["feedback_id"]

            history.append({
                "flight_id": fid,
                "timestamp": first_ts_map.get(fid, 0),
                "is_anomaly": bool(pipeline_is_anomaly) if pipeline_is_anomaly is not None else True,
                "user_label": row["user_label"] if row else None,
                "severity_cnn": severity_cnn,
                "severity_dense": severity_dense,
                "full_report": full_report,
                "callsign": callsign,
                "feedback_id": feedback_id,
                "comments": comments,
                "rule_id": rule_id,
                "other_details": other_details,
                "confidence_score": confidence_score,
                "matched_rule_ids": matched_rule_ids,
                "matched_rule_names": matched_rule_names,
                "matched_rule_categories": matched_rule_categories,
            })

        # Sort by track timestamp and apply limit
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        return history[:limit]
    except Exception as e:
        logger.error(f"Failed to fetch feedback history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/feedback/{feedback_id}")
def update_feedback(feedback_id: int, update_data: dict):
    """
    Update existing feedback entry (for re-tagging).
    Payload: {
        "rule_id": 1 (required, null means "Other"),
        "comments": "...",
        "other_details": "..." (optional, used when rule_id is null/Other)
    }
    """
    if not FEEDBACK_DB_PATH.exists():
        raise HTTPException(status_code=404, detail="Feedback database not found")

    rule_id = update_data.get("rule_id")
    comments = update_data.get("comments", "")
    other_details = update_data.get("other_details", "")

    # Require either rule_id or other_details
    if rule_id is None and not other_details:
        raise HTTPException(status_code=400, detail="Either rule_id or other_details is required")

    try:
        conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
        cursor = conn.cursor()

        # Update the feedback entry
        cursor.execute(
            """UPDATE user_feedback 
               SET rule_id = ?, comments = ?, other_details = ?
               WHERE id = ?""",
            (rule_id, comments, other_details, feedback_id)
        )

        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="Feedback entry not found")

        conn.commit()
        conn.close()

        logger.info(f"Updated feedback ID {feedback_id}: Rule={rule_id}")
        return {"status": "success", "message": "Feedback updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
def submit_feedback(feedback: dict):
    """
    Submit user feedback for a flight.
    Payload: {
        "flight_id": "...",
        "is_anomaly": true/false,
        "comments": "...",
        "rule_id": 1 (optional, required if is_anomaly=true, null means "Other"),
        "other_details": "..." (optional, used when rule_id is null/Other)
    }
    """
    flight_id = feedback.get("flight_id")
    is_anomaly = feedback.get("is_anomaly")
    comments = feedback.get("comments", "")
    rule_id = feedback.get("rule_id")  # None means "Other" option
    other_details = feedback.get("other_details", "")

    if not flight_id or is_anomaly is None:
        raise HTTPException(status_code=400, detail="Missing flight_id or is_anomaly")

    # If marking as anomaly, require rule selection (rule_id can be None for "Other")
    if is_anomaly and rule_id is None and not other_details:
        raise HTTPException(status_code=400,
                            detail="When marking as anomaly, either rule_id or other_details is required")

    logger.info(f"Received feedback for {flight_id}: Anomaly={is_anomaly}, Rule={rule_id}")

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

    # Check Research DB if not in cache or live
    if not points and DB_RESEARCH_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_RESEARCH_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Try anomalies_tracks
            cursor.execute("SELECT * FROM anomalies_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()

            # Try normal_tracks if not found in anomalies_tracks
            if not rows:
                cursor.execute("SELECT * FROM normal_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
                rows = cursor.fetchall()

            conn.close()

            if rows:
                points = [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Research DB lookup error: {e}")

    if not points:
        raise HTTPException(status_code=404,
                            detail="Flight data not found in cache, live DB, or research DB. Cannot save feedback.")

    # 2. Save to Training DB
    try:
        # Fetch anomaly report before saving (and potentially deleting it)
        full_report = None
        if DB_RESEARCH_PATH.exists():
            try:
                conn_res = sqlite3.connect(str(DB_RESEARCH_PATH))
                conn_res.row_factory = sqlite3.Row
                cur_res = conn_res.cursor()
                row_rep = cur_res.execute(
                    "SELECT full_report FROM anomaly_reports WHERE flight_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (flight_id,)
                ).fetchone()
                if row_rep:
                    raw = row_rep["full_report"]
                    if isinstance(raw, (str, bytes)):
                        full_report = json.loads(raw)
                    elif isinstance(raw, dict):
                        full_report = raw
                conn_res.close()
            except Exception as e:
                logger.warning(f"Failed to fetch anomaly report for feedback: {e}")

        # Fallback to Live DB if not in Research
        if not full_report and DB_ANOMALIES_PATH.exists():
            try:
                conn_live = sqlite3.connect(str(DB_ANOMALIES_PATH))
                conn_live.row_factory = sqlite3.Row
                cur_live = conn_live.cursor()
                row_rep = cur_live.execute(
                    "SELECT full_report FROM anomaly_reports WHERE flight_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (flight_id,)
                ).fetchone()
                if row_rep:
                    raw = row_rep["full_report"]
                    if isinstance(raw, (str, bytes)):
                        full_report = json.loads(raw)
                    elif isinstance(raw, dict):
                        full_report = raw
                conn_live.close()
            except Exception:
                pass

        save_feedback(flight_id, is_anomaly, points, comments, rule_id, other_details, full_report)

        # 3. Update Realtime DB
        if DB_ANOMALIES_PATH.exists():
            try:
                conn = sqlite3.connect(str(DB_ANOMALIES_PATH))
                cursor = conn.cursor()

                # A. Add to ignored_flights so it doesn't get re-analyzed/re-inserted
                cursor.execute(
                    "INSERT OR IGNORE INTO ignored_flights (flight_id, timestamp, reason) VALUES (?, ?, ?)",
                    (flight_id, int(datetime.now().timestamp()), "feedback_given")
                )

                # B. If user says it's NOT an anomaly, remove it from the live anomalies view (soft delete)
                if is_anomaly is False:
                    # We set is_anomaly=0 to "soft delete" it from the live view query which filters is_anomaly=1
                    cursor.execute("UPDATE anomaly_reports SET is_anomaly = 0 WHERE flight_id = ?", (flight_id,))
                    logger.info(f"Removed {flight_id} from live anomalies view (soft delete).")

                conn.commit()
                conn.close()
            except Exception as db_e:
                logger.error(f"Failed to update realtime DB: {db_e}")

        # 4. Update Research DB (Apply same logic)
        if DB_RESEARCH_PATH.exists():
            try:
                conn = sqlite3.connect(str(DB_RESEARCH_PATH))
                cursor = conn.cursor()

                # A. Add to ignored_flights (if table exists)
                try:
                    cursor.execute(
                        "INSERT OR IGNORE INTO ignored_flights (flight_id, timestamp, reason) VALUES (?, ?, ?)",
                        (flight_id, int(datetime.now().timestamp()), "feedback_given")
                    )
                except Exception:
                    # Table might not exist in research DB, ignore
                    pass

                # B. Remove from research DB regardless of anomaly status (since it's processed)
                # Remove from anomaly_reports
                cursor.execute("DELETE FROM anomaly_reports WHERE flight_id = ?", (flight_id,))
                # Remove from anomalies_tracks
                cursor.execute("DELETE FROM anomalies_tracks WHERE flight_id = ?", (flight_id,))
                # Also check normal_tracks just in case
                cursor.execute("DELETE FROM normal_tracks WHERE flight_id = ?", (flight_id,))
                logger.info(f"Deleted {flight_id} from research DB (feedback processed).")

                conn.commit()
                conn.close()
            except Exception as db_e:
                logger.error(f"Failed to update research DB: {db_e}")

    except Exception as e:
        logger.error(f"Save feedback failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")

    return {"status": "success", "message": "Feedback saved and flight added to training dataset"}


# ============================================================
# CHAT ENDPOINT WITH MAP IMAGE GENERATION
# ============================================================

# Map bounds for Israel/Jordan region
MAP_BOUNDS = {
    "MIN_LAT": 29.53523,
    "MAX_LAT": 33.614619,
    "MIN_LON": 34.145508,
    "MAX_LON": 36.386719
}

CHAT_SYSTEM_PROMPT = """
You are "FiveAir Copilot", an aviation anomaly assistant inside a live map UI.
the green is the start and the red is stop, but each one can just be the end of the bounding box we are using, 
so just if there is a airport assume it landed or took of from there 
Style:
- Clear, concise, professional.
- Short paragraphs, bullet points.
- Focus on what matters most.
- Think step-by-step internally but NEVER show chain-of-thought.

Inputs:
- Map screenshot of a flight path.
- Optional flight summary text.

Your tasks (in order of priority):

1. **Identify the core situation**
   - State the key geographic context immediately (e.g., “The aircraft is flying along the Lebanese coast”, “The track enters Lebanese airspace”, “The plane makes a significant deviation away from the typical corridor”).
   - Highlight any region that is politically or operationally sensitive (Lebanon, Syria, Israel).

2. **Explain what the aircraft was doing**
   - Use high-level aviation knowledge (routing, FIR boundaries, restricted airspace, typical LCA–TLV or TLV–LCA paths).
   - Interpret the intention: heading changes, detours, avoidance, approach vectoring, etc.
   - If the cause is unknown, offer *possible* explanations without stating them as fact.

3. **Detect anomalies (focus strongly on the big issue)**
   - If an Israeli aircraft flies inside or extremely near Lebanese airspace, treat it as a **major anomaly** and say this clearly and early.
   - Be direct: explain *why it is abnormal*, what typical routing should look like, and what the deviation implies.
   - Only mention minor issues if relevant after the major issue.

4. **Output format**
   - **Summary:** 1–2 sentences capturing the real story.
   - **Situation analysis:** What the aircraft seems to be doing and why.
   - **Main issue:** State the core anomaly very directly.
   - **Confidence:** One short line.

Rules:
- Do NOT invent details.
- Do NOT suggest specific weather or ATC instructions unless provided.
- Allowed to use general aviation knowledge (airspace restrictions, FIRs, political sensitivity).

"""


def generate_flight_map_image(points: List[Dict[str, Any]], width: int = 800, height: int = 600) -> Optional[str]:
    """
    Generate a map image with the flight path plotted on OpenStreetMap tiles.
    Returns the image as a base64 encoded string.
    Only includes points within the training bounding box (Levant Region).
    """
    if not STATICMAP_AVAILABLE:
        logger.error("staticmap not available, cannot generate map image")
        return None

    if not points or len(points) < 2:
        logger.warning("Not enough points to generate map")
        return None

    try:
        # Filter points to only include those within the training bounding box
        def is_in_bbox(p: Dict[str, Any]) -> bool:
            lat = p.get('lat', p.get('latitude'))
            lon = p.get('lon', p.get('longitude'))
            if lat is None or lon is None:
                return False
            return (TRAIN_SOUTH <= lat <= TRAIN_NORTH and
                    TRAIN_WEST <= lon <= TRAIN_EAST)

        filtered_points = [p for p in points if is_in_bbox(p)]

        if len(filtered_points) < 2:
            logger.warning("Not enough points within bounding box to generate map")
            return None

        # Create the static map
        m = StaticMap(width, height, url_template='https://tile.openstreetmap.org/{z}/{x}/{y}.png')

        # Extract coordinates for the line (from filtered points)
        coords = [(p.get('lon', p.get('longitude')), p.get('lat', p.get('latitude'))) for p in filtered_points
                  if p.get('lon', p.get('longitude')) is not None and p.get('lat', p.get('latitude')) is not None]

        if len(coords) < 2:
            logger.warning("Not enough valid coordinates")
            return None

        # Add the flight path as a red line
        line = Line(coords, 'red', 3)
        m.add_line(line)

        # Add start marker (green)
        start_marker = CircleMarker(coords[0], 'green', 12)
        m.add_marker(start_marker)

        # Add end marker (red)
        end_marker = CircleMarker(coords[-1], 'red', 12)
        m.add_marker(end_marker)

        # Render the map to bytes
        image = m.render()

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return image_base64

    except Exception as e:
        logger.error(f"Failed to generate map image: {e}")
        return None


def fetch_flight_details(flight_id: str, flight_time: int, callsign: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch detailed flight information from FR24 API using flight_summary.get_light().
    Returns a dictionary with flight details like origin, destination, airline, aircraft, etc.
    """
    details = {
        "flight_id": flight_id,
        "callsign": callsign,
        "flight_number": None,
        "origin": None,
        "destination": None,
        "airline": None,
        "aircraft_type": None,
        "aircraft_registration": None,
        "aircraft_model": None,
        "status": None,
        "scheduled_departure": None,
        "scheduled_arrival": None,
        "actual_departure": None,
        "actual_arrival": None
    }

    if not FR24_AVAILABLE:
        logger.warning("FR24 SDK not available, cannot fetch flight details")
        return details

    try:
        client = FR24Client(api_token=FR24_API_TOKEN)
        from datetime import datetime, timedelta

        # Use flight_summary.get_light() - it has the best details
        # We'll search with a wide date range to find the flight
        end_time = datetime.now()
        start_time = end_time - timedelta(days=60)

        summary_data = []
        time_from = flight_time - 60 * 60 * 24
        time_to = flight_time + 60 * 60 * 24
        # If no results and we don't have summary data, try searching by airports in the region
        if not summary_data:
            try:
                summary = client.flight_summary.get_light(
                    flight_datetime_from=datetime.fromtimestamp(time_from).strftime('%Y-%m-%dT%H:%M:%S'),
                    flight_datetime_to=datetime.fromtimestamp(time_to).strftime('%Y-%m-%dT%H:%M:%S'),
                    flight_ids=[flight_id]
                )
                all_data = summary.model_dump().get("data", [])
                # Filter to find our flight_id
                summary_data = [s for s in all_data if s.get("fr24_id") == flight_id]
                logger.info(f"Flight summary search by id found {len(summary_data)} matching flights")
            except Exception as e:
                logger.warning(f"Flight summary search by id failed: {e}")

        # Process the summary data
        if summary_data:
            # Find the matching flight or use first one
            item = None
            for s in summary_data:
                if s.get("fr24_id") == flight_id:
                    item = s
                    break
            if not item and summary_data:
                item = summary_data[0]

            if item:
                # Extract all available fields from flight_summary
                logger.info(f"Processing flight summary data: {list(item.keys())}")

                # Callsign and flight number
                details["callsign"] = item.get("callsign") or callsign
                details["flight_number"] = item.get("flight") or item.get("flight_number")

                # Origin airport - comprehensive extraction
                orig_code = item.get("orig_iata") or item.get("orig_icao") or item.get("schd_from")
                if orig_code:
                    details["origin"] = {
                        "code": orig_code,
                        "iata": item.get("orig_iata"),
                        "icao": item.get("orig_icao"),
                        "name": item.get("orig_name") or item.get("origin_name"),
                        "city": item.get("orig_city") or item.get("origin_city"),
                        "country": item.get("orig_country") or item.get("origin_country")
                    }

                # Destination airport - comprehensive extraction
                dest_code = item.get("dest_iata") or item.get("dest_icao") or item.get("schd_to")
                if dest_code:
                    details["destination"] = {
                        "code": dest_code,
                        "iata": item.get("dest_iata"),
                        "icao": item.get("dest_icao"),
                        "name": item.get("dest_name") or item.get("destination_name"),
                        "city": item.get("dest_city") or item.get("destination_city"),
                        "country": item.get("dest_country") or item.get("destination_country")
                    }

                # Airline information
                details["airline"] = (
                        item.get("airline_name") or
                        item.get("airline") or
                        item.get("airline_iata") or
                        item.get("operator")
                )

                # Aircraft information
                details["aircraft_model"] = item.get("aircraft") or item.get("aircraft_model")
                details["aircraft_type"] = item.get("aircraft_code") or item.get("equip")
                details["aircraft_registration"] = item.get("reg") or item.get("registration")

                # Flight status
                details["status"] = item.get("status") or item.get("status_text")

                # Schedule times
                details["scheduled_departure"] = item.get("schd_dep") or item.get("scheduled_departure")
                details["scheduled_arrival"] = item.get("schd_arr") or item.get("scheduled_arrival")
                details["actual_departure"] = item.get("act_dep") or item.get("actual_departure")
                details["actual_arrival"] = item.get("act_arr") or item.get("actual_arrival")

                logger.info(
                    f"Successfully extracted flight details for {flight_id}: {details['airline']} / {details['callsign']}")

    except Exception as e:
        logger.error(f"Failed to fetch flight details: {e}")

    return details


def format_flight_summary_for_llm(details: Dict[str, Any], points: List[Dict[str, Any]] = None) -> str:
    """
    Format flight details into a readable summary for the LLM.
    """
    lines = ["=== FLIGHT SUMMARY ==="]

    # Basic identification
    if details.get("callsign"):
        lines.append(f"Callsign: {details['callsign']}")
    if details.get("flight_number"):
        lines.append(f"Flight Number: {details['flight_number']}")
    if details.get("flight_id"):
        lines.append(f"Flight ID: {details['flight_id']}")

    # Airline
    if details.get("airline"):
        lines.append(f"Airline/Operator: {details['airline']}")

    # Aircraft
    aircraft_info = []
    if details.get("aircraft_model"):
        aircraft_info.append(details["aircraft_model"])
    if details.get("aircraft_type") and details.get("aircraft_type") != details.get("aircraft_model"):
        aircraft_info.append(f"[{details['aircraft_type']}]")
    if details.get("aircraft_registration"):
        aircraft_info.append(f"(Reg: {details['aircraft_registration']})")
    if aircraft_info:
        lines.append(f"Aircraft: {' '.join(aircraft_info)}")

    # Origin
    origin = details.get("origin")
    if origin and (origin.get("name") or origin.get("code")):
        origin_parts = []
        if origin.get("name"):
            origin_parts.append(origin["name"])
        if origin.get("code") or origin.get("iata") or origin.get("icao"):
            code = origin.get("iata") or origin.get("icao") or origin.get("code")
            if origin.get("name"):
                origin_parts.append(f"[{code}]")
            else:
                origin_parts.append(code)
        if origin.get("city"):
            origin_parts.append(f"- {origin['city']}")
        if origin.get("country"):
            origin_parts.append(f", {origin['country']}")
        lines.append(f"Origin: {''.join(origin_parts)}")

    # Destination
    dest = details.get("destination")
    if dest and (dest.get("name") or dest.get("code")):
        dest_parts = []
        if dest.get("name"):
            dest_parts.append(dest["name"])
        if dest.get("code") or dest.get("iata") or dest.get("icao"):
            code = dest.get("iata") or dest.get("icao") or dest.get("code")
            if dest.get("name"):
                dest_parts.append(f"[{code}]")
            else:
                dest_parts.append(code)
        if dest.get("city"):
            dest_parts.append(f"- {dest['city']}")
        if dest.get("country"):
            dest_parts.append(f", {dest['country']}")
        lines.append(f"Destination: {''.join(dest_parts)}")

    # Schedule times
    if details.get("scheduled_departure"):
        lines.append(f"Scheduled Departure: {details['scheduled_departure']}")
    if details.get("scheduled_arrival"):
        lines.append(f"Scheduled Arrival: {details['scheduled_arrival']}")
    if details.get("actual_departure"):
        lines.append(f"Actual Departure: {details['actual_departure']}")
    if details.get("actual_arrival"):
        lines.append(f"Actual Arrival: {details['actual_arrival']}")

    # Status
    if details.get("status"):
        lines.append(f"Flight Status: {details['status']}")

    # Add track summary if we have points
    if points and len(points) > 0:
        lines.append("")
        lines.append("=== TRACK INFO ===")
        lines.append(f"Total Track Points: {len(points)}")
        lines.append(f"Index Range: 0 to {len(points) - 1}")

        # Get first and last timestamps
        first_pt = points[0]
        last_pt = points[-1]

        if first_pt.get("timestamp") and last_pt.get("timestamp"):
            duration_sec = last_pt["timestamp"] - first_pt["timestamp"]
            duration_min = duration_sec // 60
            hours = duration_min // 60
            mins = duration_min % 60
            if hours > 0:
                lines.append(f"Track Duration: {hours}h {mins}m")
            else:
                lines.append(f"Track Duration: {mins} minutes")

        # Get altitude range
        alts = [p.get("alt", 0) for p in points if p.get("alt") is not None]
        if alts:
            lines.append(f"Altitude Range: {min(alts):.0f} - {max(alts):.0f} ft")

        # Get speed range
        speeds = [p.get("gspeed", 0) for p in points if p.get("gspeed") is not None]
        if speeds:
            lines.append(f"Ground Speed Range: {min(speeds):.0f} - {max(speeds):.0f} kts")

        # Start and end coordinates
        if first_pt.get("lat") and first_pt.get("lon"):
            lines.append(f"Track Start: {first_pt['lat']:.4f}°N, {first_pt['lon']:.4f}°E")
        if last_pt.get("lat") and last_pt.get("lon"):
            lines.append(f"Track End: {last_pt['lat']:.4f}°N, {last_pt['lon']:.4f}°E")

        # Add KEY_POINTS section with sampled points and their actual indices
        lines.append("")
        lines.append("=== KEY_POINTS (with indices for actions) ===")
        lines.append("Use these indices when using highlight_segment action:")

        # Sample key points: first, some evenly spaced, and last
        num_points = len(points)
        key_indices = [0]  # Always include first

        # Add evenly spaced points (up to 8 intermediate points)
        if num_points > 10:
            step = max(1, num_points // 8)
            for i in range(step, num_points - 1, step):
                if i not in key_indices:
                    key_indices.append(i)

        # Always include last
        if num_points - 1 not in key_indices:
            key_indices.append(num_points - 1)

        key_indices.sort()

        for idx in key_indices:
            pt = points[idx]
            lat = pt.get('lat', 'N/A')
            lon = pt.get('lon', 'N/A')
            alt = pt.get('alt', 'N/A')
            ts = pt.get('timestamp', 'N/A')
            heading = pt.get('heading', pt.get('track', 'N/A'))

            lat_str = f"{lat:.4f}" if isinstance(lat, (int, float)) else lat
            lon_str = f"{lon:.4f}" if isinstance(lon, (int, float)) else lon
            alt_str = f"{alt:.0f}ft" if isinstance(alt, (int, float)) else alt
            hdg_str = f"{heading:.0f}°" if isinstance(heading, (int, float)) else heading

            lines.append(f"  idx={idx}: lat={lat_str}, lon={lon_str}, alt={alt_str}, heading={hdg_str}, ts={ts}")

    return "\n".join(lines)


class ChatRequest(BaseModel):
    flight_time: int
    messages: List[Dict[str, str]]
    flight_id: str
    analysis: Optional[Dict[str, Any]] = None
    points: Optional[List[Dict[str, Any]]] = None
    user_question: str


@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """
    Process a chat request using OpenAI Vision API.
    Generates a map image of the flight path and sends it to GPT for analysis.
    """
    try:
        # Get flight points - either from request or fetch them
        points = request.points
        if not points:
            # Try to get points from unified track
            try:
                track_data = get_unified_track(request.flight_id)
                points = track_data.get("points", [])
            except:
                points = []

        # Extract callsign from points if available
        callsign = None
        if points:
            for p in points:
                if p.get("callsign"):
                    callsign = p["callsign"]
                    break

        # Fetch detailed flight information from FR24
        logger.info(f"Fetching flight details for {request.flight_id}...")
        flight_details = fetch_flight_details(request.flight_id, request.flight_time, callsign)

        # Generate map image
        map_image_base64 = None
        if points and len(points) >= 2:
            map_image_base64 = generate_flight_map_image(points)

        if not map_image_base64:
            logger.warning(f"Could not generate map image for flight {request.flight_id}")

        # Build the messages for OpenAI
        openai_messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT}
        ]

        # Add conversation history (skip the first assistant greeting)
        for msg in request.messages:
            if msg.get("role") in ["user", "assistant"]:
                openai_messages.append({"role": msg["role"], "content": msg["content"]})

        # Build the user message with image
        user_content = []
        # user_content.append({
        #     "type": "text",
        #     "text": "reason_mode: \"normal\""
        # }, )
        # Add the map image if available
        if map_image_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{map_image_base64}",
                    "detail": "high"
                }
            })

        # Add flight summary context
        flight_summary_text = format_flight_summary_for_llm(flight_details, points)
        context_text = flight_summary_text + "\n\n"

        # Add anomaly analysis context
        if request.analysis:
            # Add summary of anomaly analysis
            summary = request.analysis.get("summary", {})
            layer1 = request.analysis.get("layer_1_rules", {})

            context_text += "=== ANOMALY ANALYSIS ===\n"
            if summary:
                context_text += f"Is Anomaly: {summary.get('is_anomaly', 'Unknown')}\n"
                context_text += f"Severity CNN: {summary.get('severity_cnn', 'N/A')}\n"
                context_text += f"Severity Dense: {summary.get('severity_dense', 'N/A')}\n"
                triggers = summary.get("triggers", [])
                triggers = _rewrite_triggers_with_feedback(triggers, request.flight_id)
                if triggers:
                    context_text += f"Triggers: {', '.join([str(t) for t in triggers])}\n"

            if layer1 and layer1.get("report", {}).get("matched_rules"):
                rules = layer1["report"]["matched_rules"]
                context_text += f"Matched Rules: {', '.join([r.get('name', str(r.get('id'))) for r in rules])}\n"

            context_text += "\n"

        context_text += f"User Question: {request.user_question}"

        user_content.append({
            "type": "text",
            "text": context_text
        })

        openai_messages.append({
            "role": "user",
            "content": user_content
        })

        # Call OpenAI API using official client
        response = openai_client.chat.completions.create(
            model="gpt-5",

            messages=openai_messages
        )

        ai_response = response.choices[0].message.content

        return {"response": ai_response}

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# AI CO-PILOT ANALYZE ENDPOINT (with screenshot support)
# ============================================================

AI_COPILOT_SYSTEM_PROMPT = """
## IDENTITY AND PURPOSE

You are FiveAir Copilot, an elite aviation anomaly assistant embedded within a live tactical map interface. Your mission is to analyze flight paths in the Eastern Mediterranean (Israel, Lebanon, Syria, Jordan, Cyprus, Egypt) and provide clear, professional, and immediate situational awareness. You act as the eyes on the back of the air traffic controller, spotting risks they might miss.

---

## VISUAL INTERPRETATION RULES

You must strictly interpret the map visuals as follows:

- **Green Marker:** Represents the start of the tracked segment or the entry point into the current bounding box.
- **Red Marker:** Represents the current aircraft position, the stop point, or the exit point.
- **Airport Proximity:** If a Green or Red marker is located near a known airport, assume the aircraft has just taken off or is landing there. Do not assume mid-air start/stop unless over open terrain/sea far from airfields.

---

## OPERATIONAL LOGIC AND PRIORITIES

You must process information in this specific order of priority:

### 1. Identify the Core Situation (Geographic Context)
State the key geographic context immediately. Identify if the aircraft is flying along a specific coastline (e.g., Lebanese coast), crossing a Flight Information Region (FIR) boundary, or entering a politically sensitive region. You must highlight any region that is operationally sensitive, specifically Lebanon, Syria, and Israel borders.

### 2. Detect Anomalies (The Big Issue First)
Focus strongly on the major issue before minor details.

- **Geopolitical Anomaly:** If an Israeli aircraft flies inside or extremely near Lebanese or Syrian airspace, treat it as a Major Anomaly and state this clearly and early.
- **Pattern Anomaly:** Identify loitering (circles), sharp zig-zags, or sudden altitude drops that contradict standard commercial flight profiles.
- **Be direct:** Explain why it is abnormal. Contrast it with what typical routing should look like.

### 3. Explain Behavior (Aviation Knowledge)
Use high-level aviation knowledge regarding routing, restricted airspace, and typical paths (e.g., LCA-TLV routes). Interpret intentions such as heading changes for approach vectoring, weather avoidance, or holding patterns.

If the cause is unknown, offer possible explanations based on physics and geometry, but never state them as absolute facts.

---

## STYLE AND CONSTRAINTS

- **Tone:** Clear, concise, professional, military-grade brevity.
- **Format:** Short paragraphs and bullet points.
- **Internal Process:** Think step-by-step internally but NEVER show your chain-of-thought to the user.
- **Truthfulness:** Do NOT invent details. Do NOT suggest specific weather conditions or ATC instructions unless they are explicitly provided in the data. You are allowed to use general aviation knowledge to infer context.

---

## OUTPUT FORMAT STRUCTURE

You must always use the following headers for your response:

**Summary:** 1-2 sentences capturing the real story and high-level context.

**Situation Analysis:** Detailed breakdown of what the aircraft seems to be doing and why, based on the visual geometry.

**Main Issue:** State the core anomaly very directly.

**Confidence:** One short line (e.g., High, Medium, Low).

---

## MAP HIGHLIGHTING ACTIONS (for the UI)

When you want to point to something on the map, you MAY output a JSON action block.
The JSON must be wrapped in triple backticks with the json language tag.

**IMPORTANT INDEX RULES:**
- The flight_data array uses 0-based indexing
- You will be provided KEY_POINTS with their actual indices - USE THESE EXACT INDICES
- startIndex and endIndex refer to positions in the flight_data array
- If you're unsure about indices, prefer using lat/lon coordinates or timestamps instead

### Available Actions:

**1. Highlight a specific point by coordinates:**
```json
{"action": "highlight_point", "lat": 32.1234, "lon": 34.9876}
```

**2. Highlight a segment by index range (use indices from KEY_POINTS):**
```json
{"action": "highlight_segment", "startIndex": 120, "endIndex": 150}
```

**3. Focus on a specific time:**
```json
{"action": "focus_time", "timestamp": 1702216324}
```

### Rules for actions:
- Use AT MOST one or two actions per response
- Only use actions when:
  - The user asks "where / show me / point to it", OR
  - There is a very clear "problem segment" you want to highlight
- Prefer coordinates (lat/lon) over indices when possible for precision
- If using indices, refer to the KEY_POINTS section for accurate index values

"""


class AIAnalyzeRequest(BaseModel):
    screenshot: Optional[str] = None  # base64 PNG (optional - will generate if not provided)
    question: str
    flight_id: str
    flight_data: List[Dict[str, Any]]
    anomaly_report: Optional[Dict[str, Any]] = None
    selected_point: Optional[Dict[str, Any]] = None  # {lat, lon, timestamp}
    flight_time: Optional[int] = None  # Unix timestamp for fetching flight details
    history: List[Dict[str, str]] = []  # Conversation history [{role, content}]


def parse_actions_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse JSON action blocks from the AI response.
    Looks for ```json { "action": ... } ``` blocks.
    """
    import re
    actions = []

    # Match ```json ... ``` blocks
    json_block_pattern = r'```json\s*([\s\S]*?)```'
    matches = re.findall(json_block_pattern, response_text, re.IGNORECASE)

    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict) and 'action' in parsed:
                actions.append(parsed)
        except json.JSONDecodeError:
            continue

    return actions


def strip_actions_from_text(response_text: str) -> str:
    """
    Remove JSON action blocks from response text for clean display.
    """
    import re
    cleaned = re.sub(r'```json\s*[\s\S]*?```', '', response_text, flags=re.IGNORECASE)
    # Clean up extra whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


@app.post("/api/ai/analyze")
def ai_analyze_endpoint(request: AIAnalyzeRequest):
    """
    AI Co-Pilot endpoint that analyzes a flight with screenshot support.
    Accepts a user-captured screenshot or generates a map image if not provided.
    Returns analysis with optional map actions.
    """
    try:
        logger.info(f"AI Analyze request for flight {request.flight_id}, history: {len(request.history)} messages")

        # Build the messages for OpenAI
        openai_messages = [
            {"role": "system", "content": AI_COPILOT_SYSTEM_PROMPT}
        ]

        # Add conversation history (full conversation context)
        for msg in request.history:
            if msg.get("role") in ["user", "assistant"]:
                openai_messages.append({"role": msg["role"], "content": msg["content"]})

        # Build the user message content
        user_content = []

        # Handle image: use screenshot if provided, otherwise generate map image
        if request.screenshot:
            # Handle both with and without data URL prefix
            screenshot_data = request.screenshot
            if not screenshot_data.startswith('data:'):
                screenshot_data = f"data:image/png;base64,{screenshot_data}"

            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": screenshot_data,
                    "detail": "high"
                }
            })
            logger.info("Screenshot attached to request")
        elif request.flight_data and len(request.flight_data) >= 2:
            # No screenshot provided, generate map image from flight data
            map_image_base64 = generate_flight_map_image(request.flight_data)
            if map_image_base64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{map_image_base64}",
                        "detail": "high"
                    }
                })
                logger.info("Generated map image from flight data")
            else:
                logger.warning("Could not generate map image from flight data")

        # Extract callsign from points if available
        callsign = None
        if request.flight_data:
            for p in request.flight_data:
                if p.get("callsign"):
                    callsign = p["callsign"]
                    break

        # Extract flight_time from request or first point
        flight_time = request.flight_time
        if not flight_time and request.flight_data:
            flight_time = request.flight_data[0].get('timestamp')

        # Fetch flight details for rich context
        flight_details = fetch_flight_details(request.flight_id, flight_time, callsign) if flight_time else {
            "flight_id": request.flight_id,
            "callsign": callsign
        }

        # Build context text using format_flight_summary_for_llm
        context_parts = []

        # Use rich flight summary format
        flight_summary_text = format_flight_summary_for_llm(flight_details, request.flight_data)
        context_parts.append(flight_summary_text)

        # Add explicit time window for better grounding / timeline references
        if request.flight_data and len(request.flight_data) >= 2:
            ts0 = request.flight_data[0].get("timestamp")
            ts1 = request.flight_data[-1].get("timestamp")
            if ts0 and ts1:
                try:
                    from datetime import datetime, timezone
                    iso0 = datetime.fromtimestamp(int(ts0), tz=timezone.utc).isoformat()
                    iso1 = datetime.fromtimestamp(int(ts1), tz=timezone.utc).isoformat()
                    context_parts.append(
                        f"\n=== TIME RANGE ===\nStart: {ts0} ({iso0})\nEnd: {ts1} ({iso1})"
                    )
                except Exception:
                    context_parts.append(f"\n=== TIME RANGE ===\nStart: {ts0}\nEnd: {ts1}")

        # Selected point context
        if request.selected_point:
            sp = request.selected_point
            context_parts.append(
                f"\n=== SELECTED POINT ===\nUser selected point: lat={sp.get('lat')}, lon={sp.get('lon')}, timestamp={sp.get('timestamp')}")

        # Anomaly report summary
        if request.anomaly_report:
            context_parts.append("\n=== ANOMALY ANALYSIS ===")
            report = request.anomaly_report

            # Summary
            summary = report.get('summary', {})
            if summary:
                context_parts.append(f"Is Anomaly: {summary.get('is_anomaly', 'Unknown')}")
                context_parts.append(f"Confidence Score: {summary.get('confidence_score', 'N/A')}%")
                triggers = summary.get('triggers', [])
                triggers = _rewrite_triggers_with_feedback(triggers, request.flight_id)
                if triggers:
                    context_parts.append(f"Triggers: {', '.join(triggers)}")

            # Layer 1 Rules
            layer1 = report.get('layer_1_rules', {})
            if layer1:
                rules = layer1.get('report', {}).get('matched_rules', [])
                if rules:
                    context_parts.append("\nMatched Rules:")
                    for rule in rules:
                        rule_name = rule.get('name', f"Rule {rule.get('id')}")
                        context_parts.append(f"  - {rule_name}")
                        if rule.get('details'):
                            # Add brief details
                            details = rule['details']
                            if 'events' in details:
                                context_parts.append(f"    Events: {len(details['events'])}")

        # Add the question
        context_parts.append(f"\n=== USER QUESTION ===\n{request.question}")

        user_content.append({
            "type": "text",
            "text": "\n".join(context_parts)
        })

        openai_messages.append({
            "role": "user",
            "content": user_content
        })

        # Call OpenAI API
        logger.info("Calling OpenAI API...")
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            # max_tokens=1500
        )

        ai_response = response.choices[0].message.content
        logger.info(f"AI response received: {len(ai_response)} chars")

        # Parse actions from response
        actions = parse_actions_from_response(ai_response)

        # Clean the response text (remove action blocks for display)
        clean_response = strip_actions_from_text(ai_response) if actions else ai_response

        return {
            "response": clean_response,
            "actions": actions
        }

    except Exception as e:
        logger.error(f"AI Analyze endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# AI REASONING AGENT ENDPOINT
# ============================================================

REASONING_AGENT_PROMPT = """
You are the REASONING AGENT.
Your task is to use the provided CONTEXT (flight details, image, anomaly report) and your TOOLS (SQL and Search) to answer user questions.

most of the cases the report anomalies are false positives caused by glitches and 180 turns in ok places, so double check them.

Behave like the original ChatGPT:
- friendly but professional.
- clear, structured answers.
- think step-by-step internally (do NOT reveal chain-of-thought).
- format answers with markdown when useful.
- Be conversational and interactive

---

**CRITICAL: You must OUTPUT the actual <tool> tags when needed to use tool - do NOT just describe what you would search!**



**DO NOT ask the user for confirmation before searching - just output the <tool> block!**
        
---
### EXECUTION CONSTRAINTS

1. THINK FIRST: Before any action, output an internal thought block to detail your multi-step plan. This block MUST be enclosed in  tags.
2. ITERATION: Use tool results to refine your plan or proceed to the final answer.
3. SQL SCHEMA (Target: research.db):
    * anomaly_reports(flight_id, timestamp, is_anomaly, severity_cnn, severity_dense, full_report JSON)
    * anomalies_tracks(flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source)
    * normal_tracks(flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source)
    * **RULE:** All SQL queries MUST use a LIMIT clause (e.g., LIMIT 100).
4. TOOLS USING: After you called a tool it will be executed, and after it you will get the same question but with the tool result in the context (agent: Tool resualt:),
    DO NOT call <tool> and <final> in the same answer!!!.
    when you call final you are ending the question loop causing the system return the answer to the user.
---
### TOOL & FINAL ANSWER FORMATS

* use the tools when asked a question that you didn't got the context for, and when needed do complex sql and internet searches.


**1. INTERNAL DATA GATHERING (Hidden from user):**
To get data for your *own reasoning* (counting, checking details, statistics), use the tool block:
<tool>
sql: SELECT count(*) FROM anomaly_reports WHERE is_anomaly=1
</tool>

**2. WEB SEARCH (openai search):**
<tool>
search: LLBG METAR thunderstorm 2025-07-31
</tool>

IMPORTANT: Your response must contain the literal text "<tool>" and "</tool>" tags. 
The system will execute the search and return results. Do NOT just describe the search in prose.

**3. AVIATION WEATHER (NOAA AviationWeather METAR/TAF):**
Use this when you need actual METAR/TAF for an airport/station.
Prefer ICAO station codes (e.g., KJFK, EGLL, LLBG). If the user gives a city/airport name, you can web-search the ICAO first.
Examples:
<tool>
weather: METAR KJFK
</tool>
<tool>
weather: TAF EGLL
</tool>
<tool>
weather: KSEA
</tool>

**4. FETCHING FLIGHTS FOR USER (Visible in UI):**
When the user asks to "show", "list", "find", or "return" specific flights, use this special tag.
This triggers the UI to display the flight cards interactively.
<fetch and return>
SELECT * FROM anomaly_reports WHERE severity_cnn > 0.8 ORDER BY timestamp DESC LIMIT 5
</fetch and return>

**5. FINAL TEXT ANSWER:**
When the answer is complete (or accompanying the flight list), your response MUST be wrapped in a <final> block.

<final>
[Answer the user's question directly.]
</final>


"""


def execute_reasoning_sql(query: str) -> Dict[str, Any]:
    """
    Read-only SQL execution on research.db for the reasoning agent.
    Returns a dict with either {"rows": [...]} or {"error": "..."}.
    """
    logger.info(f"[REASONING SQL] Query: {query}")

    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "attach", "detach"]
    query_lower = query.lower()
    if any(cmd in query_lower for cmd in forbidden):
        logger.warning("[REASONING SQL] BLOCKED dangerous query")
        return {"error": "Only SELECT queries allowed."}

    # Force a LIMIT if none is present
    if "limit" not in query_lower:
        query = query.rstrip().rstrip(";")
        query += " LIMIT 100"
        logger.info(f"[REASONING SQL] LIMIT appended -> {query}")

    try:
        conn = sqlite3.connect(str(DB_RESEARCH_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        rows = cur.execute(query).fetchall()
        data = [dict(r) for r in rows]
        logger.info(f"[REASONING SQL] Returned {len(data)} rows")
        conn.close()
        return {"rows": data}
    except Exception as e:
        logger.error(f"[REASONING SQL ERROR] {e}")
        return {"error": str(e)}


def execute_web_search(query: str) -> str:
    """
    Perform a web search using OpenAI's web_search tool via the Responses API.
    Returns the search results as text.
    """
    logger.info(f"[WEB SEARCH] Query: {query}")

    try:
        # Use OpenAI's Responses API with web_search tool
        resp = openai_client.responses.create(
            model="gpt-4.1-mini",  # Fast model for web search
            input=query,
            tools=[{"type": "web_search"}],
        )

        # Responses API returns results via output_text
        result_text = resp.output_text
        logger.info(f"[WEB SEARCH] Got {len(result_text)} chars")

        return result_text

    except Exception as e:
        logger.error(f"[WEB SEARCH ERROR] {e}")
        return f"Web search error: {e}"


def execute_aviation_weather(query: str) -> str:
    """
    Fetch aviation weather from NOAA AviationWeather Data API (aviationweather.gov).

    Expected query examples:
      - "METAR KJFK"
      - "TAF EGLL"
      - "KSEA" (defaults to METAR)
      - "METAR KJFK hours=6" (optional; defaults vary by product)

    Returns a human-readable text summary (primarily raw METAR/TAF text).
    """
    import re

    q = (query or "").strip()
    if not q:
        return "Weather tool error: empty query. Use e.g. 'METAR KJFK' or 'TAF EGLL'."

    q_upper = q.upper()

    # Decide product
    want_metar = ("METAR" in q_upper) or ("SPECI" in q_upper)
    want_taf = "TAF" in q_upper
    if not want_metar and not want_taf:
        want_metar = True  # default

    # Optional hoursBeforeNow (supports 'hours=6' or 'hoursBeforeNow=6')
    hours_before: Optional[int] = None
    m_hours = re.search(r"\bHOURS(?:BEFORENOW)?\s*=\s*(\d{1,3})\b", q_upper)
    if m_hours:
        try:
            hours_before = int(m_hours.group(1))
        except Exception:
            hours_before = None

    # Extract station codes: 3-4 letter tokens (ICAO usually 4, some products use 3)
    tokens = re.split(r"[\s,;/]+", q_upper)
    stations: List[str] = []
    for t in tokens:
        if re.fullmatch(r"[A-Z]{3,4}", t) and t not in {"METAR", "TAF", "SPECI", "HOURS", "HOUR"}:
            stations.append(t)
    # De-dup while preserving order
    seen = set()
    stations = [s for s in stations if not (s in seen or seen.add(s))]
    if not stations:
        return (
            "Weather tool error: couldn't find an ICAO station code in your query. "
            "Use e.g. 'METAR KJFK' or 'TAF EGLL'."
        )
    if len(stations) > 5:
        stations = stations[:5]

    def fetch_json(product: str, stations_list: List[str], default_hours: int) -> List[Dict[str, Any]]:
        """
        product: "metar" or "taf"
        Uses AviationWeather Data API:
          - https://aviationweather.gov/api/data/metar
          - https://aviationweather.gov/api/data/taf
        """
        base_url = f"https://aviationweather.gov/api/data/{product}"
        params = {
            "ids": ",".join(stations_list),
            "format": "json",
            "hours": str(hours_before if hours_before is not None else default_hours),
        }
        logger.info(f"[AVIATION WEATHER] product={product} ids={params['ids']} hours={params['hours']}")
        r = http_requests.get(
            base_url,
            params=params,
            timeout=12,
            headers={"User-Agent": "fiveair-anomaly-service/1.0"},
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        # Some failures might return an object; normalize for the agent.
        return [{"_raw": data}]

    def extract_raw_text(items: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            raw = item.get("rawOb") or item.get("raw_text") or item.get("rawText") or item.get("raw")
            if isinstance(raw, str) and raw.strip():
                out.append(raw.strip())
                continue
            # fallback: some products may expose a "text" field
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                out.append(text.strip())
        return out

    lines: List[str] = []
    try:
        if want_metar:
            items = fetch_json("metar", stations, default_hours=6)
            raws = extract_raw_text(items)
            if raws:
                lines.append("METAR:")
                lines.extend([f"- {r}" for r in raws])
            else:
                lines.append("METAR: (no results)")

        if want_taf:
            items = fetch_json("taf", stations, default_hours=24)
            raws = extract_raw_text(items)
            if raws:
                lines.append("TAF:")
                lines.extend([f"- {r}" for r in raws])
            else:
                lines.append("TAF: (no results)")

        return "\n".join(lines).strip()
    except Exception as e:
        return f"Weather tool error: {e}"


def run_reasoning_agent(
        user_message: str,
        conversation_history: List[Dict[str, str]],
        max_steps: int = 8,
        map_image_base64: Optional[str] = None,
        flight_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Core reasoning agent loop.
    The LLM can:
      - ask for <tool> sql: ...
      - ask for <tool> search: ...
      - return flights with <fetch and return> SQL </fetch and return>
      - finally return <final> ... </final>

    If map_image_base64 is provided, it will be included as visual context.
    """
    import time
    current_time = int(time.time())

    # Build context with current time
    time_context = f"\n\nCURRENT TIME: {current_time} (Unix timestamp)\nFor reference: 1 day = 86400 seconds, 1 hour = 3600 seconds"

    # Add flight context if provided
    if flight_context:
        time_context += f"\n\n{flight_context}"

    messages = [
        {"role": "system", "content": REASONING_AGENT_PROMPT + time_context},
    ]

    # Add conversation history
    for msg in conversation_history:
        if msg.get("role") in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Build user message - with or without image
    if map_image_base64:
        # Include image with the user message
        user_content = [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{map_image_base64}",
                "detail": "high"
            }
        }, {
            "type": "text",
            "text": user_message
        }

            #     , {
            #     "type": "text",
            #     "text": "reason_mode: \"high\""
            # }
        ]

        messages.append({"role": "user", "content": user_content})
        logger.info("[REASONING AGENT] Added map image to context")
    else:
        # Simple text message
        messages.append({"role": "user", "content": user_message})

    for step in range(max_steps):
        logger.info(f"[REASONING AGENT] Step {step + 1}/{max_steps}, messages: {len(messages)}")

        resp = openai_client.chat.completions.create(
            model="gpt-5",
            messages=messages,
        )
        msg = resp.choices[0].message.content
        logger.info(f"[REASONING AGENT] LLM output:\n{msg[:500]}...")

        # Record assistant response
        messages.append({"role": "assistant", "content": msg})

        # 1) Check for <fetch and return> - user wants flights displayed
        if "<fetch and return>" in msg and "</fetch and return>" in msg:
            try:
                sql_query = msg.split("<fetch and return>")[1].split("</fetch and return>")[0].strip()
                logger.info(f"[REASONING AGENT] Fetch and return SQL: {sql_query}")

                # Execute the query to get flights
                result = execute_reasoning_sql(sql_query)

                if "error" in result:
                    return {
                        "type": "message",
                        "response": f"Sorry, there was an error executing the query: {result['error']}"
                    }

                # Parse the flights into the expected format
                flights = []
                raw_rows = result.get("rows", [])

                # Gather flight IDs to fetch callsigns if needed
                flight_ids = [r.get("flight_id") for r in raw_rows if r.get("flight_id")]
                callsigns = {}

                if flight_ids and DB_RESEARCH_PATH.exists():
                    try:
                        conn = sqlite3.connect(str(DB_RESEARCH_PATH))
                        cursor = conn.cursor()
                        placeholders = ",".join(["?"] * len(flight_ids))

                        # Try anomalies_tracks
                        try:
                            cursor.execute(
                                f"SELECT flight_id, callsign FROM anomalies_tracks WHERE flight_id IN ({placeholders}) AND callsign IS NOT NULL AND callsign != ''",
                                flight_ids)
                            for fid, cs in cursor.fetchall():
                                if cs: callsigns[fid] = cs
                        except:
                            pass

                        # Try normal_tracks
                        try:
                            cursor.execute(
                                f"SELECT flight_id, callsign FROM normal_tracks WHERE flight_id IN ({placeholders}) AND callsign IS NOT NULL AND callsign != ''",
                                flight_ids)
                            for fid, cs in cursor.fetchall():
                                if cs and fid not in callsigns: callsigns[fid] = cs
                        except:
                            pass

                        conn.close()
                    except Exception as e:
                        logger.error(f"Error fetching callsigns for reasoning results: {e}")

                for row in raw_rows:
                    report = row.get("full_report")
                    if isinstance(report, str):
                        try:
                            report = json.loads(report)
                        except:
                            report = {}

                    flight_id = row.get("flight_id")
                    callsign = callsigns.get(flight_id)

                    if not callsign and isinstance(report, dict):
                        callsign = report.get("summary", {}).get("callsign")

                    flights.append({
                        "flight_id": flight_id,
                        "timestamp": row.get("timestamp"),
                        "is_anomaly": bool(row.get("is_anomaly")),
                        "severity_cnn": row.get("severity_cnn"),
                        "severity_dense": row.get("severity_dense"),
                        "full_report": report,
                        "callsign": callsign
                    })

                # Extract any accompanying message
                response_text = msg
                # Remove the fetch and return block
                response_text = response_text.split("<fetch and return>")[0].strip()
                if "<final>" in response_text:
                    response_text = response_text.split("<final>")[1].split("</final>")[0].strip()
                if not response_text:
                    response_text = f"Found {len(flights)} flight(s) matching your query."

                return {
                    "type": "flights",
                    "response": response_text,
                    "flights": flights
                }

            except Exception as e:
                logger.error(f"[REASONING AGENT] Error parsing fetch and return: {e}")
                return {
                    "type": "message",
                    "response": f"Error processing flight query: {str(e)}"
                }

        # 2) Final answer?
        if "<final>" in msg:
            try:
                final = msg.split("<final>")[1].split("</final>")[0].strip()
                thinking = msg.split("<thinking>")[1].split("</thinking>")[0].strip() if "<thinking>" in msg else None
                if thinking is None:
                    thinking = next(message["content"].split("<thinking>")[1].split("</thinking>")[0].strip() for message in messages if "<thinking>" in message["content"])
                if final:
                    logger.info(f"[REASONING AGENT] Final answer: {final[:200]}...")
                    if thinking:
                        return {"type": "message", "response": "<thinking>"+ thinking+ "</thinking>" + "\n\n" + final}
            except Exception as e:
                logger.warning(f"[REASONING AGENT] Error parsing <final>: {e}")
                return {"type": "message", "response": msg.strip()}

        # 3) SQL tool usage?
        if "<tool>" in msg and "sql:" in msg.lower():
            try:
                tool_block = msg.split("<tool>")[1].split("</tool>")[0]
                sql_query = tool_block.split("sql:", 1)[1].strip()
                if sql_query.lower().startswith("sql:"):
                    sql_query = sql_query[4:].strip()

                logger.info(f"[REASONING AGENT] SQL tool call: {sql_query}")
                sql_result = execute_reasoning_sql(sql_query)

                # Feed tool result back
                result_str = json.dumps(sql_result, indent=2, default=str)
                if len(result_str) > 4000:
                    result_str = result_str[:4000] + "\n... (truncated)"

                messages.append({
                    "role": "agent",
                    "content": f"Tool result (sql):\n{result_str}"
                })
                continue
            except Exception as e:
                logger.error(f"[REASONING AGENT] SQL tool parsing error: {e}")
                messages.append({
                    "role": "agent",
                    "content": f"Tool error: {e}"
                })
                continue

        # 4) Search tool usage?
        if "<tool>" in msg and "search:" in msg.lower():
            try:
                tool_block = msg.split("<tool>")[1].split("</tool>")[0]
                search_query = tool_block.split("search:", 1)[1].strip()

                # Strip quotes if present
                if search_query.startswith('"') and search_query.endswith('"'):
                    search_query = search_query[1:-1]
                elif search_query.startswith("'") and search_query.endswith("'"):
                    search_query = search_query[1:-1]

                logger.info(f"[REASONING AGENT] Search tool call: {search_query}")
                search_result = execute_web_search(search_query)

                # Truncate if too long
                if len(search_result) > 4000:
                    search_result = search_result[:4000] + "\n... (truncated)"

                messages.append({
                    "role": "user",
                    "content": f"Tool result (search):\n{search_result}"
                })
                continue
            except Exception as e:
                logger.error(f"[REASONING AGENT] Search tool parsing error: {e}")
                messages.append({
                    "role": "user",
                    "content": f"Search tool error: {e}"
                })
                continue

        # 5) Aviation weather tool usage?
        if "<tool>" in msg and "weather:" in msg.lower():
            try:
                tool_block = msg.split("<tool>")[1].split("</tool>")[0]
                weather_query = tool_block.split("weather:", 1)[1].strip()

                # Strip quotes if present
                if weather_query.startswith('"') and weather_query.endswith('"'):
                    weather_query = weather_query[1:-1]
                elif weather_query.startswith("'") and weather_query.endswith("'"):
                    weather_query = weather_query[1:-1]

                logger.info(f"[REASONING AGENT] Weather tool call: {weather_query}")
                weather_result = execute_aviation_weather(weather_query)

                if len(weather_result) > 4000:
                    weather_result = weather_result[:4000] + "\n... (truncated)"

                messages.append({
                    "role": "user",
                    "content": f"Tool result (weather):\n{weather_result}"
                })
                continue
            except Exception as e:
                logger.error(f"[REASONING AGENT] Weather tool parsing error: {e}")
                messages.append({
                    "role": "user",
                    "content": f"Weather tool error: {e}"
                })
                continue

        # 6) No tools and no <final> - assume this is the final response
        logger.info("[REASONING AGENT] No tool call or <final> detected, returning raw message")
        return {"type": "message", "response": msg.strip()}

    # Max steps reached
    logger.warning("[REASONING AGENT] Max steps reached")
    return {"type": "message",
            "response": "I've done extensive analysis but couldn't complete the request. Please try a more specific question."}


class ReasoningRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    flight_id: Optional[str] = None
    points: Optional[List[Dict[str, Any]]] = None
    anomaly_report: Optional[Dict[str, Any]] = None


@app.post("/api/ai/reasoning")
def reasoning_endpoint(request: ReasoningRequest):
    """
    AI Reasoning Agent endpoint.
    Accepts a user message and conversation history.
    Optionally accepts flight context (flight_id, points, anomaly_report) for visual analysis.
    Returns either a text response or a list of flights to display.
    """
    try:
        logger.info(f"[REASONING API] Message: {request.message[:100]}...")

        # Generate map image if flight context is provided
        map_image_base64 = None
        flight_context = None

        if request.flight_id and request.points and len(request.points) >= 2:
            logger.info(f"[REASONING API] Generating map for flight {request.flight_id}")
            map_image_base64 = generate_flight_map_image(request.points)

            # Extract callsign from points
            callsign = None
            for p in request.points:
                if p.get("callsign"):
                    callsign = p["callsign"]
                    break

            # Get flight time from first point
            flight_time = request.points[0].get("timestamp", 0) if request.points else 0

            # Fetch detailed flight information from FR24 (origin, destination, airline, etc.)
            logger.info(f"[REASONING API] Fetching flight details for {request.flight_id}...")
            flight_details = fetch_flight_details(request.flight_id, flight_time, callsign)

            # Use format_flight_summary_for_llm to build comprehensive flight context
            flight_context = format_flight_summary_for_llm(flight_details, request.points)
            flight_context += "\n"

            # Add explicit time window to support timeline reasoning and web searches (storms, restrictions, etc.)
            try:
                ts0 = request.points[0].get("timestamp")
                ts1 = request.points[-1].get("timestamp")
                if ts0 and ts1:
                    from datetime import datetime, timezone
                    iso0 = datetime.fromtimestamp(int(ts0), tz=timezone.utc).isoformat()
                    iso1 = datetime.fromtimestamp(int(ts1), tz=timezone.utc).isoformat()
                    flight_context += f"\n=== TIME RANGE ===\nStart: {ts0} ({iso0})\nEnd: {ts1} ({iso1})\n"
            except Exception:
                pass

            # Add anomaly report summary
            if request.anomaly_report:
                summary = request.anomaly_report.get("summary", {})
                layer1 = request.anomaly_report.get("layer_1_rules", {})

                flight_context += f"\n=== ANOMALY ANALYSIS ===\n"
                if summary:
                    flight_context += f"Is Anomaly: {summary.get('is_anomaly', 'Unknown')}\n"
                    flight_context += f"Confidence Score: {summary.get('confidence_score', 'N/A')}%\n"
                    triggers = summary.get('triggers', [])
                    triggers = _rewrite_triggers_with_feedback(triggers, request.flight_id)
                    if triggers:
                        flight_context += f"Triggers: {', '.join([str(t) for t in triggers])}\n"

                if layer1 and layer1.get("report", {}).get("matched_rules"):
                    rules = layer1["report"]["matched_rules"]
                    flight_context += f"Matched Rules: {', '.join([r.get('name', str(r.get('id'))) for r in rules])}\n"

            flight_context += "\nA map image of this flight's path is attached.\n"

        # Prefix flight context with a clear focus instruction
        if flight_context:
            prefixed_context = (
                    "You are looking at ONE SPECIFIC FLIGHT. "
                    "Use this context to understand what it is doing before considering any tools.\n\n"
                    + flight_context
            )
        else:
            prefixed_context = None

        result = run_reasoning_agent(
            request.message,
            request.history,
            map_image_base64=map_image_base64,
            flight_context=prefixed_context
        )

        return result

    except Exception as e:
        logger.error(f"[REASONING API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Run from root with: python service/api.py
    uvicorn.run(app, host="0.0.0.0", port=8001)
