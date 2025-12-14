from __future__ import annotations

import sys
import time
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fr24sdk.client import Client
from fr24sdk.models.geographic import Boundary
from core.models import FlightTrack, TrackPoint
from anomaly_pipeline import AnomalyPipeline
from rules.rule_engine import FILTER_EXCLUDED_PREFIXES

# --- Configuration ---
DB_PATH = Path("realtime/live_anomalies.db")
# Use the EXACT bounding box used for training to avoid OOD data
TRAIN_NORTH = 34.597042
TRAIN_SOUTH = 28.536275
TRAIN_WEST  = 32.299805
TRAIN_EAST  = 37.397461

# Live fetch boundary (can be same or slightly smaller/larger)
MIN_LAT = TRAIN_SOUTH
MAX_LAT = TRAIN_NORTH
MIN_LON = TRAIN_WEST
MAX_LON = TRAIN_EAST

API_TOKEN = "019aca50-8288-7260-94b5-6d82fbeb351c|dC21vuw2bsf2Y43qAlrBKb7iSM9ibqSDT50x3giN763b577b"
POLL_INTERVAL = 10  # seconds
MIN_POINTS_FOR_ANALYSIS = 50 # Need history for ML models
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ... (Imports and Logging remain same) ...
class FlightState:
    """
    Tracks the history of a single flight in memory.
    """
    def __init__(self, flight_id: str):
        self.flight_id = flight_id
        self.points: List[TrackPoint] = []
        self.last_update = 0.0
        self.last_analyzed_count = 0

    def add_point(self, point: TrackPoint):
        # Only add if timestamp is new
        if not self.points or point.timestamp > self.points[-1].timestamp:
            self.points.append(point)
            self.last_update = time.time()

            
    def to_flight_track(self) -> FlightTrack:
        return FlightTrack(flight_id=self.flight_id, points=self.points)

class RealtimeMonitor:
    def __init__(self):
        self.client = Client(api_token=API_TOKEN)
        self.boundary = Boundary(
            north=MAX_LAT,
            south=MIN_LAT,
            west=MIN_LON,
            east=MAX_LON
        )
        self.active_flights: Dict[str, FlightState] = {}
        self.pipeline = AnomalyPipeline()
        self.setup_db()
        
        # Initialize Memory Repo - DISABLED per user request (too much memory)
        # from core.memory_repo import InMemoryRepository
        # self.memory_repo = InMemoryRepository(self.active_flights)
        self.memory_repo = None
        
    def setup_db(self):
        # 1. Reports DB
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flight_id TEXT,
                timestamp INTEGER,
                is_anomaly BOOLEAN,
                severity_cnn REAL,
                severity_dense REAL,
                full_report JSON
            )
        """)
        # 1b. Ignored Flights Table (for feedback handling)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ignored_flights (
                flight_id TEXT PRIMARY KEY,
                timestamp INTEGER,
                reason TEXT
            )
        """)
        conn.commit()
        conn.close()
        
        # 2. Live Tracks DB (for Rule Engine Proximity)
        # We use a separate file to avoid locking issues
        self.live_db_path = Path("realtime/live_tracks.db")
        conn = sqlite3.connect(str(self.live_db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flight_tracks (
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
                source TEXT
            )
        """)
        # Index for fast range queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ts ON flight_tracks (timestamp)")
        conn.commit()
        conn.close()
        
        # Update Pipeline to use this DB for Rules
        self.pipeline.db_path = self.live_db_path
        # Re-init rule engine with this DB
        try:
            from core.db import FlightRepository, DbConfig
            from rules.rule_engine import AnomalyRuleEngine
            repo = FlightRepository(DbConfig(path=self.live_db_path))
            self.pipeline.rule_engine = AnomalyRuleEngine(repository=repo, rules_path=self.pipeline.rules_path)
            logger.info(f"Pipeline Rule Engine re-pointed to {self.live_db_path}")
        except Exception as e:
            logger.error(f"Failed to re-point Rule Engine: {e}")

    def sync_points_to_db(self, points: List[TrackPoint]):
        if not points:
            return
        try:
            conn = sqlite3.connect(str(self.live_db_path))
            cursor = conn.cursor()
            
            data = [(p.flight_id, p.timestamp, p.lat, p.lon, p.alt, p.gspeed, p.vspeed, p.track, p.squawk, p.callsign, p.source) for p in points]
            
            cursor.executemany(
                """
                INSERT INTO flight_tracks 
                (flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                data
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Live DB Batch Sync Error: {e}")

    def sync_point_to_db(self, p: TrackPoint):
        self.sync_points_to_db([p])

    def log_report(self, report: dict):
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            flight_id = report["summary"]["flight_id"]
            is_anom = report["summary"]["is_anomaly"]
            
            # Extract Severities if available
            sev_cnn = 0.0
            if "layer_4_deep_cnn" in report and "severity" in report["layer_4_deep_cnn"]:
                sev_cnn = report["layer_4_deep_cnn"]["severity"]
                
            sev_dense = 0.0
            if "layer_3_deep_dense" in report and "severity" in report["layer_3_deep_dense"]:
                sev_dense = report["layer_3_deep_dense"]["severity"]

            # We can also log Transformer score if needed, but DB schema needs update.
            # For now, just store full report.
            
            cursor.execute(
                "INSERT INTO anomaly_reports (flight_id, timestamp, is_anomaly, severity_cnn, severity_dense, full_report) VALUES (?, ?, ?, ?, ?, ?)",
                (flight_id, int(time.time()), is_anom, sev_cnn, sev_dense, json.dumps(report))
            )
            conn.commit()
            conn.close()
            
            if is_anom:
                logger.warning(f"ANOMALY DETECTED: {flight_id} (CNN Sev: {sev_cnn:.1f})")
                
        except Exception as e:
            logger.error(f"DB Error: {e}")

    def fetch_and_process(self):
        try:
            # Load ignored flights first
            ignored_ids = set()
            try:
                conn = sqlite3.connect(str(DB_PATH))
                cursor = conn.cursor()
                cursor.execute("SELECT flight_id FROM ignored_flights")
                rows = cursor.fetchall()
                ignored_ids = {r[0] for r in rows}
                conn.close()
            except Exception as e:
                logger.error(f"Failed to load ignored flights: {e}")

            logger.info("Fetching live positions...")
            response = self.client.live.flight_positions.get_full(bounds=self.boundary, altitude_ranges=["1000-50000"])
            
            live_data = response.model_dump()["data"]
            current_ids = set()
            
            for item in live_data:
                time.sleep(1)
                flight_id = item["fr24_id"]

                # Check callsign prefixes
                if item.get("callsign"):
                    callsign = item["callsign"].strip().upper()
                    if callsign.startswith(FILTER_EXCLUDED_PREFIXES):
                        continue

                current_ids.add(flight_id)
                
                # Init state if new
                if flight_id not in self.active_flights:
                    self.active_flights[flight_id] = FlightState(flight_id)
                    logger.info(f"New flight tracked: {flight_id} ({item.get('callsign', 'N/A')})")
                    
                    # Fetch historical track to warm up the state
                    try:
                        logger.info(f"Fetching history for {flight_id}...")
                        hist_resp = self.client.flight_tracks.get(flight_id=[flight_id])
                        if hist_resp:
                            flight_data = hist_resp.model_dump()["data"][0]
                            track_points = flight_data.get("tracks", [])
                            
                            # Convert and add all historical points
                            hist_points_to_sync = []
                            for tp in track_points:
                                # Parse TS
                                ts_str = tp["timestamp"]
                                if isinstance(ts_str, str):
                                    ts_hist = int(datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp())
                                else:
                                    ts_hist = ts_str
                                    
                                p_hist = TrackPoint(
                                    flight_id=flight_id,
                                    timestamp=ts_hist,
                                    lat=tp["lat"],
                                    lon=tp["lon"],
                                    alt=tp["alt"],
                                    gspeed=tp.get("gspeed"),
                                    vspeed=tp.get("vspeed"),
                                    track=tp.get("track"),
                                    squawk=str(tp.get("squawk")) if tp.get("squawk") else None,
                                    callsign=tp.get("callsign"),
                                    source=tp.get("source")
                                )
                                self.active_flights[flight_id].add_point(p_hist)
                                hist_points_to_sync.append(p_hist)
                            
                            self.sync_points_to_db(hist_points_to_sync)
                            logger.info(f"Loaded {len(track_points)} historical points for {flight_id}")
                    except Exception as e:
                        time.sleep(5)
                        logger.warning(f"Could not fetch history for {flight_id}: {e}")
                
                state = self.active_flights[flight_id]
                
                # Parse Point
                # Note: get_full returns a flattened structure, need to map correctly
                # item is like: {'fr24_id': '...', 'lat': ..., 'lon': ..., ...}
                # We need to be careful with timestamp format from SDK
                
                # Timestamp from live feed is usually just 'timestamp' (unix int)
                # but sometimes ISO string. Let's assume unix int or convert.
                ts = item.get("timestamp")
                if isinstance(ts, str):
                    ts = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
                
                point = TrackPoint(
                    flight_id=flight_id,
                    timestamp=ts or int(time.time()),
                    lat=item["lat"],
                    lon=item["lon"],
                    alt=item.get("alt", 0),
                    gspeed=item.get("gspeed"),
                    vspeed=item.get("vspeed"),
                    track=item.get("track"),
                    squawk=item.get("squawk"),
                    callsign=item.get("callsign"),
                    source=item.get("source")
                )
                state.add_point(point)
                
                # Sync to DB for Rule Engine Visibility (Proximity Support)
                # We write just the latest point to a 'live_tracks' or append to history?
                # The RuleEngine uses `flight_tracks` table. We can insert into a temp DB 
                # or the main one if we want history.
                # Let's use a dedicated realtime DB for points to avoid bloating the training DB.
                # We need to adapt RuleEngine to look at THIS db.
                
                self.sync_point_to_db(point)
                
                # Check if ready for analysis
                # Analyze every 10 new points (approx 1-2 mins) to save CPU
                if len(state.points) >= MIN_POINTS_FOR_ANALYSIS:
                    if len(state.points) - state.last_analyzed_count >= 10:
                        # Skip analysis if flight is ignored (feedback given)
                        if flight_id in ignored_ids:
                            state.last_analyzed_count = len(state.points)
                            continue

                        # Run Pipeline
                        track = state.to_flight_track()
                        # Pass memory repo for context-aware rules (Disabled for now)
                        report = self.pipeline.analyze(track, repository=None)
                        self.log_report(report)
                        state.last_analyzed_count = len(state.points)

            # Cleanup stale flights
            # If we haven't seen a flight in this fetch, it might have landed or left bounds
            # For now, keep them for a bit (e.g. 5 mins) then drop
            now = time.time()
            to_remove = []
            for fid, state in self.active_flights.items():
                if fid not in current_ids:
                    if now - state.last_update > 300: # 5 mins timeout
                        to_remove.append(fid)
            
            for fid in to_remove:
                del self.active_flights[fid]
                logger.info(f"Dropped stale flight: {fid}")

        except Exception as e:
            logger.error(f"Fetch loop error: {e}")

    def run(self):
        logger.info(f"Starting Realtime Monitor (Lat: {MIN_LAT}-{MAX_LAT}, Lon: {MIN_LON}-{MAX_LON})")
        while True:
            self.fetch_and_process()
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    monitor = RealtimeMonitor()
    monitor.run()

