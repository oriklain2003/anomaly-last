import argparse
import sqlite3
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Try importing fr24sdk, handle if not available (though it should be)
try:
    from fr24sdk.client import Client
    from core.models import FlightTrack, TrackPoint
    from anomaly_pipeline import AnomalyPipeline
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("research_replay.log")
    ]
)
logger = logging.getLogger("research_replay")

API_TOKEN = "019aca50-8288-7260-94b5-6d82fbeb351c|dC21vuw2bsf2Y43qAlrBKb7iSM9ibqSDT50x3giN763b577b"

from core.config import TRAIN_NORTH, TRAIN_SOUTH, TRAIN_EAST, TRAIN_WEST

class ResearchReplay:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.client = Client(api_token=API_TOKEN)
        self.pipeline = AnomalyPipeline()
        self.setup_db()
        
    def setup_db(self):
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 1. Anomalies Tracks
        cursor.execute("""
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
        
        # 2. Normal Tracks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS normal_tracks (
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
        
        # 3. Anomaly Report
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
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def save_tracks(self, flight: FlightTrack, is_anomaly: bool):
        table = "anomalies_tracks" if is_anomaly else "normal_tracks"
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            data = []
            for p in flight.points:
                data.append((
                    p.flight_id, p.timestamp, p.lat, p.lon, p.alt, 
                    p.gspeed, p.vspeed, p.track, p.squawk, p.callsign, p.source
                ))
            
            # Chunk inserts to avoid too many SQL variables
            chunk_size = 500
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                cursor.executemany(f"""
                    INSERT OR IGNORE INTO {table}
                    (flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, chunk)
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save tracks for {flight.flight_id}: {e}")

    def save_report(self, report: dict, timestamp: int):
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            flight_id = report["summary"]["flight_id"]
            is_anom = report["summary"]["is_anomaly"]
            
            sev_cnn = 0.0
            if "layer_4_deep_cnn" in report and "severity" in report["layer_4_deep_cnn"]:
                sev_cnn = report["layer_4_deep_cnn"]["severity"]
                
            sev_dense = 0.0
            if "layer_3_deep_dense" in report and "severity" in report["layer_3_deep_dense"]:
                sev_dense = report["layer_3_deep_dense"]["severity"]
                
            cursor.execute(
                "INSERT INTO anomaly_reports (flight_id, timestamp, is_anomaly, severity_cnn, severity_dense, full_report) VALUES (?, ?, ?, ?, ?, ?)",
                (flight_id, timestamp, is_anom, sev_cnn, sev_dense, json.dumps(report))
            )
            conn.commit()
            conn.close()
        except Exception as e:
             logger.error(f"Failed to save report for {report.get('summary', {}).get('flight_id')}: {e}")

    def fetch_full_track(self, flight_id: str) -> Optional[FlightTrack]:
        time.sleep(3)
        try:
            # Based on flight_fetcher.py
            tracks = self.client.flight_tracks.get(flight_id=[flight_id])
            
            data_list = tracks.model_dump()["data"]
            if not data_list:
                return None
                
            flight_data = data_list[0]
            fr24_id = flight_data["fr24_id"]
            track_points = flight_data["tracks"]
            
            points = []
            for tp in track_points:
                ts_val = tp["timestamp"]
                if isinstance(ts_val, str):
                    ts = int(datetime.fromisoformat(ts_val.replace("Z", "+00:00")).timestamp())
                else:
                    ts = int(ts_val)
                
                points.append(TrackPoint(
                    flight_id=fr24_id,
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
            
            return FlightTrack(flight_id=fr24_id, points=points)
            
        except Exception as e:
            logger.error(f"Error fetching full track for {flight_id}: {e}")
            return None

    def _is_track_in_bounds(self, track: FlightTrack) -> bool:
        """Check if any point of the track is within the training box."""
        for p in track.points:
            if (TRAIN_SOUTH <= p.lat <= TRAIN_NORTH) and (TRAIN_WEST <= p.lon <= TRAIN_EAST):
                return True
        return False

    def run(self, start_date: datetime, end_date: datetime, jump: int, sleep: int):
        current = start_date
        processed_flights = set()

        logger.info(f"Starting replay from {start_date} to {end_date}")
        logger.info(f"Params: jump={jump}s, sleep={sleep}s")

        while current < end_date:
            window_end = current + timedelta(seconds=jump)
            if window_end > end_date:
                window_end = end_date
                
            logger.info(f"Scanning window: {current} -> {window_end}")
            
            try:
                 s_str = current.strftime('%Y-%m-%dT%H:%M:%S')
                 e_str = window_end.strftime('%Y-%m-%dT%H:%M:%S')
                 
                 # We use airports as a proxy for region since bounds are not supported
                 # Key airports in the monitoring region (Levant)
                 airports_list = [
                     "LLBG", "LLET", "LLHA", "LLOV", "LLRM", "LLHZ", "LLIB", # Israel
                     "OJAI", "OJAM", "OJAQ", # Jordan
                     "HEAR", "HETB", "HECA", # Egypt
                     "OLBA", "OSDI" # Lebanon, Syria
                 ]
                 
                 # Note: get_light parameter naming might vary by version, assuming standard
                 response = self.client.flight_summary.get_light(
                     flight_datetime_from=s_str,
                     flight_datetime_to=e_str,
                     airports=airports_list
                 )
                 
                 data = response.model_dump().get("data", [])
                 logger.info(f"Found {len(data)} flights/summaries.")
                 
                 for item in data:
                     flight_id = item.get("fr24_id") or item.get("id")
                     
                     # Skip if already processed
                     if not flight_id or flight_id in processed_flights:
                         continue
                     
                     # Fetch and Process
                     processed_flights.add(flight_id)
                     
                     track = self.fetch_full_track(flight_id)
                     
                     if track:
                        # Filter: Must be inside the bounding box
                        if not self._is_track_in_bounds(track):
                             logger.info(f"Skipping {flight_id} (outside bounds)")
                             continue

                        if len(track.points) > 10:
                             logger.info(f"Analyzing {flight_id} ({len(track.points)} pts)...")
                             report = self.pipeline.analyze(track)
                             
                             is_anomaly = report["summary"]["is_anomaly"]
                             
                             self.save_tracks(track, is_anomaly)
                             
                             # Use the timestamp of the LAST point in the track as the event time
                             # This ensures the anomaly appears on the correct historical date in the UI
                             if track.points:
                                 last_ts = track.points[-1].timestamp
                             else:
                                 last_ts = int(time.time())
                                 
                             self.save_report(report, last_ts)
                             
                             logger.info(f"Saved {flight_id}. Anomaly: {is_anomaly} (TS: {datetime.fromtimestamp(last_ts)})")
                        else:
                             logger.info(f"Skipping {flight_id} (insufficient data)")
                     else:
                        logger.info(f"Skipping {flight_id} (track fetch failed)")
                     
                     time.sleep(0.2) # Mild rate limit
                     
            except Exception as e:
                logger.error(f"Window processing error: {e}")
            
            current = window_end
            logger.info(f"Sleeping for {sleep}s...")
            time.sleep(sleep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay historical flights and run anomaly detection.")
    parser.add_argument("start_date", help="Start Date (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("end_date", help="End Date (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--jump", type=int, default=3600, help="Time step in seconds (default 1 hour)")
    parser.add_argument("--sleep", type=int, default=5, help="Sleep between steps in seconds (default 5)")
    parser.add_argument("--db", default="realtime/research.db", help="Output Database path")
    
    args = parser.parse_args()
    
    try:
        s_date = datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S")
        e_date = datetime.strptime(args.end_date, "%Y-%m-%d %H:%M:%S")
        
        replay = ResearchReplay(args.db)
        replay.run(s_date, e_date, args.jump, args.sleep)
        
    except ValueError:
        print("Invalid date format. Use YYYY-MM-DD HH:MM:SS")
        sys.exit(1)

