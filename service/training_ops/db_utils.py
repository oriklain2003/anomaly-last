import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
TRAINING_DB_PATH = ROOT_DIR / "training_ops/training_dataset.db"
FEEDBACK_DB_PATH = ROOT_DIR / "training_ops/feedback.db"

logger = logging.getLogger(__name__)

def init_dbs():
    """Initialize the feedback and training databases."""
    TRAINING_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Feedback DB (Meta-data about user choices)
    conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flight_id TEXT,
            timestamp INTEGER,
            user_label INTEGER, -- 0: Normal, 1: Anomaly
            comments TEXT,
            model_version TEXT
        )
    """)
    conn.commit()
    conn.close()

    # 2. Training DB (The actual flight data repository)
    # We merge everything here: base data + feedback data
    conn = sqlite3.connect(str(TRAINING_DB_PATH))
    cursor = conn.cursor()
    
    # Table for Normal Flights
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flight_tracks (
            flight_id TEXT,
            timestamp INTEGER,
            lat REAL,
            lon REAL,
            alt REAL,
            heading REAL,
            gspeed REAL,
            vspeed REAL,
            track REAL,
            squawk TEXT,
            callsign TEXT,
            source TEXT
        )
    """)
    
    # Table for Anomalous Flights
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomalous_tracks (
            flight_id TEXT,
            timestamp INTEGER,
            lat REAL,
            lon REAL,
            alt REAL,
            heading REAL,
            gspeed REAL,
            vspeed REAL,
            track REAL,
            squawk TEXT,
            callsign TEXT,
            source TEXT
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ft_fid ON flight_tracks (flight_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_at_fid ON anomalous_tracks (flight_id)")
    conn.commit()
    conn.close()

def save_feedback(flight_id: str, is_anomaly: bool, points: List[Dict[str, Any]], comments: str = ""):
    """
    Save user feedback and the corresponding flight data.
    """
    init_dbs()
    
    # 1. Save Metadata
    try:
        conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_feedback (flight_id, timestamp, user_label, comments, model_version) VALUES (?, ?, ?, ?, ?)",
            (flight_id, int(datetime.now().timestamp()), 1 if is_anomaly else 0, comments, "v1")
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save feedback metadata: {e}")
        # We continue to save the data itself as that's crucial

    # 2. Save Flight Data to Training DB
    # Decide which table based on user label
    table = "anomalous_tracks" if is_anomaly else "flight_tracks"
    
    try:
        conn = sqlite3.connect(str(TRAINING_DB_PATH))
        cursor = conn.cursor()
        
        # Check if already exists to avoid duplicates (simple check)
        cursor.execute(f"SELECT count(*) FROM {table} WHERE flight_id = ?", (flight_id,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Insert points
            # Points structure expectation: dict with keys matching columns
            # map keys if necessary
            rows = []
            for p in points:
                rows.append((
                    flight_id,
                    p.get("timestamp", 0),
                    p.get("lat", 0.0),
                    p.get("lon", 0.0),
                    p.get("alt", 0.0),
                    p.get("heading", 0.0),
                    p.get("gspeed", 0.0),
                    p.get("vspeed", 0.0),
                    p.get("track", 0.0),
                    str(p.get("squawk", "")),
                    p.get("callsign", ""),
                    p.get("source", "feedback")
                ))
            
            cursor.executemany(
                f"""INSERT INTO {table} 
                   (flight_id, timestamp, lat, lon, alt, heading, gspeed, vspeed, track, squawk, callsign, source) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows
            )
            conn.commit()
            logger.info(f"Saved {len(rows)} points for {flight_id} into {table}")
        else:
            logger.info(f"Flight {flight_id} already in training DB ({table})")
            
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to save training data: {e}")
        raise e

