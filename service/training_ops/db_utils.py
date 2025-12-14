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

TRACK_TABLE_COLUMNS = [
    ("flight_id", "TEXT"),
    ("timestamp", "INTEGER"),
    ("lat", "REAL"),
    ("lon", "REAL"),
    ("alt", "REAL"),
    ("heading", "REAL"),
    ("gspeed", "REAL"),
    ("vspeed", "REAL"),
    ("track", "REAL"),
    ("squawk", "TEXT"),
    ("callsign", "TEXT"),
    ("source", "TEXT"),
]

def ensure_table_columns(cursor: sqlite3.Cursor, table_name: str) -> None:
    """
    Make sure legacy databases contain the full schema.
    SQLite's ALTER TABLE ADD COLUMN is idempotent for new columns, so it's safe.
    """
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row[1] for row in cursor.fetchall()}

    for column_name, column_type in TRACK_TABLE_COLUMNS:
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            logger.warning(
                "Backfilled missing column '%s' on table '%s'",
                column_name,
                table_name,
            )

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
            model_version TEXT,
            rule_id INTEGER,
            other_details TEXT,
            full_report_json TEXT
        )
    """)
    
    # Ensure rule_id and other_details columns exist for older DBs
    try:
        cursor.execute("ALTER TABLE user_feedback ADD COLUMN rule_id INTEGER")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE user_feedback ADD COLUMN other_details TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE user_feedback ADD COLUMN full_report_json TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
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

    # Ensure legacy DBs pick up the full schema (e.g., heading column)
    ensure_table_columns(cursor, "flight_tracks")
    ensure_table_columns(cursor, "anomalous_tracks")

    conn.commit()
    conn.close()

def save_feedback(flight_id: str, is_anomaly: bool, points: List[Dict[str, Any]], comments: str = "", rule_id: Optional[int] = None, other_details: str = "", full_report: Optional[Dict[str, Any]] = None):
    """
    Save user feedback and the corresponding flight data.
    
    Args:
        flight_id: The flight identifier
        is_anomaly: Whether the user marked this as an anomaly
        points: List of track points for the flight
        comments: Optional user comments
        rule_id: The rule ID that caused the anomaly (required if is_anomaly=True, None means "Other")
        other_details: Details when rule_id is None (Other option selected)
        full_report: The anomaly report JSON (optional)
    """
    init_dbs()
    
    # Serialize full_report
    report_json = None
    if full_report:
        try:
            report_json = json.dumps(full_report)
        except Exception:
            pass

    # 1. Save Metadata
    try:
        conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_feedback (flight_id, timestamp, user_label, comments, model_version, rule_id, other_details, full_report_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (flight_id, int(datetime.now().timestamp()), 1 if is_anomaly else 0, comments, "v1", rule_id, other_details, report_json)
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

