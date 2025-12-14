import sqlite3
import os
from pathlib import Path

TEMP_DB = Path("training_ops/temp_merged.db")
LAST_DB = Path("last.db")
CONSOLIDATED_DB = Path("training_ops/consolidated.db")
FEEDBACK_DB = Path("training_ops/feedback.db")

def merge_data():
    if TEMP_DB.exists():
        os.remove(TEMP_DB)
    
    conn = sqlite3.connect(TEMP_DB)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute("""
        CREATE TABLE flight_tracks (
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
    
    # Attach last.db
    print("Merging data from last.db...")
    cursor.execute(f"ATTACH DATABASE '{LAST_DB}' AS source_last")
    cursor.execute("INSERT INTO flight_tracks SELECT * FROM source_last.flight_tracks")
    count_last = cursor.rowcount
    print(f"Imported {count_last} rows from last.db")
    
    # Attach consolidated and feedback
    print("Merging normal flights from consolidated.db (via feedback)...")
    cursor.execute(f"ATTACH DATABASE '{CONSOLIDATED_DB}' AS source_cons")
    cursor.execute(f"ATTACH DATABASE '{FEEDBACK_DB}' AS feedback")
    
    # Get Normal Flight IDs
    cursor.execute("SELECT DISTINCT flight_id FROM feedback.user_feedback WHERE user_label = 0")
    normal_ids = [r[0] for r in cursor.fetchall()]
    print(f"Found {len(normal_ids)} normal flights in feedback.")
    
    # Let's check overlap first.
    cursor.execute("SELECT DISTINCT flight_id FROM flight_tracks")
    existing_ids = set(r[0] for r in cursor.fetchall())
    
    to_insert = [fid for fid in normal_ids if fid not in existing_ids]
    print(f"Adding {len(to_insert)} flights from consolidated.db (skipping {len(normal_ids) - len(to_insert)} already in last.db)")
    
    if to_insert:
        # SQLite limit for host parameters is usually 999 or 32766.
        # We'll do it in chunks or loop.
        
        # Actually, let's use a temporary table for IDs to join.
        cursor.execute("CREATE TEMPORARY TABLE target_ids (flight_id TEXT)")
        cursor.executemany("INSERT INTO target_ids VALUES (?)", [(fid,) for fid in to_insert])
        
        cursor.execute("""
            INSERT INTO flight_tracks 
            SELECT t.* FROM source_cons.flight_tracks t
            JOIN target_ids i ON t.flight_id = i.flight_id
        """)
        count_cons = cursor.rowcount
        print(f"Imported {count_cons} rows from consolidated.db")
    
    conn.commit()
    
    # Final count
    cursor.execute("SELECT COUNT(DISTINCT flight_id) FROM flight_tracks")
    final_flights = cursor.fetchone()[0]
    print(f"Total unique flights in merged DB: {final_flights}")
    
    conn.close()

if __name__ == "__main__":
    merge_data()



