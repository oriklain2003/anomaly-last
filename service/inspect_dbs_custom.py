import sqlite3
import os
from pathlib import Path

db_paths = [
    Path("realtime/live_tracks.db"),
    Path("realtime/research.db"),
    Path("flight_cache.db")
]

for db_path in db_paths:
    print(f"\n--- Inspecting {db_path} ---")
    if not db_path.exists():
        print(f"{db_path} not found.")
        continue
        
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables: {tables}")
        
        for table in tables:
            table_name = table[0]
            print(f"  Table: {table_name}")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            col_names = [col[1] for col in columns]
            print(f"    Columns: {col_names}")
            
            # Check row count
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"    Row count: {count}")
                
                # Sample a row
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                row = cursor.fetchone()
                print(f"    Sample row: {row}")
            except Exception as e:
                print(f"    Error querying table: {e}")
                
        conn.close()
    except Exception as e:
        print(f"Error inspecting {db_path}: {e}")

