import sqlite3
from pathlib import Path

def inspect_db(db_path):
    print(f"--- Inspecting {db_path} ---")
    if not Path(db_path).exists():
        print("File does not exist.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables:", tables)
        
        if tables:
            # Check counts in flight_tracks if it exists
            for table_name in [t[0] for t in tables]:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"Table '{table_name}' has {count} rows.")
                
                if table_name == 'flight_tracks':
                     cursor.execute(f"SELECT flight_id FROM {table_name} LIMIT 5")
                     ids = cursor.fetchall()
                     print(f"Sample IDs in {table_name}: {[i[0] for i in ids]}")

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

inspect_db("flight_tracks3.db")
inspect_db("rules/flight_tracks3.db")

