
import sqlite3
from pathlib import Path
import datetime

db_path = Path("service/realtime/research.db") # Assuming run from root, adjusted relative path if needed
# Check if file exists at different location
if not db_path.exists():
    db_path = Path("realtime/research.db")

print(f"Checking DB at: {db_path.absolute()}")

if not db_path.exists():
    print("DB file not found!")
else:
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        print("\nTable counts:")
        for table in ["anomaly_reports", "anomalies_tracks", "normal_tracks"]:
            try:
                cursor.execute(f"SELECT count(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"{table}: {count}")
            except Exception as e:
                print(f"{table}: Error ({e})")
        
        print("\nSample timestamps from anomaly_reports:")
        cursor.execute("SELECT timestamp, flight_id FROM anomaly_reports ORDER BY timestamp DESC LIMIT 5")
        for row in cursor.fetchall():
            ts = row[0]
            dt = datetime.datetime.fromtimestamp(ts)
            print(f"{ts} -> {dt} (Flight: {row[1]})")
            
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

