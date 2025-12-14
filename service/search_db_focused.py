import sqlite3
import json

db_path = "realtime/research.db"
print(f"Checking {db_path} for 'Go-around detected'...")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT flight_id, full_report FROM anomaly_reports WHERE full_report LIKE '%Go-around detected%'")
    results = cursor.fetchall()
    
    found_ids = set()
    if results:
        for row in results:
            found_ids.add(row[0])
            
    print(f"Found {len(found_ids)} unique flights in DB: {found_ids}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")



