import sqlite3
from datetime import datetime
from pathlib import Path

print("=" * 60)
print("DATABASE INVESTIGATION")
print("=" * 60)

# Check research.db
print("\n--- research.db ---")
try:
    if Path('research.db').exists():
        conn = sqlite3.connect('research.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in c.fetchall()]
        print('Tables:', tables)
        
        for table in tables:
            c.execute(f"SELECT COUNT(*) FROM {table}")
            print(f'  {table}: {c.fetchone()[0]} rows')
        conn.close()
    else:
        print("File does not exist!")
except Exception as e:
    print('Error:', e)

# Check live_tracks.db
print("\n--- realtime/live_tracks.db ---")
try:
    if Path('realtime/live_tracks.db').exists():
        conn = sqlite3.connect('realtime/live_tracks.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in c.fetchall()]
        print('Tables:', tables)
        
        for table in tables:
            c.execute(f"SELECT COUNT(*) FROM {table}")
            count = c.fetchone()[0]
            print(f'  {table}: {count} rows')
            
            # Check date range for tables with timestamp
            try:
                c.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table}")
                row = c.fetchone()
                if row[0]:
                    print(f'    From: {datetime.fromtimestamp(row[0])}')
                    print(f'    To: {datetime.fromtimestamp(row[1])}')
            except:
                pass
        conn.close()
    else:
        print("File does not exist!")
except Exception as e:
    print('Error:', e)

# Check live_anomalies.db
print("\n--- realtime/live_anomalies.db ---")
try:
    if Path('realtime/live_anomalies.db').exists():
        conn = sqlite3.connect('realtime/live_anomalies.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in c.fetchall()]
        print('Tables:', tables)
        
        for table in tables:
            c.execute(f"SELECT COUNT(*) FROM {table}")
            count = c.fetchone()[0]
            print(f'  {table}: {count} rows')
            
            # Check date range
            try:
                c.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table}")
                row = c.fetchone()
                if row[0]:
                    print(f'    From: {datetime.fromtimestamp(row[0])}')
                    print(f'    To: {datetime.fromtimestamp(row[1])}')
            except:
                pass
        conn.close()
    else:
        print("File does not exist!")
except Exception as e:
    print('Error:', e)

print("\n" + "=" * 60)
