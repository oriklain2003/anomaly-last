import sqlite3
from datetime import datetime
from pathlib import Path

dbs = [
    'realtime/research.db',
    'service/present_anomalies.db',
    'service/llm_full_pipeline/llm_tracks.db',
    'training_ops/consolidated.db',
    'training_ops/temp_merged.db',
    'training_ops/feedback.db'
]

for db in dbs:
    print(f'\n--- {db} ---')
    try:
        if not Path(db).exists():
            print('File not found')
            continue
            
        conn = sqlite3.connect(db)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in c.fetchall()]
        print('Tables:', tables)
        
        for table in tables[:5]:
            c.execute(f'SELECT COUNT(*) FROM [{table}]')
            count = c.fetchone()[0]
            print(f'  {table}: {count} rows')
            
            # Try to get date range if has timestamp
            if count > 0:
                try:
                    c.execute(f'SELECT MIN(timestamp), MAX(timestamp) FROM [{table}]')
                    row = c.fetchone()
                    if row[0] and row[0] > 0:
                        print(f'    From: {datetime.fromtimestamp(row[0])}')
                        print(f'    To: {datetime.fromtimestamp(row[1])}')
                except:
                    pass
                    
            # Check for flight_id count
            try:
                c.execute(f'SELECT COUNT(DISTINCT flight_id) FROM [{table}]')
                flight_count = c.fetchone()[0]
                if flight_count > 0:
                    print(f'    Distinct flights: {flight_count}')
            except:
                pass
                
        conn.close()
    except Exception as e:
        print('Error:', e)

print('\n' + '='*60)
print('SUMMARY: Looking for your 6659 flights...')
print('='*60)

