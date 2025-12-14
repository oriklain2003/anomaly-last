import sqlite3
from datetime import datetime

# Check training database
from training_ops.db_utils import TRAINING_DB_PATH, FEEDBACK_DB_PATH

print(f"Training DB: {TRAINING_DB_PATH}")
print(f"Exists: {TRAINING_DB_PATH.exists()}")

if TRAINING_DB_PATH.exists():
    conn = sqlite3.connect(str(TRAINING_DB_PATH))
    cur = conn.cursor()
    
    # Check tables
    tables = [r[0] for r in cur.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]
    print(f"\nTables: {tables}")
    
    # Check anomalous_tracks
    if 'anomalous_tracks' in tables:
        count = cur.execute('SELECT COUNT(*) FROM anomalous_tracks').fetchone()[0]
        print(f"\nTotal rows in anomalous_tracks: {count}")
        
        if count > 0:
            # Get date range
            cur.execute('SELECT MIN(timestamp), MAX(timestamp) FROM anomalous_tracks')
            min_ts, max_ts = cur.fetchone()
            if min_ts and max_ts:
                print(f'Date range: {datetime.fromtimestamp(min_ts)} to {datetime.fromtimestamp(max_ts)}')
            
            # Test the specific query
            start_ts = 1751230800  # June 30, 2025
            end_ts = 1751317199
            print(f'\nQuerying for: {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(end_ts)}')
            
            # Get feedback flight IDs
            conn_fb = sqlite3.connect(str(FEEDBACK_DB_PATH))
            feedback_ids = [r[0] for r in conn_fb.execute('SELECT DISTINCT flight_id FROM user_feedback WHERE user_label = 1').fetchall()]
            conn_fb.close()
            print(f'Feedback flight IDs: {len(feedback_ids)}')
            
            # Check how many are in the date range
            placeholders = ','.join(['?'] * len(feedback_ids))
            query = f'SELECT COUNT(DISTINCT flight_id) FROM anomalous_tracks WHERE flight_id IN ({placeholders}) AND timestamp BETWEEN ? AND ?'
            
            count_in_range = cur.execute(query, feedback_ids + [start_ts, end_ts]).fetchone()[0]
            print(f'Feedback flights in that date range: {count_in_range}')
            
            # Show what flights ARE in the date range (not filtered by feedback)
            all_in_range = cur.execute(
                'SELECT COUNT(DISTINCT flight_id) FROM anomalous_tracks WHERE timestamp BETWEEN ? AND ?',
                (start_ts, end_ts)
            ).fetchone()[0]
            print(f'ALL anomalous flights in that date range: {all_in_range}')
    
    conn.close()

