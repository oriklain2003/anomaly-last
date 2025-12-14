import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
from training_ops.db_utils import TRAINING_DB_PATH, FEEDBACK_DB_PATH

# Get feedback flight IDs
conn_fb = sqlite3.connect(str(FEEDBACK_DB_PATH))
feedback_ids = [r[0] for r in conn_fb.execute('SELECT DISTINCT flight_id FROM user_feedback WHERE user_label = 1').fetchall()]
conn_fb.close()

# Check training DB
conn = sqlite3.connect(str(TRAINING_DB_PATH))
cur = conn.cursor()

# Count feedback flights by date
dates_count = defaultdict(int)

for flight_id in feedback_ids[:50]:  # Check first 50
    result = cur.execute(
        'SELECT MIN(timestamp) FROM anomalous_tracks WHERE flight_id = ?',
        (flight_id,)
    ).fetchone()
    
    if result[0]:
        date = datetime.fromtimestamp(result[0]).date()
        dates_count[date] += 1

conn.close()

print("Dates with feedback flights in training DB:")
for date in sorted(dates_count.keys(), reverse=True)[:10]:
    print(f"  {date}: {dates_count[date]} flights")

if dates_count:
    best_date = max(dates_count, key=dates_count.get)
    print(f"\n✅ Try this date: {best_date} ({dates_count[best_date]} flights)")

