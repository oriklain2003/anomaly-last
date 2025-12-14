import sqlite3
import pandas as pd
import json

def query_to_csv(db_path, query, csv_path):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")
        if len(df) > 0:
            print("Columns:", df.columns.tolist())
            # If there's a 'points' column (JSON), we might need to parse it
            # But usually tracks are stored either as one-row-per-flight (with JSON points) or one-row-per-point.
            # Let's inspect the first row to see the structure.
            print("First row sample:")
            print(df.iloc[0].to_dict())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

# Query for the specific flight
flight_id = '3b165a04'
db_path = "flight_cache.db"

# First, check schema of anomalies_tracks
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(anomalies_tracks)")
columns = cursor.fetchall()
print("Table Schema:", columns)
conn.close()

# Now dump the flight
query_to_csv(
    db_path,
    f"SELECT * FROM tracks where flight_id = '{flight_id}'",
    "emoee.csv"
)
