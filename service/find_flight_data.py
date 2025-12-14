
import sqlite3
import pandas as pd

def find_flight(flight_id):
    dbs = ['rules/flight_tracks2.db', 'rules/flight_tracks.db', 'flight_cache.db', 'last.db', 'realtime/live_tracks.db']
    for db in dbs:
        try:
            conn = sqlite3.connect(db)
            cursor = conn.cursor()
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table in tables:
                table_name = table[0]
                try:
                    # Try to find column names to query correctly
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [info[1] for info in cursor.fetchall()]
                    
                    if 'flight_id' in columns:
                        query = f"SELECT * FROM {table_name} WHERE flight_id = ?"
                        df = pd.read_sql_query(query, conn, params=(flight_id,))
                        if not df.empty:
                            print(f"Found in {db} table {table_name}: {len(df)} points")
                            print(df.head())
                            return df, db, table_name
                    elif 'id' in columns: # Sometimes id is the flight_id
                         query = f"SELECT * FROM {table_name} WHERE id = ?"
                         df = pd.read_sql_query(query, conn, params=(flight_id,))
                         if not df.empty:
                            print(f"Found in {db} table {table_name}: {len(df)} points")
                            return df, db, table_name
                except Exception as e:
                    pass
            conn.close()
        except Exception as e:
            print(f"Error checking {db}: {e}")

find_flight('3d3a4a0e')

