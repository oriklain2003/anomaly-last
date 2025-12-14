import sqlite3
import os

dbs = [
    "realtime/research.db",
    "last.db",
    "service/llm_full_pipeline/llm_research.db",
    "flight_cache.db",
    "training_ops/consolidated.db"
]

for db_path in dbs:
    if not os.path.exists(db_path):
        print(f"Skipping {db_path} (not found)")
        continue
        
    print(f"\n--- Schema for {db_path} ---")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            print(f"Table: {table_name}")
            
            # Get columns
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
                
        conn.close()
    except Exception as e:
        print(f"Error reading {db_path}: {e}")



