import sqlite3
import logging
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).resolve().parent))

from training_ops.db_utils import ensure_table_columns

logging.basicConfig(level=logging.INFO)

DB_PATH = Path("last.db")

def update():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    try:
        ensure_table_columns(cursor, "flight_tracks")
        ensure_table_columns(cursor, "anomalous_tracks")
        conn.commit()
        print("Schema updated.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    update()



