import subprocess
import sqlite3
import logging
import sys
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("full_process")

# Configuration
ROOT_DIR = Path(__file__).resolve().parent
RULES_DIR = ROOT_DIR / "rules"
TRAINING_DIR = ROOT_DIR / "training_ops"

SOURCE_DB_PATH = ROOT_DIR.parent / "flight_tracks_new.db"
TARGET_DB_PATH = ROOT_DIR / "last.db"

def run_command(command, cwd=None, env=None):
    """Run a shell command and stream output."""
    logger.info(f"Running: {' '.join(command)} in {cwd or '.'}")
    try:
        # Update env if provided
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
            
        result = subprocess.run(
            command,
            cwd=cwd,
            env=run_env,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        print(e.stdout)
        print(e.stderr)
        raise e

def ensure_schema():
    """Ensure last.db has the correct schema (e.g. heading column)."""
    logger.info("Checking schema of last.db...")
    
    # Add parent path to import core.db utils if needed, 
    # but we can just do the sqlite work directly here to be self-contained
    
    conn = sqlite3.connect(str(TARGET_DB_PATH))
    cursor = conn.cursor()
    
    required_columns = {
        "heading": "REAL",
        # Add others if needed, but heading was the main missing one
    }
    
    tables = ["flight_tracks", "anomalous_tracks"]
    
    for table in tables:
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            existing_cols = {row[1] for row in cursor.fetchall()}
            
            for col, dtype in required_columns.items():
                if col not in existing_cols:
                    logger.info(f"Adding missing column '{col}' to table '{table}'")
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            # Table might not exist yet, which is fine, merge will create it
            pass
            
    conn.commit()
    conn.close()

def merge_databases():
    """Merge new data from source to target."""
    if not SOURCE_DB_PATH.exists():
        logger.error(f"Source DB {SOURCE_DB_PATH} not found")
        return

    logger.info(f"Merging from {SOURCE_DB_PATH} to {TARGET_DB_PATH}")
    
    conn_target = sqlite3.connect(str(TARGET_DB_PATH))
    cursor = conn_target.cursor()
    
    # Attach source
    cursor.execute("ATTACH DATABASE ? AS source", (str(SOURCE_DB_PATH),))
    
    # 1. Merge flight_tracks
    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS main.flight_tracks (
            flight_id TEXT, timestamp INTEGER, lat REAL, lon REAL, alt REAL,
            heading REAL, gspeed REAL, vspeed REAL, track REAL, squawk TEXT,
            callsign TEXT, source TEXT,
            PRIMARY KEY (flight_id, timestamp)
        )
    """)
    
    # Get common columns
    cursor.execute("PRAGMA table_info(flight_tracks)")
    target_cols = [row[1] for row in cursor.fetchall()]
    
    cursor.execute("PRAGMA source.table_info(flight_tracks)")
    source_cols = [row[1] for row in cursor.fetchall()]
    
    common_cols = list(set(target_cols) & set(source_cols))
    cols_str = ", ".join(common_cols)
    
    logger.info(f"Merging flight_tracks using columns: {cols_str}")
    cursor.execute(f"""
        INSERT OR IGNORE INTO main.flight_tracks ({cols_str})
        SELECT {cols_str} FROM source.flight_tracks
    """)
    logger.info(f"Merged flight_tracks (Rows affected: {cursor.rowcount})")

    # 2. Merge anomalous_tracks
    # Check if source has anomalies
    try:
        cursor.execute("SELECT count(*) FROM source.anomalous_tracks")
        has_anomalies = True
    except sqlite3.OperationalError:
        has_anomalies = False
    
    if has_anomalies:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS main.anomalous_tracks (
                flight_id TEXT, timestamp INTEGER, lat REAL, lon REAL, alt REAL,
                heading REAL, gspeed REAL, vspeed REAL, track REAL, squawk TEXT,
                callsign TEXT, source TEXT,
                PRIMARY KEY (flight_id, timestamp)
            )
        """)
        
        cursor.execute("PRAGMA table_info(anomalous_tracks)")
        target_anom_cols = [row[1] for row in cursor.fetchall()]
        
        cursor.execute("PRAGMA source.table_info(anomalous_tracks)")
        source_anom_cols = [row[1] for row in cursor.fetchall()]
        
        common_anom_cols = list(set(target_anom_cols) & set(source_anom_cols))
        anom_cols_str = ", ".join(common_anom_cols)
        
        logger.info(f"Merging anomalous_tracks using columns: {anom_cols_str}")
        cursor.execute(f"""
            INSERT OR IGNORE INTO main.anomalous_tracks ({anom_cols_str})
            SELECT {anom_cols_str} FROM source.anomalous_tracks
        """)
        logger.info(f"Merged anomalous_tracks (Rows affected: {cursor.rowcount})")
    
    conn_target.commit()
    conn_target.close()

def main():
    print("=== Starting Full Process Pipeline ===")
    
    # 1. Run Rules
    print("\n--- Step 1: Running Rule Engine ---")
    # We need to set PYTHONPATH so rules can import core
    env = os.environ.copy()
    env["PYTHONPATH"] = ".." 
    
    try:
        # Note: run_rules.py relies on relative paths, so we run it from the rules dir
        run_command([sys.executable, "run_rules.py"], cwd=RULES_DIR, env=env)
    except Exception as e:
        logger.error(f"Rule engine failed: {e}")
        return

    # 2. Move Anomalies
    print("\n--- Step 2: Moving Anomalies in Source DB ---")
    try:
        # move_anomalies.py needs to point to the source DB
        # From rules dir, source db is ../../flight_tracks_new.db
        run_command(
            [sys.executable, "move_anomalies.py", "--db", "../../flight_tracks_new.db"], 
            cwd=RULES_DIR
        )
    except Exception as e:
        logger.error(f"Moving anomalies failed: {e}")
        return

    # 3. Merge Data
    print("\n--- Step 3: Merging Data to Training DB ---")
    try:
        ensure_schema()
        merge_databases()
    except Exception as e:
        logger.error(f"Merging failed: {e}")
        return

    # 4. Retrain Models
    print("\n--- Step 4: Retraining Models ---")
    try:
        # Run from root (service dir)
        run_command([sys.executable, "training_ops/retrain.py"], cwd=ROOT_DIR)
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        return

    print("\n=== Full Process Completed Successfully ===")

if __name__ == "__main__":
    main()

