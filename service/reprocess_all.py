import subprocess
import sqlite3
import logging
import sys
import os
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("reprocess_all")

# Configuration
ROOT_DIR = Path(__file__).resolve().parent
RULES_DIR = ROOT_DIR / "rules"
TRAINING_DIR = ROOT_DIR / "training_ops"

ORIGINAL_DB_PATH = ROOT_DIR / "last.db"
TEMP_DB_PATH = ROOT_DIR / "temp_recalc.db"

def run_command(command, cwd=None, env=None):
    """Run a shell command and stream output."""
    logger.info(f"Running: {' '.join(command)} in {cwd or '.'}")
    try:
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

def prepare_temp_db():
    """
    Creates temp_recalc.db.
    Merges flight_tracks and anomalous_tracks from last.db into
    a SINGLE flight_tracks table in temp_recalc.db (treating all as candidates).
    """
    logger.info(f"Preparing {TEMP_DB_PATH} from {ORIGINAL_DB_PATH}")
    
    if TEMP_DB_PATH.exists():
        TEMP_DB_PATH.unlink()
        
    conn_temp = sqlite3.connect(str(TEMP_DB_PATH))
    cursor = conn_temp.cursor()
    
    # Create flight_tracks table (copy schema from original if possible, or generic)
    # We use a generic schema that covers our needs plus heading
    cursor.execute("""
        CREATE TABLE flight_tracks (
            flight_id TEXT,
            timestamp INTEGER,
            lat REAL,
            lon REAL,
            alt REAL,
            heading REAL,
            gspeed REAL,
            vspeed REAL,
            track REAL,
            squawk TEXT,
            callsign TEXT,
            source TEXT,
            PRIMARY KEY (flight_id, timestamp)
        )
    """)
    
    # Also create anomalous_tracks (empty for now)
    cursor.execute("""
        CREATE TABLE anomalous_tracks (
            flight_id TEXT,
            timestamp INTEGER,
            lat REAL,
            lon REAL,
            alt REAL,
            heading REAL,
            gspeed REAL,
            vspeed REAL,
            track REAL,
            squawk TEXT,
            callsign TEXT,
            source TEXT,
            PRIMARY KEY (flight_id, timestamp)
        )
    """)

    # Attach original DB
    if not ORIGINAL_DB_PATH.exists():
        logger.error(f"{ORIGINAL_DB_PATH} does not exist!")
        sys.exit(1)
        
    cursor.execute("ATTACH DATABASE ? AS original", (str(ORIGINAL_DB_PATH),))
    
    # Get columns for dynamic insert to avoid mismatches
    cursor.execute("PRAGMA table_info(flight_tracks)")
    dest_cols = [row[1] for row in cursor.fetchall()]
    cols_str = ", ".join(dest_cols)
    
    # 1. Copy Normal Flights
    logger.info("Copying normal flights...")
    # Check if original.flight_tracks exists
    try:
        cursor.execute(f"""
            INSERT OR IGNORE INTO main.flight_tracks ({cols_str})
            SELECT {cols_str} FROM original.flight_tracks
        """)
        logger.info(f"Copied {cursor.rowcount} normal rows.")
    except sqlite3.Error as e:
        logger.warning(f"Error copying normal flights: {e}")

    # 2. Copy Anomalous Flights (treat as candidates for re-evaluation)
    logger.info("Copying anomalous flights to candidate table...")
    try:
        cursor.execute(f"""
            INSERT OR IGNORE INTO main.flight_tracks ({cols_str})
            SELECT {cols_str} FROM original.anomalous_tracks
        """)
        logger.info(f"Copied {cursor.rowcount} anomalous rows back to candidate pool.")
    except sqlite3.Error as e:
        logger.warning(f"Error copying anomalous flights (maybe table doesn't exist): {e}")
        
    conn_temp.commit()
    conn_temp.close()

def replace_original_db():
    """Backs up original DB and replaces it with the new one."""
    logger.info("Replacing last.db with reprocessed data...")
    backup_path = ORIGINAL_DB_PATH.with_suffix(".db.bak")
    if ORIGINAL_DB_PATH.exists():
        shutil.move(ORIGINAL_DB_PATH, backup_path)
        logger.info(f"Backed up original DB to {backup_path}")
        
    shutil.move(TEMP_DB_PATH, ORIGINAL_DB_PATH)
    logger.info("Restored last.db from temp DB.")

def main():
    print("=== Starting Complete Reprocessing Pipeline ===")
    
    # 1. Consolidate Data
    print("\n--- Step 1: Consolidate Data to Temp DB ---")
    prepare_temp_db()
    
    # 2. Run Rules on Temp DB
    print("\n--- Step 2: Running Rule Engine on Temp DB ---")
    env = os.environ.copy()
    env["PYTHONPATH"] = ".." 
    
    # Pass relative path to temp db (from rules dir)
    # rules dir is service/rules. temp db is service/temp_recalc.db
    # so relative path is ../temp_recalc.db
    try:
        run_command(
            [sys.executable, "run_rules.py", "--db", "../temp_recalc.db"], 
            cwd=RULES_DIR, 
            env=env
        )
    except Exception:
        logger.error("Rule engine failed. Aborting.")
        return

    # 3. Move Anomalies in Temp DB
    print("\n--- Step 3: Separating Anomalies in Temp DB ---")
    try:
        run_command(
            [sys.executable, "move_anomalies.py", "--db", "../temp_recalc.db"],
            cwd=RULES_DIR
        )
    except Exception:
        logger.error("Move anomalies failed. Aborting.")
        return

    # 4. Replace Original DB
    print("\n--- Step 4: Updating Main Database ---")
    replace_original_db()

    # 5. Retrain Models
    print("\n--- Step 5: Retraining Models ---")
    try:
        run_command([sys.executable, "training_ops/retrain.py"], cwd=ROOT_DIR)
    except Exception:
        logger.error("Retraining failed.")
        return

    print("\n=== Reprocessing Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()

