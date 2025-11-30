import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("merge_db")

SOURCE_DB = Path("../flight_tracks_new.db")
TARGET_DB = Path("last.db")

def merge():
    if not SOURCE_DB.exists():
        logger.error(f"Source DB {SOURCE_DB} not found")
        return
    if not TARGET_DB.exists():
        logger.error(f"Target DB {TARGET_DB} not found")
        return

    logger.info(f"Merging from {SOURCE_DB} to {TARGET_DB}")
    conn_target = sqlite3.connect(TARGET_DB)
    cursor = conn_target.cursor()
    
    # Attach source
    cursor.execute("ATTACH DATABASE ? AS source", (str(SOURCE_DB),))
    
    # Get columns for flight_tracks
    cursor.execute("PRAGMA table_info(flight_tracks)")
    target_cols = [row[1] for row in cursor.fetchall()]
    cols_str = ", ".join(target_cols)
    
    # Merge flight_tracks
    logger.info("Merging flight_tracks...")
    try:
        cursor.execute(f"""
            INSERT OR IGNORE INTO main.flight_tracks ({cols_str})
            SELECT {cols_str} FROM source.flight_tracks
        """)
        logger.info(f"Inserted rows into flight_tracks")
    except Exception as e:
        logger.error(f"Error merging flight_tracks: {e}")

    
    # Check if source has anomalous_tracks
    try:
        cursor.execute("SELECT count(*) FROM source.anomalous_tracks")
        has_anom_source = True
    except sqlite3.OperationalError:
        has_anom_source = False
        logger.info("No anomalous_tracks table in source DB")

    if has_anom_source:
        # Check if target has anomalous_tracks
        try:
            cursor.execute("SELECT count(*) FROM main.anomalous_tracks")
        except sqlite3.OperationalError:
             logger.info("Creating anomalous_tracks in target...")
             # We try to copy structure from source
             cursor.execute("CREATE TABLE main.anomalous_tracks AS SELECT * FROM source.anomalous_tracks WHERE 0")
             conn_target.commit()

        # Get columns for anomalous_tracks
        cursor.execute("PRAGMA table_info(anomalous_tracks)")
        anom_target_cols = [row[1] for row in cursor.fetchall()]
        anom_cols_str = ", ".join(anom_target_cols)

        logger.info("Merging anomalous_tracks...")
        try:
            cursor.execute(f"""
                INSERT OR IGNORE INTO main.anomalous_tracks ({anom_cols_str})
                SELECT {anom_cols_str} FROM source.anomalous_tracks
            """)
            logger.info(f"Inserted rows into anomalous_tracks")
        except Exception as e:
            logger.error(f"Error merging anomalous_tracks: {e}")
    
    conn_target.commit()
    conn_target.close()
    logger.info("Merge complete.")

if __name__ == "__main__":
    merge()

