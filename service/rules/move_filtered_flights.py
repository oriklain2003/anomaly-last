from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move filtered flight IDs from flight_tracks to filtered_tracks table."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("../last.db"),
        help="Path to SQLite DB"
    )
    parser.add_argument(
        "--filtered-ids-file",
        type=Path,
        default=Path("filtered_flight_ids.json"),
        help="Path to JSON file containing filtered flight IDs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving the data"
    )
    return parser.parse_args()


def create_filtered_tracks_table(conn: sqlite3.Connection) -> None:
    """Create the filtered_tracks table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS filtered_tracks (
            flight_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            alt REAL NOT NULL,
            gspeed REAL,
            vspeed REAL,
            track REAL,
            squawk TEXT,
            callsign TEXT,
            source TEXT,
            PRIMARY KEY (flight_id, timestamp)
        )
    """)
    conn.commit()
    logger.info("Ensured filtered_tracks table exists")


def load_filtered_ids(file_path: Path) -> List[str]:
    """Load filtered flight IDs from JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Filtered IDs file not found: {file_path}")
    
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Support both formats: {"filtered_flights": [...]} or just a list
    if isinstance(data, dict) and "filtered_flights" in data:
        filtered_flights = data["filtered_flights"]
        # Extract flight_id if it's a dict with flight_id key
        if filtered_flights and isinstance(filtered_flights[0], dict):
            flight_ids = [f["flight_id"] for f in filtered_flights]
        else:
            flight_ids = filtered_flights
    elif isinstance(data, list):
        flight_ids = data
    else:
        raise ValueError(f"Unexpected JSON format in {file_path}")
    
    logger.info(f"Loaded {len(flight_ids)} filtered flight IDs from {file_path}")
    return flight_ids


def move_flights_to_filtered_table(
    conn: sqlite3.Connection,
    flight_ids: List[str],
    dry_run: bool = False
) -> Dict[str, int]:
    """Move flights from flight_tracks to filtered_tracks table."""
    stats = {
        "moved_flights": 0,
        "moved_rows": 0,
        "skipped_flights": 0
    }
    
    for flight_id in flight_ids:
        # Check if flight exists in flight_tracks
        cursor = conn.execute(
            "SELECT COUNT(*) FROM flight_tracks WHERE flight_id = ?",
            (flight_id,)
        )
        row_count = cursor.fetchone()[0]
        
        if row_count == 0:
            logger.warning(f"Flight {flight_id} not found in flight_tracks table")
            stats["skipped_flights"] += 1
            continue
        
        if dry_run:
            logger.info(f"[DRY RUN] Would move flight {flight_id} ({row_count} rows)")
            stats["moved_flights"] += 1
            stats["moved_rows"] += row_count
            continue
        
        # Move data: INSERT into filtered_tracks, then DELETE from flight_tracks
        try:
            # Insert into filtered_tracks
            conn.execute("""
                INSERT OR IGNORE INTO filtered_tracks
                SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, 
                       track, squawk, callsign, source
                FROM flight_tracks
                WHERE flight_id = ?
            """, (flight_id,))
            
            # Delete from flight_tracks
            conn.execute(
                "DELETE FROM flight_tracks WHERE flight_id = ?",
                (flight_id,)
            )
            
            conn.commit()
            
            stats["moved_flights"] += 1
            stats["moved_rows"] += row_count
            logger.info(f"Moved flight {flight_id} ({row_count} rows) to filtered_tracks")
            
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error moving flight {flight_id}: {e}")
            stats["skipped_flights"] += 1
    
    return stats


def main() -> None:
    args = parse_args()
    
    if not args.db.exists():
        raise FileNotFoundError(f"Database file not found: {args.db}")
    
    # Load filtered flight IDs
    try:
        flight_ids = load_filtered_ids(args.filtered_ids_file)
    except FileNotFoundError:
        logger.error(f"Filtered IDs file not found: {args.filtered_ids_file}")
        logger.info("Run run_rules.py first to generate filtered_flight_ids.json")
        return
    
    if not flight_ids:
        logger.warning("No filtered flight IDs to process")
        return
    
    # Connect to database
    conn = sqlite3.connect(str(args.db))
    
    try:
        # Create filtered_tracks table if it doesn't exist
        create_filtered_tracks_table(conn)
        
        # Move flights
        if args.dry_run:
            logger.info("=== DRY RUN MODE ===")
        
        stats = move_flights_to_filtered_table(conn, flight_ids, dry_run=args.dry_run)
        
        # Print summary
        print("\n=== Migration Summary ===")
        print(f"Flights processed: {len(flight_ids)}")
        print(f"Flights moved: {stats['moved_flights']}")
        print(f"Total rows moved: {stats['moved_rows']}")
        print(f"Flights skipped: {stats['skipped_flights']}")
        
        if args.dry_run:
            print("\n[DRY RUN] No data was actually moved. Remove --dry-run to perform the migration.")
        else:
            print(f"\nSuccessfully moved {stats['moved_flights']} flights to filtered_tracks table.")
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()

