from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move all flights from anomalous_tracks back to flight_tracks table."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("../last.db"),
        help="Path to SQLite DB"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving the data"
    )
    return parser.parse_args()


def move_flights_back(
    conn: sqlite3.Connection,
    dry_run: bool = False
) -> Dict[str, int]:
    """Move all flights from anomalous_tracks back to flight_tracks table."""
    stats = {
        "moved_flights": 0,
        "moved_rows": 0,
    }
    
    # Get all unique flight_ids from anomalous_tracks
    cursor = conn.execute("SELECT DISTINCT flight_id FROM anomalous_tracks")
    flight_ids = [row[0] for row in cursor.fetchall()]
    
    if not flight_ids:
        logger.info("No flights found in anomalous_tracks.")
        return stats

    for flight_id in flight_ids:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM anomalous_tracks WHERE flight_id = ?",
            (flight_id,)
        )
        row_count = cursor.fetchone()[0]
        
        if dry_run:
            logger.info(f"[DRY RUN] Would move flight {flight_id} ({row_count} rows) back to flight_tracks")
            stats["moved_flights"] += 1
            stats["moved_rows"] += row_count
            continue
            
        try:
            # Insert into flight_tracks
            # Note: We are inserting NULL for 'heading' as it is missing in anomalous_tracks
            conn.execute("""
                INSERT INTO flight_tracks (
                    flight_id, timestamp, lat, lon, alt, gspeed, vspeed, 
                    track, squawk, callsign, source, heading
                )
                SELECT 
                    flight_id, timestamp, lat, lon, alt, gspeed, vspeed, 
                    track, squawk, callsign, source, NULL
                FROM anomalous_tracks
                WHERE flight_id = ?
            """, (flight_id,))
            
            # Delete from anomalous_tracks
            conn.execute(
                "DELETE FROM anomalous_tracks WHERE flight_id = ?",
                (flight_id,)
            )
            
            stats["moved_flights"] += 1
            stats["moved_rows"] += row_count
            
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error moving flight {flight_id}: {e}")
            raise
            
    conn.commit()
    return stats


def main() -> None:
    args = parse_args()
    
    if not args.db.exists():
        raise FileNotFoundError(f"Database file not found: {args.db}")
    
    conn = sqlite3.connect(str(args.db))
    
    try:
        if args.dry_run:
            logger.info("=== DRY RUN MODE ===")
        
        stats = move_flights_back(conn, dry_run=args.dry_run)
        
        print("\n=== Reverse Migration Summary ===")
        print(f"Flights moved back: {stats['moved_flights']}")
        print(f"Total rows moved back: {stats['moved_rows']}")
        
        if args.dry_run:
            print("\n[DRY RUN] No data was actually moved.")
        else:
            print(f"\nSuccessfully moved {stats['moved_flights']} flights from anomalous_tracks to flight_tracks.")
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()

