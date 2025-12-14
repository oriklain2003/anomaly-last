import argparse
import json
import logging
import sys
import os
import io
from pathlib import Path

# Ensure stdout uses utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add the project root to sys.path to allow imports from core
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from core.db import DbConfig, FlightRepository
from rules.rule_engine import AnomalyRuleEngine
import flight_fetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test anomaly rules on a specific flight using live/fetched data.")
    parser.add_argument("flight_id", type=str, help="The Flight ID to test")
    parser.add_argument("--db", type=str, default="../last.db", help="Path to the database file (for rule engine context)")
    parser.add_argument("--rules", type=str, default="../anomaly_rule.json", help="Path to the rules JSON file")
    
    args = parser.parse_args()

    # Resolve paths relative to the script location if they are relative paths
    script_dir = Path(__file__).parent
    
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = (script_dir / args.db).resolve()
        
    rules_path = Path(args.rules)
    if not rules_path.is_absolute():
        rules_path = (script_dir / args.rules).resolve()

    logger.info(f"Using Database for context: {db_path}")
    logger.info(f"Using Rules: {rules_path}")
    
    if not rules_path.exists():
        logger.error(f"Rules file not found at {rules_path}")
        sys.exit(1)

    try:
        # Initialize repository and engine
        # We still initialize repository for the engine in case rules need historical context
        repository = None
        if db_path.exists():
            repository = FlightRepository(DbConfig(path=db_path))
        else:
            logger.warning(f"Database file not found at {db_path}. Engine will run without historical context.")

        engine = AnomalyRuleEngine(repository, rules_path)

        # Fetch flight data using flight_fetcher
        logger.info(f"Fetching data for flight {args.flight_id} using flight_fetcher...")
        try:
            track = flight_fetcher.get(args.flight_id)
        except Exception as e:
            logger.error(f"Failed to fetch flight data: {e}")
            sys.exit(1)

        if not track or not track.points:
            logger.error(f"Flight {args.flight_id} not found or has no points.")
            sys.exit(1)

        logger.info(f"Flight {track.flight_id} fetched with {len(track.points)} points.")

        # Evaluate flight
        # First check gateway filters
        should_filter, reason = engine.apply_gateway_filters(track)
        if should_filter:
            logger.warning(f"Flight would be filtered by gateway filters: {reason}")
            # We continue anyway to test the rules as requested
        
        logger.info("Evaluating rules...")
        report = engine.evaluate_track(track)
        
        # Output results
        print("\n" + "="*50)
        print(f"RULE EVALUATION RESULTS FOR FLIGHT: {track.flight_id}")
        print("="*50)
        
        matched_count = len(report["matched_rules"])
        total_count = report["total_rules"]
        
        print(f"\nSummary: {matched_count} rules matched out of {total_count} checked.")
        
        if matched_count > 0:
            print("\nMatched Rules:")
            for rule in report["matched_rules"]:
                print(f"  [X] Rule {rule['id']}: {rule['name']}")
                print(f"      Summary: {rule['summary']}")
                # print(f"      Details: {rule['details']}") # Uncomment for more verbose output
                print("-" * 30)
        else:
            print("\nNo anomalies detected by current rules.")

        print("\n" + "="*50)

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
