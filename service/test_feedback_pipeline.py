import sqlite3
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add root directory to path so we can import modules
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from core.models import FlightTrack, TrackPoint
    from anomaly_pipeline import AnomalyPipeline
    from training_ops.db_utils import FEEDBACK_DB_PATH, TRAINING_DB_PATH
except OSError as e:
    print("="*60)
    print("CRITICAL ERROR: Failed to import AnomalyPipeline or dependencies.")
    print(f"Error details: {e}")
    print("="*60)
    print("This appears to be an issue with PyTorch (c10.dll) or other ML libraries.")
    print("Please check your Python environment and installed packages.")
    sys.exit(1)
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_flight_from_db(flight_id: str) -> Optional[FlightTrack]:
    """Load a flight from the training database (checking both tables)."""
    if not TRAINING_DB_PATH.exists():
        logger.error(f"Training DB not found at {TRAINING_DB_PATH}")
        return None

    try:
        conn = sqlite3.connect(str(TRAINING_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check anomalous_tracks first
        cursor.execute("SELECT * FROM anomalous_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
        rows = cursor.fetchall()

        if not rows:
            # Check flight_tracks
            cursor.execute("SELECT * FROM flight_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()
        
        conn.close()

        if not rows:
            return None

        points = []
        for row in rows:
            # Convert row to dict
            p_data = dict(row)
            points.append(TrackPoint(
                flight_id=flight_id,
                timestamp=p_data.get("timestamp"),
                lat=p_data.get("lat"),
                lon=p_data.get("lon"),
                alt=p_data.get("alt"),
                gspeed=p_data.get("gspeed"),
                vspeed=p_data.get("vspeed"),
                track=p_data.get("track"),
                squawk=p_data.get("squawk"),
                callsign=p_data.get("callsign"),
                source=p_data.get("source")
            ))
        
        return FlightTrack(flight_id=flight_id, points=points)

    except Exception as e:
        logger.error(f"Error loading flight {flight_id}: {e}")
        return None

def main():
    if not FEEDBACK_DB_PATH.exists():
        print(f"Feedback DB not found at {FEEDBACK_DB_PATH}")
        return

    print("Initializing Pipeline...")
    pipeline = AnomalyPipeline()
    print("Pipeline Initialized.")

    print(f"Reading feedback from {FEEDBACK_DB_PATH}...")
    try:
        conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT flight_id, user_label FROM user_feedback")
        feedback_rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Error reading feedback DB: {e}")
        return

    print(f"Found {len(feedback_rows)} feedback entries.")

    correct_predictions = []
    wrong_predictions = []
    missing_data_ids = []

    for flight_id, user_label in feedback_rows:
        expected_anomaly = bool(user_label == 1)
        expected_str = "ANOMALY" if expected_anomaly else "NORMAL"
        
        print(f"\nTesting Flight: {flight_id} (Expected: {expected_str})")
        
        flight = load_flight_from_db(flight_id)
        if not flight:
            print(f"  [-] Flight data not found in training DB.")
            missing_data_ids.append(flight_id)
            continue

        if len(flight.points) < 50:
             print(f"  [-] Flight has only {len(flight.points)} points (need 50+). Skipping.")
             continue

        # Run pipeline
        try:
            # Suppress pipeline stdout to keep output clean(er)
            # import io
            # from contextlib import redirect_stdout
            # with redirect_stdout(io.StringIO()): 
            report = pipeline.analyze(flight)
            
            is_anomaly = report["summary"]["is_anomaly"]
            confidence = report["summary"]["confidence_score"]
            triggers = report["summary"]["triggers"]
            
            result_str = "ANOMALY" if is_anomaly else "NORMAL"
            match = (is_anomaly == expected_anomaly)
            
            print(f"  [>] Result: {result_str} (Confidence: {confidence}%)")
            print(f"  [>] Triggers: {triggers}")
            
            if match:
                print("  [+] CORRECT")
                correct_predictions.append({
                    "id": flight_id,
                    "expected": expected_str,
                    "actual": result_str,
                    "confidence": confidence
                })
            else:
                print("  [!] WRONG")
                wrong_predictions.append({
                    "id": flight_id,
                    "expected": expected_str,
                    "actual": result_str,
                    "confidence": confidence
                })

        except Exception as e:
            print(f"  [!] Error running pipeline: {e}")

    # Summary
    total = len(correct_predictions) + len(wrong_predictions)
    accuracy = (len(correct_predictions) / total * 100) if total > 0 else 0

    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    print(f"Total Tested: {total}")
    print(f"Correct: {len(correct_predictions)}")
    print(f"Wrong: {len(wrong_predictions)}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if missing_data_ids:
        print(f"Missing Data for IDs: {len(missing_data_ids)}")

    print("\nCorrect Predictions:")
    for p in correct_predictions:
        print(f"  - {p['id']}: Expected {p['expected']} -> Got {p['actual']} ({p['confidence']}%)")

    print("\nWrong Predictions:")
    for p in wrong_predictions:
        print(f"  - {p['id']}: Expected {p['expected']} -> Got {p['actual']} ({p['confidence']}%)")

if __name__ == "__main__":
    main()

