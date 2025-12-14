import sys
import os

# Fix DLL load error on Windows by importing torch first (try/except block)
try:
    import torch
except ImportError:
    pass
except OSError:
    pass

import sqlite3
import json
from pathlib import Path
import time
import dataclasses

# Setup paths
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from new_service.pipeline import get_enhanced_pipeline
from flight_fetcher import get, deserialize_flight
from core.models import FlightTrack, TrackPoint

DB_RESEARCH_PATH = Path("realtime/research.db")
OUTPUT_FILE = Path("web2/public/test_results.json")

def run_batch_test():
    print("Starting batch test...")

    if not DB_RESEARCH_PATH.exists():
        print(f"Research DB not found at {DB_RESEARCH_PATH}")
        return

    conn = sqlite3.connect(str(DB_RESEARCH_PATH))
    cursor = conn.cursor()

    # Get 50 unique flight_ids that are anomalies
    query = """
        SELECT DISTINCT flight_id FROM anomaly_reports 
        WHERE is_anomaly = 1 
        LIMIT 500
    """
    cursor.execute(query)
    flight_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"Found {len(flight_ids)} anomalies to test.")

    pipeline = get_enhanced_pipeline()
    results = []

    for i, flight_id in enumerate(flight_ids):
        print(f"Processing {i+1}/{len(flight_ids)}: {flight_id}")
        try:
            # Fetch flight data
            # We try to use flight_fetcher.get which handles cache/live
            flight = get(flight_id=flight_id)

            if not flight or not flight.points:
                print(f"  - No data for {flight_id}")
                continue

            # Run pipeline
            start_time = time.time()
            analysis = pipeline.analyze(flight)
            duration = time.time() - start_time

            # Compare old vs new
            # Old result is basically analysis without 'llm_analysis' and 'glitch_analysis'
            # But since we ran the enhanced pipeline, 'analysis' contains everything.

            old_is_anomaly = True # We selected from anomalies, so it was an anomaly.

            summary = analysis.get("summary", {})
            base_is_anomaly = summary.get("is_anomaly", False)

            # Check if LLM/Glitch changed verdict
            new_is_anomaly = base_is_anomaly
            llm_data = analysis.get("llm_analysis")
            if llm_data:
                # If LLM says it's NOT an anomaly (is_anomaly=False), then verdict changed.
                new_is_anomaly = llm_data.get("is_anomaly", True)

                # Special case for 'normal but noisy' logic
                if llm_data.get("logical_judgment") == "normal but noisy":
                    new_is_anomaly = False

            results.append({
                "flight_id": flight_id,
                "original_anomaly": True, # From DB selection
                "base_pipeline_anomaly": base_is_anomaly,
                "new_pipeline_anomaly": new_is_anomaly,
                "glitch_score": analysis.get("glitch_analysis", {}).get("glitch_score", 0),
                "llm_explanation": llm_data.get("explanation") if llm_data else "N/A",
                "llm_judgment": llm_data.get("logical_judgment") if llm_data else "N/A",
                "full_report": analysis
            })

        except Exception as e:
            print(f"  - Error processing {flight_id}: {e}")

    # Write results
    output_path = Path(root_path) / OUTPUT_FILE
    # Ensure dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_batch_test()
