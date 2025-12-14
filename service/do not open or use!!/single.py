import sqlite3
import json
import sys
import os
from pathlib import Path

# Add root to path to import existing modules
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Fix DLL load error on Windows by importing torch first
try:
    import torch
except ImportError:
    pass
except OSError:
    pass

from flight_fetcher import deserialize_flight
from new_service.pipeline import get_enhanced_pipeline

CACHE_DB_PATH = Path("flight_cache.db")


def run_test(flight_id):
    print(f"--- Loading Flight {flight_id} from Cache ---")
    if not CACHE_DB_PATH.exists():
        print("Cache DB not found!")
        return

    conn = sqlite3.connect(str(CACHE_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        print(f"Flight {flight_id} not found in cache.")
        return

    flight = deserialize_flight(row[0])
    print(f"Loaded flight with {len(flight.points)} points.")

    print("\n--- Running Enhanced Pipeline ---")
    pipeline = get_enhanced_pipeline()

    # Force the base pipeline to be initialized if it isn't already
    # (happens inside get_enhanced_pipeline -> EnhancedPipeline.__init__)

    result = pipeline.analyze(flight)

    print("\n--- Analysis Result ---")

    # Check if new layers ran
    if "llm_analysis" in result:
        print("\n[GLITCH ANALYSIS]")
        print(json.dumps(result.get("glitch_analysis"), indent=2))

        print("\n[LLM ANALYSIS]")
        llm = result.get("llm_analysis", {})
        print(f"Judgment: {llm.get('logical_judgment')}")
        print(f"Explanation: {llm.get('explanation')}")
        print(f"Score: {llm.get('logical_anomaly_score')}")
        print(f"Is Anomaly: {llm.get('is_anomaly')}")
        print("-" * 20)
        print(f"Reasoning: {llm.get('reasoning')}")
    else:
        print("\n[Base Pipeline Only]")
        print("The flight was NOT flagged as an anomaly by the base pipeline, so Glitch/LLM layers were skipped.")
        summary = result.get("summary", {})
        print(f"Is Anomaly: {summary.get('is_anomaly')}")
        print(f"Confidence Score: {summary.get('confidence_score')}")
        print(f"Triggers: {summary.get('triggers')}")


if __name__ == "__main__":
    run_test("3ad265b4")

