from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

# Setup paths
# We assume we are running from the root of the repo (service/)
# We need to ensure that the root is in sys.path so 'core' and 'service' can be imported.

ROOT_DIR = Path(__file__).resolve().parents[3] # service/service/llm_full_pipeline/runners -> service
sys.path.insert(0, str(ROOT_DIR))

# Also add the inner service directory if needed, though usually root is enough
# But 'core' is in root.

from service.llm_full_pipeline.evaluator.comprehensive_evaluator import ComprehensiveEvaluator
from service.llm_full_pipeline.evaluator.dataset_config import DATASET

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive anomaly detection evaluation.")
    parser.add_argument("--db", type=str, default="last.db", help="Path to flight database")
    args = parser.parse_args()

    print(f"Root Dir: {ROOT_DIR}")
    print(f"Sys Path: {sys.path[0]}")
    print("Starting comprehensive evaluation...")
    print(f"Dataset size: {sum(len(v) for v in DATASET.values())} flights")
    
    evaluator = ComprehensiveEvaluator(db_path=Path(args.db))
    summary = evaluator.evaluate_dataset(DATASET)
    
    print("\n--- Evaluation Complete ---")
    print(f"Total Flights: {summary.total_flights}")
    
    print("\nMetrics by Layer:")
    for layer, m in summary.metrics_by_layer.items():
        print(f"\n{layer.upper()}:")
        print(f"  Accuracy: {m.accuracy:.2%}")
        print(f"  Precision: {m.precision:.2%}")
        print(f"  Recall: {m.recall:.2%}")
        print(f"  FP: {m.fp}, FN: {m.fn}, TP: {m.tp}, TN: {m.tn}")

if __name__ == "__main__":
    main()
