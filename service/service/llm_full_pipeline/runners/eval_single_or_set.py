from __future__ import annotations

import sys
import os
import webbrowser
from pathlib import Path
from datetime import datetime
import json
import dataclasses

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from service.llm_full_pipeline.evaluator.comprehensive_evaluator import ComprehensiveEvaluator, EvaluationSummary

# ==========================================
#  HARD-CODED FLIGHT LISTS (EDIT THESE)
# ==========================================

NORMAL_FLIGHTS = [
"3d39341b",
"3d3a6ff2",
"3d3e8f59",
"3d40188e",
"3d401b74",
"3ac6c17d",
"3d39854d",
"3ace76ec",
"3acde608",
"3ace9b3f",
"3ace768b",
"3ace6188",
"3ace66eb",
"3ace7da4",
"3ace8f18",
"3ace95ad",
"3ace7846",
"3ace5e13",
"3ace4f6c",
"3ace4fb0",
"3ace4407",
"3ace10f7",
"3acdcba6",
"3acde194",
"3acd34d6",
"3ad0256b",
"3ad01276",
"3acff4b8",
]

ANOMALY_FLIGHTS = [
"3d3e0536",
"3ace03d1",
"3acea60f",
"3acdaf78",
"3ace9bfd",
"3ace8fa2",
"3ace8df0",
"3ace87ed",
"3acda3a3",
"3acc94d7",
"3accf024",
"3bc6854c",
]

GLITCH_FLIGHTS = [
"3ace9bcf",
"3ace887b",
]

BORDERLINE_FLIGHTS = [
"3acea6ba",
"3ace756d",
"3aceaca8",
"3ace16cf",
"3acdc6e0",
]

# ==========================================

def main():
    print("Starting Ad-hoc Evaluation...")
    
    # Build dataset
    dataset = {
        "normal": NORMAL_FLIGHTS,
        "anomaly": ANOMALY_FLIGHTS,
        "glitch": GLITCH_FLIGHTS,
        "borderline": BORDERLINE_FLIGHTS
    }
    
    total = sum(len(v) for v in dataset.values())
    if total == 0:
        print("No flights defined in lists! Please edit the script to add flight IDs.")
        return

    print(f"Evaluating {total} flights...")
    
    # Run evaluation
    # We assume DB is at standard location or passed via some config if needed.
    # Here we default to 'last.db' in root or let the evaluator find it.
    db_path = ROOT_DIR / "training_ops/training_dataset.db"
    if not db_path.exists():
        print(f"Warning: {db_path} not found. Evaluator might fail if DB is missing.")
    
    evaluator = ComprehensiveEvaluator(db_path=db_path)
    summary = evaluator.evaluate_dataset(dataset)
    
    # Save with adhoc suffix
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(__file__).resolve().parent.parent / "evaluator" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{timestamp}_adhoc.json"
    
    data = dataclasses.asdict(summary)
    report_path.write_text(json.dumps(data, indent=2))
    print(f"Report saved to {report_path}")
    
    # Open Dashboard
    dashboard_url = "http://localhost:8085/audit_dashboard.html"
    print(f"Opening dashboard at {dashboard_url}...")
    webbrowser.open(dashboard_url)

if __name__ == "__main__":
    main()



