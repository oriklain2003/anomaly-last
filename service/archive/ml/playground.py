from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# When running as a script, ensure we can import from the project root
if __name__ == "__main__" and __package__ is None:
    # Add the service directory (project root) to sys.path
    # This assumes ml/playground.py is being run, so parent.parent is 'service'
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        
    from ml.features import FeatureExtractor
    from ml.isolation_model import IsolationForestAnomalyModel
    from core.db import DbConfig, FlightRepository
    from core.models import FlightTrack
else:
    # Standard relative imports when part of a package
    from ..core.db import DbConfig, FlightRepository
    from ..core.models import FlightTrack
    from .features import FeatureExtractor
    from .isolation_model import IsolationForestAnomalyModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test trained ML model on a specific flight.")
    parser.add_argument("flight_id", type=str, help="ID of the flight to analyze")
    parser.add_argument("--db", type=Path, default=Path("../last.db"), help="SQLite DB containing flight_tracks table")
    parser.add_argument("--table", type=str, default="anomalous_tracks", help="Table to search for the flight")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory containing trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Fix path resolution if running from different directories
    if args.model_dir.is_absolute():
        model_dir = args.model_dir
    else:
        # Assume relative to current working directory
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            # Try relative to the script location
            model_dir = Path(__file__).parent / args.model_dir

    model_path = model_dir / "iforest.joblib"
    normalizer_path = model_dir / "normalizers.json"
    
    if not model_path.exists() or not normalizer_path.exists():
        print(f"Error: Model files not found in {model_dir}")
        print("Please run 'python ml/run.py' first to train the model.")
        sys.exit(1)

    # Handle DB path resolution similarly
    db_path = args.db
    if not db_path.exists() and not db_path.is_absolute():
         # Try relative to script location's parent (service root)
         potential_db = Path(__file__).resolve().parent.parent / args.db
         if potential_db.exists():
             db_path = potential_db

    print(f"Loading flight {args.flight_id} from table '{args.table}' in {db_path}...")
    repo = FlightRepository(DbConfig(path=db_path, table=args.table))
    try:
        flight = repo.fetch_flight(args.flight_id)
    except Exception as e:
        print(f"Error accessing DB: {e}")
        sys.exit(1)
    
    if not flight.points:
        print(f"Error: Flight {args.flight_id} not found or has no points.")
        sys.exit(1)
        
    print(f"Loaded {len(flight.points)} track points.")
    
    print("Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_flight_features(flight)
    df = pd.DataFrame(features)
    
    print(f"Loading model from {model_path}...")
    # We don't need to pass feature_cols here because they are stored in the model file
    model = IsolationForestAnomalyModel.load(model_path, normalizer_path)
    
    print("Evaluating flight...")
    point_labels = model.predict_labels(df)
    anomaly_scores = model.predict_scores(df)
    
    # Add results to dataframe for display
    df["is_anomaly"] = point_labels
    df["anomaly_score"] = anomaly_scores
    
    # Calculate statistics
    n_total = len(df)
    n_anomalies = point_labels.sum()
    anomaly_rate = n_anomalies / n_total
    
    # Flight-level decision (using same logic as run.py)
    is_flight_anomalous = anomaly_rate > 0.05
    
    print("\n=== Analysis Results ===")
    print(f"Flight ID:      {args.flight_id}")
    print(f"Total Points:   {n_total}")
    print(f"Anomalous Pts:  {n_anomalies}")
    print(f"Anomaly Rate:   {anomaly_rate:.2%}")
    print(f"Flight Status:  {'🔴 ANOMALOUS' if is_flight_anomalous else '🟢 NORMAL'}")
    
    if n_anomalies > 0:
        print("\n=== Top 5 Anomalous Points ===")
        cols_to_show = ["timestamp", "alt", "cum_turn_300", "avg_speed_300", "anomaly_score"]
        # Filter columns that exist
        cols_to_show = [c for c in cols_to_show if c in df.columns]
        top_anomalies = df[df["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False).head(5)
        print(top_anomalies[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
