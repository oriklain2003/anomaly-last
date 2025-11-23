from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Check if we are running as a script
if __name__ == "__main__" and __package__ is None:
    # Add the parent directory to sys.path to allow imports
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ml.features import FeatureExtractor
    from ml.isolation_model import IsolationForestAnomalyModel
    from core.db import DbConfig, FlightRepository
    from core.models import FlightTrack
else:
    from ..core.db import DbConfig, FlightRepository
    from ..core.models import FlightTrack
    from .features import FeatureExtractor
    from .isolation_model import IsolationForestAnomalyModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test trained ML model on a specific flight.")
    # parser.add_argument("flight_id", type=str, help="ID of the flight to analyze", default="3b637311")
    parser.add_argument("--db", type=Path, default=Path("../last.db"), help="SQLite DB containing flight_tracks table")
    parser.add_argument("--table", type=str, default="anomalous_tracks", help="Table to search for the flight")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory containing trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    model_path = args.model_dir / "iforest.joblib"
    normalizer_path = args.model_dir / "normalizers.json"
    
    if not model_path.exists() or not normalizer_path.exists():
        print(f"Error: Model files not found in {args.model_dir}")
        print("Please run 'python ml/run.py' first to train the model.")
        sys.exit(1)

    print(f"Loading flight {'3b637311'} from table '{args.table}'...")
    repo = FlightRepository(DbConfig(path=args.db, table=args.table))
    flight = repo.fetch_flight("3b637311")
    
    if not flight.points:
        print(f"Error: Flight {'3b637311'} not found or has no points.")
        sys.exit(1)
        
    print(f"Loaded {len(flight.points)} track points.")
    
    print("Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_flight_features(flight)
    df = pd.DataFrame(features)
    
    print("Loading model...")
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
    print(f"Flight ID:      {'3b637311'}")
    print(f"Total Points:   {n_total}")
    print(f"Anomalous Pts:  {n_anomalies}")
    print(f"Anomaly Rate:   {anomaly_rate:.2%}")
    print(f"Flight Status:  {'🔴 ANOMALOUS' if is_flight_anomalous else '🟢 NORMAL'}")
    
    if n_anomalies > 0:
        print("\n=== Top 5 Anomalous Points ===")
        cols_to_show = ["timestamp", "alt", "cum_turn_300", "avg_speed_300", "anomaly_score"]
        top_anomalies = df[df["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False).head(5)
        print(top_anomalies[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()

