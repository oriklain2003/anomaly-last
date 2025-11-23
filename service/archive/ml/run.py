from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict

import numpy as np
import pandas as pd

# Check if we are running as a script
if __name__ == "__main__":
    # Add the parent directory to sys.path to allow imports
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.features import FeatureExtractor
from ml.isolation_model import IsolationForestAnomalyModel
from ml.normalization import NormalizationStats
from core.db import DbConfig, FlightRepository
from core.models import FlightTrack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate IsolationForest on flight tracks.")
    parser.add_argument("--db", type=Path, default=Path("../last.db"), help="SQLite DB containing flight_tracks table")
    parser.add_argument("--table", type=str, default="flight_tracks", help="Table name with track points")
    parser.add_argument("--anomalies-table", type=str, default="anomalous_tracks", help="Table name with KNOWN anomalous track points (ground truth)")
    parser.add_argument("--limit", type=int, help="Limit number of flights to load for quicker experimentation")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of flights for training (rest for testing)")
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly rate (contamination)")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory to store trained models")
    return parser.parse_args()


def collect_flights(repository: FlightRepository, limit: int | None = None) -> List[FlightTrack]:
    flights = list(repository.iter_flights(limit=limit, min_points=10))
    flights = [f for f in flights if f.points]
    # Sort by timestamp to simulate temporal split if needed, though random split is often fine for iid
    flights.sort(key=lambda f: f.points[0].timestamp if f.points else 0)
    return flights


def split_flights(flights: Sequence[FlightTrack], ratio: float) -> Tuple[List[FlightTrack], List[FlightTrack]]:
    if not flights:
        raise ValueError("No flights available for training/testing.")
    cut = max(1, min(len(flights) - 1, int(len(flights) * ratio)))
    return list(flights[:cut]), list(flights[cut:])


def flights_to_dataframe(flights: Iterable[FlightTrack], extractor: FeatureExtractor) -> pd.DataFrame:
    rows: List[dict] = []
    for flight in flights:
        rows.extend(extractor.extract_flight_features(flight))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    
    # Ensure model directory exists
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "iforest.joblib"
    normalizer_path = args.model_dir / "normalizers.json"

    # 1. Load assumed NORMAL data (or mixed data)
    print(f"Loading training data from '{args.table}'...")
    repo_normal = FlightRepository(DbConfig(path=args.db, table=args.table))
    flights_normal = collect_flights(repo_normal, limit=args.limit)

    if len(flights_normal) < 2:
        raise RuntimeError("Need at least two flights to perform train/test split.")

    print(f"Total normal/mixed flights collected: {len(flights_normal)}")
    train_flights, test_flights_normal = split_flights(flights_normal, args.train_ratio)
    
    # 2. Load KNOWN ANOMALIES (if any)
    print(f"Loading known anomalies from '{args.anomalies_table}'...")
    try:
        repo_anomalies = FlightRepository(DbConfig(path=args.db, table=args.anomalies_table))
        flights_anomalies = collect_flights(repo_anomalies, limit=args.limit)
        print(f"Total known anomalous flights collected: {len(flights_anomalies)}")
    except Exception as e:
        print(f"Warning: Could not load anomalies from {args.anomalies_table} ({e}). Proceeding without ground truth anomalies.")
        flights_anomalies = []
    
    extractor = FeatureExtractor()
    feature_cols = extractor.feature_columns()
    
    print("Extracting features...")
    train_df = flights_to_dataframe(train_flights, extractor)
    test_df_normal = flights_to_dataframe(test_flights_normal, extractor)
    test_df_anomalies = flights_to_dataframe(flights_anomalies, extractor)

    if train_df.empty:
        raise RuntimeError("Insufficient training data.")
    
    print(f"Training data: {len(train_df)} points from {len(train_flights)} flights")
    print(f"Test (Normal): {len(test_df_normal)} points from {len(test_flights_normal)} flights")
    print(f"Test (Anom):   {len(test_df_anomalies)} points from {len(flights_anomalies)} flights")

    print("Training model...")
    normalizer = NormalizationStats.from_dataframe(train_df, feature_cols)
    model = IsolationForestAnomalyModel(feature_cols=feature_cols, contamination=args.contamination)
    model.fit(train_df, normalizer)

    print("\n=== Evaluation Results (Point-Level) ===")
    
    if not test_df_normal.empty:
        pred_normal = model.predict_labels(test_df_normal)
        fpr_point = pred_normal.sum() / len(pred_normal)
        print(f"[Assumed Normal] Point-level Anomaly Rate: {fpr_point:.2%} (Target: ~{args.contamination:.2%})")
    
    if not test_df_anomalies.empty:
        pred_anomalies = model.predict_labels(test_df_anomalies)
        recall_point = pred_anomalies.sum() / len(pred_anomalies)
        print(f"[Known Anomaly]  Point-level Recall:       {recall_point:.2%}")

    print("\n=== Flight-Level Anomaly Rate Distribution ===")
    
    def get_flight_rates(df, model):
        if df.empty: return pd.Series()
        labels = model.predict_labels(df)
        temp = df[["flight_id"]].copy()
        temp["is_anomaly"] = labels
        return temp.groupby("flight_id")["is_anomaly"].mean()

    rates_normal = get_flight_rates(test_df_normal, model)
    rates_anom = get_flight_rates(test_df_anomalies, model)

    if not rates_normal.empty:
        print("\n[Assumed Normal] Anomaly Rate Stats:")
        print(rates_normal.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
        
    if not rates_anom.empty:
        print("\n[Known Anomaly] Anomaly Rate Stats:")
        print(rates_anom.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\n=== Evaluation Results (Flight-Level) ===")
    
    best_threshold = 0.05
    best_score = -1
    
    # Sweep thresholds to find best separation
    thresholds = np.linspace(0.0, 0.5, 51)
    print(f"{'Threshold':<10} | {'FPR (Normal)':<12} | {'Recall (Anom)':<12} | {'Difference':<10}")
    print("-" * 50)
    
    for thresh in thresholds:
        fpr = 0.0
        recall = 0.0
        
        if not rates_normal.empty:
            fpr = (rates_normal > thresh).mean()
        
        if not rates_anom.empty:
            recall = (rates_anom > thresh).mean()
            
        score = recall - fpr
        if score > best_score:
            best_score = score
            best_threshold = thresh
        
        # Print some steps
        if int(thresh * 100) % 5 == 0:
             print(f"{thresh:<10.2f} | {fpr:<12.2%} | {recall:<12.2%} | {score:<10.2f}")

    print("-" * 50)
    print(f"Optimal Flight Threshold (Max Recall-FPR): {best_threshold:.2f}")
    
    # Report stats at optimal threshold
    if not rates_normal.empty:
        n_flagged = (rates_normal > best_threshold).sum()
        fpr = n_flagged / len(rates_normal)
        print(f"[Assumed Normal] Flag Rate: {fpr:.2%} ({n_flagged}/{len(rates_normal)})")
        
    if not rates_anom.empty:
        n_caught = (rates_anom > best_threshold).sum()
        recall = n_caught / len(rates_anom)
        print(f"[Known Anomaly]  Recall:    {recall:.2%} ({n_caught}/{len(rates_anom)})")

    model.save(model_path, normalizer_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
