from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from mlboost.features import FlightAggregator
from mlboost.model import XGBoostAnomalyModel
from playground import get  # Import the get() function to fetch live data

def main():
    print("=== Fetching Live Flight Data ===")
    # 1. Get the flight object directly from your playground code
    flight = get()
    print(f"Fetched Flight: {flight.flight_id} with {len(flight.points)} points")

    if not flight.points:
        print("Error: Flight has no points.")
        return

    # 2. Process the flight into features (Aggregation)
    print("Extracting features...")
    aggregator = FlightAggregator()
    # We don't provide a label because we are predicting
    features_dict = aggregator.extract_flight_row(flight)
    
    if not features_dict:
        print("Error: Could not extract features (flight might be too short).")
        return

    # Convert to DataFrame (single row)
    df = pd.DataFrame([features_dict])
    
    # 3. Load the trained model
    model_path = Path("mlboost/output/xgb_model.joblib")
    print(f"Loading model from {model_path}...")
    try:
        model = XGBoostAnomalyModel.load(model_path)
    except FileNotFoundError:
        print("Error: Model file not found. Please run 'python -m mlboost.run' first to train the model.")
        return

    # 4. Ensure columns match (XGBoost is sensitive to column order/presence)
    # We select only the columns that the model expects
    try:
        # Add missing columns with 0 if necessary (though aggregator should match)
        # But more importantly, select only the model's features in order
        missing_cols = set(model.feature_names) - set(df.columns)
        if missing_cols:
            print(f"Warning: Missing columns filled with 0: {missing_cols}")
            for c in missing_cols:
                df[c] = 0
        
        # Reorder to match training data
        df = df[model.feature_names]
        
    except KeyError as e:
        print(f"Error aligning columns: {e}")
        return

    # 5. Predict
    print("\n=== Prediction Results ===")
    prob = model.predict_proba(df)[0]
    is_anomaly = prob > 0.5
    
    print(f"Anomaly Probability: {prob:.4%}")
    print(f"Prediction:          {'ANOMALY' if is_anomaly else 'NORMAL'}")
    
    # Optional: Explain why? (Simple feature print)
    print("\nKey Flight Stats:")
    print(f"  - Duration:      {features_dict.get('duration', 0)/60:.1f} min")
    print(f"  - Avg Turn Rate: {features_dict.get('turn_rate_mean', 0):.2f} deg/sec")
    print(f"  - Max Turn Rate: {features_dict.get('turn_rate_max', 0):.2f} deg/sec")
    print(f"  - Min Altitude:  {features_dict.get('alt_min', 0):.0f} ft")
    print(f"  - StdDev Turn (5m): {features_dict.get('cum_turn_300_std', 0):.2f}")

if __name__ == "__main__":
    main()

