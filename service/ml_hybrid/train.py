import sys
import argparse
import json
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler

# Add root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.db import DbConfig, FlightRepository
from ml_deep.preprocessing import TrajectoryResampler
from ml_hybrid.model import HybridAutoencoder

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid Anomaly Detector")
    parser.add_argument("--db", type=Path, default=Path("last.db"), help="Path to last.db")
    parser.add_argument("--output-dir", type=Path, default=Path("ml_hybrid/output"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of flights")
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading Flight Data...")
    if not args.db.exists():
        print(f"Error: {args.db} not found.")
        return

    repo = FlightRepository(DbConfig(path=args.db, table="flight_tracks"))
    flights = list(repo.iter_flights(limit=args.limit, min_points=50))
    
    if not flights:
        print("No flights found.")
        return
        
    print(f"Loaded {len(flights)} flights.")

    # 2. Preprocess
    print("Resampling and Extracting Features...")
    resampler = TrajectoryResampler(num_points=50)
    
    vectors = []
    valid_count = 0
    
    # Features: Lat, Lon, Alt, GSpeed
    feature_cols = ["lat", "lon", "alt", "gspeed"]
    
    for f in flights:
        df = resampler.process(f)
        if not df.empty:
            # Extract (50, 4)
            mat = df[feature_cols].values
            vectors.append(mat)
            valid_count += 1
            
    if not vectors:
        print("No valid vectors generated.")
        return
        
    # Shape: (N, 50, 4)
    X = np.array(vectors)
    print(f"Data Shape: {X.shape}")
    
    # 3. Scale Data
    # We need to scale features. Since it's 3D, we reshape to 2D, scale, then reshape back.
    N, Seq, Feat = X.shape
    X_flat = X.reshape(N * Seq, Feat)
    
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(N, Seq, Feat)
    
    # Save Scaler
    joblib.dump(scaler, args.output_dir / "scaler.joblib")
    print("Scaler saved.")
    
    # 4. Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    dataset = TensorDataset(torch.FloatTensor(X_scaled))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = HybridAutoencoder(input_dim=Feat, seq_len=Seq).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            
            recon = model(x_batch)
            # MSE Loss
            loss = torch.mean((x_batch - recon) ** 2)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs}: Loss {total_loss/len(loader):.6f}")
        
    # 5. Determine Threshold
    model.eval()
    with torch.no_grad():
        x_all = torch.FloatTensor(X_scaled).to(device)
        # Get reconstruction error per sample (mean over seq and feat)
        errors = model.get_reconstruction_error(x_all).cpu().numpy()
        
    # Set threshold at 99th percentile
    threshold = float(np.quantile(errors, 0.99))
    print(f"Anomaly Threshold (99th percentile): {threshold:.6f}")
    
    # Save Artifacts
    torch.save(model.state_dict(), args.output_dir / "hybrid_model.pth")
    joblib.dump(threshold, args.output_dir / "threshold.joblib")
    
    print(f"Model and artifacts saved to {args.output_dir}")

if __name__ == "__main__":
    main()
