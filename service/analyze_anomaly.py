
import sqlite3
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from core.models import FlightTrack, TrackPoint
from ml_deep.detector import DeepAnomalyDetector
from ml_transformer.detector import TransformerAnomalyDetector
from ml_deep.preprocessing import TrajectoryResampler

def load_flight(flight_id):
    conn = sqlite3.connect('flight_cache.db')
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print("Flight not found in cache.")
        return None
        
    data = json.loads(row[0])
    points = []
    for p in data['points']:
        points.append(TrackPoint(
            flight_id=flight_id,
            timestamp=p['timestamp'],
            lat=p['lat'],
            lon=p['lon'],
            alt=p.get('alt', 0),
            gspeed=p.get('gspeed', 0),
            track=p.get('track', 0)
        ))
        
    return FlightTrack(flight_id=flight_id, points=points)

def analyze_deep_ae(flight, detector):
    print("\n--- Analyzing Deep Dense AE ---")
    resampler = TrajectoryResampler(num_points=50)
    df = resampler.process(flight)
    
    if df.empty:
        print("Flight too short")
        return
        
    vec = resampler.flatten(df)
    vec_reshaped = vec.reshape(1, -1)
    
    # Cluster
    cluster_id = detector.clusterer.predict(vec_reshaped)[0]
    cluster_id_str = str(cluster_id)
    print(f"Cluster ID: {cluster_id}")
    
    if cluster_id_str not in detector.aes:
        print(f"No model for cluster {cluster_id}")
        return
        
    model = detector.aes[cluster_id_str]
    
    # Normalize
    vec_norm = detector.clusterer.scaler.transform(vec_reshaped)
    
    print(f"Deep AE Cluster {cluster_id} Scaler Mean (first 4): {detector.clusterer.scaler.mean_[:4]}")
    print(f"Deep AE Input (first 4): {vec_reshaped[0,:4]}")
    print(f"Deep AE Normalized (first 4): {vec_norm[0,:4]}")

    tensor_in = torch.FloatTensor(vec_norm).to(detector.device)
    
    with torch.no_grad():
        recon = model(tensor_in)
        mse_loss = torch.mean((tensor_in - recon) ** 2).item()
    
    print(f"MSE Loss: {mse_loss:.6f}")
    print(f"Threshold: {detector.thresholds[cluster_id_str]:.6f}")
    print(f"Is Anomaly: {mse_loss > detector.thresholds[cluster_id_str]}")
    
    # Error analysis
    error_vec = (tensor_in - recon).cpu().numpy().flatten() ** 2
    error_matrix = error_vec.reshape(50, 4) # Lat, Lon, Alt, Track
    
    features = ["lat", "lon", "alt", "track"]
    feature_errors = np.mean(error_matrix, axis=0)
    
    print("Average MSE per feature:")
    for i, f in enumerate(features):
        print(f"  {f}: {feature_errors[i]:.6f}")
        
    # Find worst time steps
    step_errors = np.mean(error_matrix, axis=1)
    worst_steps = np.argsort(step_errors)[-5:][::-1]
    
    print("Worst time steps (0-49):")
    for step in worst_steps:
        print(f"  Step {step}: Error {step_errors[step]:.6f} (Lat: {error_matrix[step,0]:.4f}, Lon: {error_matrix[step,1]:.4f}, Alt: {error_matrix[step,2]:.4f}, Track: {error_matrix[step,3]:.4f})")

def analyze_transformer(flight, detector):
    print("\n--- Analyzing Transformer AE ---")
    resampler = TrajectoryResampler(num_points=50)
    df = resampler.process(flight)
    
    if df.empty:
        print("Flight too short")
        return

    # Cluster (using same logic as deep detector usually, but let's use transformer's own flow)
    # Transformer detector uses the clusterer loaded from its own dir
    vec_flat = resampler.flatten(df).reshape(1, -1)
    cluster_id = str(detector.clusters.predict(vec_flat)[0])
    print(f"Cluster ID: {cluster_id}")
    
    if cluster_id not in detector.models:
        print("No model for cluster")
        return
        
    norm = detector.norms[cluster_id]
    mat = resampler.to_matrix(df)
    
    # Normalize
    # Note: transformer detector normalizes differently: (mat - mean) / std
    # mean shape (1, 1, 4), mat shape (50, 4)
    mat_input = mat.reshape(1, 50, 4)
    mat_norm = (mat_input - norm["mean"]) / norm["std"]
    
    print(f"Cluster 0 Norm Mean: {norm['mean']}")
    print(f"Cluster 0 Norm Std: {norm['std']}")
    print(f"Input First Point: {mat_input[0,0,:]}")
    print(f"Normalized First Point: {mat_norm[0,0,:]}")

    tensor_in = torch.FloatTensor(mat_norm).to(detector.device)
    model = detector.models[cluster_id]
    
    with torch.no_grad():
        recon = model(tensor_in)
        mse_loss = torch.mean((tensor_in - recon) ** 2).item()
        
    print(f"MSE Loss: {mse_loss:.6f}")
    print(f"Threshold: {detector.thresholds[cluster_id]:.6f}")
    print(f"Is Anomaly: {mse_loss > detector.thresholds[cluster_id]}")
    
    # Error analysis
    error_matrix = ((tensor_in - recon) ** 2).cpu().numpy().squeeze() # (50, 4)
    
    features = ["lat", "lon", "alt", "track"]
    feature_errors = np.mean(error_matrix, axis=0)
    
    print("Average MSE per feature:")
    for i, f in enumerate(features):
        print(f"  {f}: {feature_errors[i]:.6f}")
        
    step_errors = np.mean(error_matrix, axis=1)
    worst_steps = np.argsort(step_errors)[-5:][::-1]
    
    print("Worst time steps (0-49):")
    for step in worst_steps:
        print(f"  Step {step}: Error {step_errors[step]:.6f} (Lat: {error_matrix[step,0]:.4f}, Lon: {error_matrix[step,1]:.4f}, Alt: {error_matrix[step,2]:.4f}, Track: {error_matrix[step,3]:.4f})")

def main():
    flight_id = '3d3a4a0e'
    flight = load_flight(flight_id)
    if not flight:
        return

    # Initialize Detectors
    print("Loading Deep Detector...")
    deep_detector = DeepAnomalyDetector(Path("ml_deep/output"))
    
    print("Loading Transformer Detector...")
    trans_detector = TransformerAnomalyDetector(model_dir="ml_transformer/output")
    
    # Print raw flight data check
    print("\n--- Raw Flight Data (First 5 Points) ---")
    for p in flight.points[:5]:
        print(f"Time: {p.timestamp}, Lat: {p.lat}, Lon: {p.lon}, Alt: {p.alt}, Track: {p.track}")
        
    # Print resampled data check
    resampler = TrajectoryResampler(num_points=50)
    df = resampler.process(flight)
    print("\n--- Resampled Data (First 5 Points) ---")
    print(df.head())

    analyze_deep_ae(flight, deep_detector)
    analyze_transformer(flight, trans_detector)

if __name__ == "__main__":
    main()

