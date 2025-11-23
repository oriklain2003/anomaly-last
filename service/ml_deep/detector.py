import torch
import json
import numpy as np
from pathlib import Path
from .preprocessing import TrajectoryResampler
from .clustering import TrajectoryClusterer
from .model import TrajectoryAutoencoder

class DeepAnomalyDetector:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Clustering Model
        self.clusterer = TrajectoryClusterer.load(model_dir / "clusters.joblib")
        
        # Load Thresholds
        with open(model_dir / "thresholds.json", "r") as f:
            self.thresholds = json.load(f)
            
        # Load Autoencoders (Lazy Load or Preload all)
        self.aes = {}
        for cid in self.thresholds.keys():
            # Initialize Model Structure
            # We need to know input dimension. 
            # We can get it from the scaler mean shape in the clusterer
            input_dim = self.clusterer.scaler.mean_.shape[0]
            
            model = TrajectoryAutoencoder(input_dim=input_dim).to(self.device)
            
            # Load Weights
            weight_path = model_dir / f"ae_cluster_{cid}.pt"
            if weight_path.exists():
                model.load_state_dict(torch.load(weight_path, map_location=self.device))
                model.eval()
                self.aes[cid] = model
            else:
                print(f"Warning: Model for cluster {cid} not found at {weight_path}")

    def predict(self, flight):
        # 1. Preprocess
        resampler = TrajectoryResampler(num_points=50)
        df = resampler.process(flight)
        
        if df.empty:
            return {"error": "Flight too short or invalid"}
            
        vec = resampler.flatten(df)
        vec_reshaped = vec.reshape(1, -1) # Shape (1, 200)
        
        # 2. Determine Flow (Cluster)
        cluster_id = self.clusterer.predict(vec_reshaped)[0]
        cluster_id_str = str(cluster_id)
        
        if cluster_id_str not in self.aes:
            return {"error": f"No model for cluster {cluster_id}"}
            
        # 3. Check Anomaly (Autoencoder)
        model = self.aes[cluster_id_str]
        
        # Normalize input (using same scaler as training!)
        vec_norm = self.clusterer.scaler.transform(vec_reshaped)
        tensor_in = torch.FloatTensor(vec_norm).to(self.device)
        
        with torch.no_grad():
            loss = model.get_reconstruction_error(tensor_in).item()
            
        threshold = self.thresholds[cluster_id_str]
        is_anomaly = loss > threshold
        
        # Calculate Severity (Score / Threshold)
        severity = loss / threshold
        
        return {
            "cluster_id": int(cluster_id),
            "score": loss,
            "threshold": threshold,
            "severity": severity,
            "is_anomaly": is_anomaly,
            "status": "ANOMALY" if is_anomaly else "NORMAL"
        }

