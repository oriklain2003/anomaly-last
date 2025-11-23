import torch
import json
import numpy as np
from pathlib import Path
from .preprocessing import TrajectoryResampler
from .clustering import TrajectoryClusterer
from .model import TrajectoryCNN

class DeepCNNDetector:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.clusterer = TrajectoryClusterer.load(model_dir / "clusters.joblib")
        
        with open(model_dir / "thresholds.json", "r") as f:
            self.thresholds = json.load(f)
            
        self.models = {}
        self.norms = {}
        
        for cid in self.thresholds.keys():
            # Load Norm
            with open(model_dir / f"norm_{cid}.json", "r") as f:
                self.norms[cid] = json.load(f)
                
            # Load Model
            model = TrajectoryCNN(num_features=4, seq_len=50).to(self.device)
            weight_path = model_dir / f"cnn_{cid}.pt"
            if weight_path.exists():
                model.load_state_dict(torch.load(weight_path, map_location=self.device))
                model.eval()
                self.models[cid] = model

    def predict(self, flight):
        resampler = TrajectoryResampler(num_points=50)
        df = resampler.process(flight)
        
        if df.empty:
            return {"error": "Invalid flight"}
            
        # 1. Cluster (using flat vector)
        vec_flat = resampler.flatten(df).reshape(1, -1)
        cluster_id = str(self.clusterer.predict(vec_flat)[0])
        
        if cluster_id not in self.models:
            return {"error": "Unknown Cluster Model"}
            
        # 2. Prepare Matrix for CNN
        mat = resampler.to_matrix(df) # [50, 4]
        
        # Normalize using Cluster Stats
        mean = np.array(self.norms[cluster_id]["mean"]) # [1, 1, 4]
        std = np.array(self.norms[cluster_id]["std"])
        
        mat_norm = (mat.reshape(1, 50, 4) - mean) / std
        
        # 3. Predict
        tensor_in = torch.FloatTensor(mat_norm).to(self.device)
        model = self.models[cluster_id]
        
        with torch.no_grad():
            loss = model.get_reconstruction_error(tensor_in).item()
            
        thresh = self.thresholds[cluster_id]
        is_anom = loss > thresh
        
        return {
            "cluster_id": int(cluster_id),
            "score": loss,
            "threshold": thresh,
            "severity": loss / thresh,
            "is_anomaly": is_anom,
            "status": "ANOMALY" if is_anom else "NORMAL"
        }

