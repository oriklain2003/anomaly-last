import torch
import numpy as np
import joblib
import json
from pathlib import Path
from .model import TrajectoryTransformerAE
from ml_deep.preprocessing import TrajectoryResampler
from ml_deep.clustering import TrajectoryClusterer

class TransformerAnomalyDetector:
    def __init__(self, model_dir="ml_transformer/output"):
        self.model_dir = Path(model_dir)
        # Use the class method to load
        self.clusters = TrajectoryClusterer.load(self.model_dir / "clusters.joblib")
        
        with open(self.model_dir / "thresholds.json", "r") as f:
            self.thresholds = json.load(f)
            
        self.models = {}
        self.norms = {}
        self.resampler = TrajectoryResampler(num_points=50)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models for each cluster
        for cid in self.thresholds.keys():
            # Load Norm
            with open(self.model_dir / f"norm_{cid}.json", "r") as f:
                norm_data = json.load(f)
                self.norms[cid] = {
                    "mean": np.array(norm_data["mean"]),
                    "std": np.array(norm_data["std"])
                }
            
            # Load Model
            model = TrajectoryTransformerAE(input_dim=4, seq_len=50).to(self.device)
            model.load_state_dict(torch.load(self.model_dir / f"transformer_{cid}.pt", map_location=self.device))
            model.eval()
            self.models[cid] = model

    def predict(self, flight_track):
        """
        Returns anomaly score and is_anomaly boolean.
        """
        # 1. Resample
        df = self.resampler.process(flight_track)
        if df.empty:
            return 0.0, False
            
        # 2. Cluster
        vec_flat = self.resampler.flatten(df).reshape(1, -1)
        cluster_id = str(self.clusters.predict(vec_flat)[0])
        
        if cluster_id not in self.models:
            return 0.0, False # Unknown cluster or too small
            
        # 3. Prepare Input
        mat = self.resampler.to_matrix(df) # (50, 4)
        norm = self.norms[cluster_id]
        mat_norm = (mat - norm["mean"]) / norm["std"] # (1, 50, 4) (broadcasting works if shapes match)
        # Reshape mean/std are (1, 1, 4), mat is (50, 4). 
        # Need to ensure broadcast. (50,4) - (1,1,4) works if first dim is ignored or broadcasted? 
        # numpy broadcasting: (50, 4) and (1, 1, 4) -> (1, 50, 4). 
        # So we need mat to be (1, 50, 4) first.
        
        mat_input = mat.reshape(1, 50, 4)
        mat_norm = (mat_input - norm["mean"]) / norm["std"]
        
        tensor_in = torch.FloatTensor(mat_norm).to(self.device)
        
        # 4. Inference
        with torch.no_grad():
            error = self.models[cluster_id].get_reconstruction_error(tensor_in).item()
            
        threshold = self.thresholds[cluster_id]
        
        return error, error > threshold

