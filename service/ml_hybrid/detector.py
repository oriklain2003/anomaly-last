import torch
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union

from core.models import FlightTrack
from ml_hybrid.model import HybridAutoencoder

class HybridAnomalyDetector:
    def __init__(self, model_dir: Path):
        self.model_path = model_dir / "hybrid_model.pth"
        self.scaler_path = model_dir / "scaler.joblib"
        self.threshold_path = model_dir / "threshold.joblib"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.threshold = 0.05 # Default
        
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Hybrid model not found at {self.model_path}")
            
        # Load Scaler
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
            
        # Load Threshold
        if self.threshold_path.exists():
            self.threshold = joblib.load(self.threshold_path)
            
        # Load Model
        # We need to know input dim from saved args or assume 4 (lat, lon, alt, speed)
        self.model = HybridAutoencoder(input_dim=4, seq_len=50)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, flight: FlightTrack) -> torch.Tensor:
        # Extract features: Lat, Lon, Alt, GSpeed
        # Normalize using scaler
        # Pad/Truncate to seq_len=50
        
        points = flight.sorted_points()
        data = []
        for p in points:
            data.append([p.lat, p.lon, p.alt, p.gspeed or 0])
            
        arr = np.array(data)
        
        # Scale
        if self.scaler:
            arr = self.scaler.transform(arr)
            
        # Fixed size 50
        target_len = 50
        current_len = len(arr)
        
        if current_len > target_len:
            # Take middle or sample? Let's take middle for anomaly context or just first 50?
            # Usually we want the whole track. Let's interpolate.
            # For simplicity in this demo, we take the first 50 or pad.
            # Better: Resample.
            indices = np.linspace(0, current_len - 1, target_len).astype(int)
            arr = arr[indices]
        elif current_len < target_len:
            # Pad with last value
            padding = np.tile(arr[-1], (target_len - current_len, 1))
            arr = np.vstack([arr, padding])
            
        # Convert to Tensor [1, Seq, Feat]
        tensor = torch.FloatTensor(arr).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, flight: FlightTrack) -> Dict[str, Any]:
        if not self.model:
            return {"error": "Model not loaded"}
            
        try:
            x = self.preprocess(flight)
            
            with torch.no_grad():
                loss = self.model.get_reconstruction_error(x)
                score = loss.item()
                
            is_anomaly = score > self.threshold
            
            return {
                "score": score,
                "threshold": self.threshold,
                "is_anomaly": is_anomaly,
                "severity": score / self.threshold if self.threshold > 0 else 0
            }
        except Exception as e:
            return {"error": str(e)}
