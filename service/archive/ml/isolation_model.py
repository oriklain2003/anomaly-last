from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .normalization import NormalizationStats


@dataclass
class IsolationForestAnomalyModel:
    feature_cols: List[str]
    contamination: float = 0.01
    n_estimators: int = 300
    random_state: int = 42

    def __post_init__(self) -> None:
        self._model: Optional[IsolationForest] = None
        self._normalizer: Optional[NormalizationStats] = None
        self._threshold: Optional[float] = None  # Store threshold from training

    def fit(self, df_features: pd.DataFrame, normalizer: NormalizationStats) -> None:
        """Fit the Isolation Forest model."""
        self._normalizer = normalizer
        X = normalizer.normalize_df(df_features, self.feature_cols)
        
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X)
        
        # Calculate and store the threshold based on TRAINING data
        train_scores = -self._model.score_samples(X)
        self._threshold = float(np.percentile(train_scores, 100 * (1 - self.contamination)))

    def predict_scores(self, df_features: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        if self._model is None or self._normalizer is None:
            raise RuntimeError("Model must be trained before prediction.")
        X = self._normalizer.normalize_df(df_features, self.feature_cols)
        raw = self._model.score_samples(X)
        return -raw

    def predict_labels(self, df_features: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predict labels (1 for anomaly, 0 for normal)."""
        if self._threshold is None and threshold is None:
            raise RuntimeError("Model has not been fitted, so no threshold is available. Please call fit() first or provide an explicit threshold.")
            
        scores = self.predict_scores(df_features)
        limit = threshold if threshold is not None else self._threshold
        return (scores >= limit).astype(int)

    def predict_flights(self, df_features: pd.DataFrame, anomaly_threshold_pct: float = 0.05) -> Dict[str, bool]:
        """
        Predict anomaly status for entire flights.
        
        A flight is considered anomalous if more than `anomaly_threshold_pct` 
        of its points are flagged as anomalous.
        
        Args:
            df_features: DataFrame with features AND 'flight_id' column.
            anomaly_threshold_pct: Percentage of anomalous points required to flag the flight (default 5%).
            
        Returns:
            Dictionary {flight_id: is_anomalous}
        """
        if "flight_id" not in df_features.columns:
            raise ValueError("DataFrame must contain 'flight_id' column for flight-level prediction.")
            
        # Get point-level predictions
        labels = self.predict_labels(df_features)
        
        # Create a temporary DF for aggregation
        df_temp = df_features[["flight_id"]].copy()
        df_temp["is_anomaly"] = labels
        
        # Aggregate per flight
        flight_stats = df_temp.groupby("flight_id")["is_anomaly"].agg(["count", "sum"])
        flight_stats["anomaly_rate"] = flight_stats["sum"] / flight_stats["count"]
        
        # Determine flight status
        flight_status = (flight_stats["anomaly_rate"] > anomaly_threshold_pct).to_dict()
        return flight_status

    def save(self, model_path: Path | str, normalizer_path: Path | str) -> None:
        if self._model is None or self._normalizer is None or self._threshold is None:
            raise RuntimeError("Model must be trained before saving.")
        
        model_file = Path(model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "model": self._model,
            "threshold": self._threshold,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "feature_cols": self.feature_cols
        }
        joblib.dump(state, model_file)
        self._normalizer.save(normalizer_path)

    @classmethod
    def load(cls, model_path: Path | str, normalizer_path: Path | str, feature_cols: Optional[List[str]] = None) -> "IsolationForestAnomalyModel":
        """Load a trained model and normalizer."""
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
            
        state = joblib.load(model_file)
        normalizer = NormalizationStats.load(normalizer_path)
        
        if isinstance(state, IsolationForest):
            # Legacy fallback
            loaded_sk_model = state
            threshold = None
            contamination = loaded_sk_model.contamination
            n_estimators = loaded_sk_model.n_estimators
            random_state = loaded_sk_model.random_state
            f_cols = feature_cols or []
        else:
            loaded_sk_model = state["model"]
            threshold = state["threshold"]
            contamination = state["contamination"]
            n_estimators = state["n_estimators"]
            random_state = state["random_state"]
            f_cols = state["feature_cols"]
        
        instance = cls(
            feature_cols=f_cols,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        instance._model = loaded_sk_model
        instance._normalizer = normalizer
        instance._threshold = threshold
        
        return instance
