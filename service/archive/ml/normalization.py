from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class NormalizationStats:
    normalizers: Dict[str, float]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_cols: List[str],
        percentile: float = 99.0,
    ) -> "NormalizationStats":
        stats: Dict[str, float] = {}
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing from dataframe; cannot normalize.")
            series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                stats[col] = 1.0
                continue
            value = float(np.percentile(np.abs(series), percentile))
            stats[col] = value or 1.0
        return cls(stats)

    def normalize_df(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        columns = []
        for col in feature_cols:
            base = df[col].to_numpy(dtype=float)
            divisor = self.normalizers.get(col, 1.0)
            columns.append(base / divisor)
        return np.stack(columns, axis=1) if columns else np.empty((0, 0))

    def normalize_array(self, values: np.ndarray, feature_cols: List[str]) -> np.ndarray:
        arr = np.asarray(values, dtype=float).copy()
        for idx, col in enumerate(feature_cols):
            divisor = self.normalizers.get(col, 1.0)
            arr[:, idx] = arr[:, idx] / divisor
        return arr

    def save(self, path: Path | str) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.normalizers, handle, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "NormalizationStats":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

