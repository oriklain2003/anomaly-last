import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.features import FeatureExtractor
from core.db import DbConfig, FlightRepository
from core.models import FlightTrack

def collect_flights(repo, limit=100):
    flights = list(repo.iter_flights(limit=limit, min_points=10))
    return [f for f in flights if f.points]

def get_df(flights, extractor):
    rows = []
    for f in flights:
        rows.extend(extractor.extract_flight_features(f))
    return pd.DataFrame(rows)

# Load data
db_path = Path("last.db")
repo_normal = FlightRepository(DbConfig(path=db_path, table="flight_tracks"))
repo_anom = FlightRepository(DbConfig(path=db_path, table="anomalous_tracks"))

print("Loading sample flights...")
flights_normal = collect_flights(repo_normal, limit=200)
flights_anom = collect_flights(repo_anom, limit=200)

extractor = FeatureExtractor()
df_normal = get_df(flights_normal, extractor)
df_anom = get_df(flights_anom, extractor)

cols = ['cum_turn_300', 'cum_alt_60', 'avg_speed_300', 'turn_rate', 'climb_rate']

print("\n=== Feature Statistics (Normal vs Anomalous) ===")
for col in cols:
    if col not in df_normal.columns: continue
    
    norm_mean = df_normal[col].mean()
    anom_mean = df_anom[col].mean()
    norm_std = df_normal[col].std()
    anom_std = df_anom[col].std()
    
    print(f"\nFeature: {col}")
    print(f"  Normal Mean: {norm_mean:.2f} (std: {norm_std:.2f})")
    print(f"  Anom Mean:   {anom_mean:.2f} (std: {anom_std:.2f})")
    print(f"  Diff:        {abs(norm_mean - anom_mean):.2f}")
