from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler # Kept for compatibility if needed, but likely unused
from functools import lru_cache

from core.db import DbConfig, FlightRepository
from core.path_utils import resample_track_points
from core.geodesy import haversine_nm

def trajectory_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute mean haversine distance between two resampled paths.
    a, b shape: (N, 3) -> lat, lon, alt
    """
    d = 0.0
    # Determine length to iterate (should be same)
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    
    for i in range(length):
        la, lo, _ = a[i]
        lb, lo2, _ = b[i]
        d += haversine_nm(la, lo, lb, lo2)
    
    return d / length

def align_dtw(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """
    Align sequence s1 to s2 using DTW and return the warped s1.
    s1, s2: (N, 3)
    """
    n = len(s1)
    m = len(s2)
    
    # DTW Cost Matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    # We'll store pointers to reconstruct path: 0=match, 1=insert(i-1), 2=delete(j-1)
    # But for simple DBA we just need the path indices.
    # Optimization: We can just compute cost and backtrack.
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Cost: 3D distance or just Lat/Lon?
            # User logic focused on haversine for clustering. 
            # For centroid, we should probably include altitude or treat it separately.
            # Let's use simple Euclidean on lat/lon/alt? No, different scales.
            # Let's use Haversine for lat/lon and ignore alt for alignment cost?
            # Or normalize alt.
            # For simplicity and consistency with trajectory_distance, let's use Haversine.
            cost = haversine_nm(s1[i-1][0], s1[i-1][1], s2[j-1][0], s2[j-1][1])
            
            # Standard DTW recursion
            dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                   dtw[i, j-1],    # deletion
                                   dtw[i-1, j-1])  # match
                                   
    # Backtrack
    i, j = n, m
    matches: Dict[int, List[np.ndarray]] = {}
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            current_matches = matches.setdefault(j-1, [])
            current_matches.append(s1[i-1])
            
            # Find direction
            cost_match = dtw[i-1, j-1]
            cost_ins = dtw[i-1, j]
            cost_del = dtw[i, j-1]
            
            minimum = min(cost_match, cost_ins, cost_del)
            
            if minimum == cost_match:
                i -= 1
                j -= 1
            elif minimum == cost_ins:
                i -= 1
            else:
                j -= 1
        elif i > 0:
            # Only i > 0, j = 0. Map remaining i to j=0?
            matches.setdefault(0, []).append(s1[i-1])
            i -= 1
        elif j > 0:
            # Only j > 0, i = 0. Map i=0 to remaining j?
            # Or just skip?
            # Usually this means s2 has extra points at start.
            # We can duplicate s1[0] for them.
            matches.setdefault(j-1, []).append(s1[0])
            j -= 1

    # Construct aligned path
    aligned = np.zeros_like(s2)
    for k in range(m):
        pts = matches.get(k, [])
        if pts:
            aligned[k] = np.mean(np.stack(pts), axis=0)
        else:
            # If no points mapped to k (unlikely with standard DTW unless huge gaps), use s2[k] or interpolate
            aligned[k] = s2[k]
            
    return aligned

def compute_centroid(paths: List[np.ndarray]) -> np.ndarray:
    """
    Compute DTW Barycenter Averaging (DBA) for trajectory centroid.
    """
    if not paths:
        raise ValueError("Cannot compute centroid of empty paths")
        
    # Start with simple averaging for initialization
    centroid = np.mean(paths, axis=0)

    # Iterate 3–5 times for refinement
    for _ in range(3):
        new_centroid = np.zeros_like(centroid)
        weight = np.zeros((centroid.shape[0], 1))

        for p in paths:
            # Align p to centroid using DTW
            aligned = align_dtw(p, centroid)
            new_centroid += aligned
            weight += 1

        centroid = new_centroid / weight

    return centroid

def _collect_tracks(
    repository: FlightRepository,
    *,
    limit: Optional[int],
    num_samples: int,
) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
    """
    Fetch flights, resample them, and compute distance matrix.
    """
    flight_ids: List[str] = []
    resampled_paths: Dict[str, np.ndarray] = {}

    print("[build-paths] Collecting and resampling tracks...")
    # Using a larger batch size or just iterating
    count = 0
    for track in repository.iter_flights(limit=limit, min_points=5):
        resampled = resample_track_points(track.points, num_samples=num_samples)
        if resampled is None:
            continue

        flight_ids.append(track.flight_id)
        resampled_paths[track.flight_id] = resampled
        count += 1
        
        if count % 100 == 0:
            print(f"[build-paths] Processed {count} flights...", end='\r')
            
    print(f"\n[build-paths] Collected {len(flight_ids)} valid tracks.")
    
    if not flight_ids:
        raise RuntimeError("No flights could be resampled. Check DB content or num_samples.")

    # Compute distance matrix
    print(f"[build-paths] Computing distance matrix for {len(flight_ids)} paths...")
    path_list = [resampled_paths[fid] for fid in flight_ids]
    N = len(path_list)
    dist_matrix = np.zeros((N, N))
    
    # Build upper triangle
    # This can be slow for large N.
    # We could use joblib for parallelization if needed, but start simple.
    processed_pairs = 0
    total_pairs = N * (N - 1) // 2
    
    for i in range(N):
        for j in range(i + 1, N):
            d = trajectory_distance(path_list[i], path_list[j])
            dist_matrix[i, j] = dist_matrix[j, i] = d
            
            processed_pairs += 1
            if processed_pairs % 50000 == 0:
                 print(f"[build-paths] Matrix progress: {processed_pairs}/{total_pairs}", end='\r')
                 
    print("\n[build-paths] Distance matrix computed.")
    return flight_ids, dist_matrix, resampled_paths

def _cluster_paths(
    dist_matrix: np.ndarray,
    min_samples: int,
    min_cluster_size: int,
) -> Tuple[object, np.ndarray]:
    """
    HDBSCAN using precomputed trajectory distance matrix.
    """
    model = hdbscan.HDBSCAN(
        metric="precomputed",
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=0.0
    )
    labels = model.fit_predict(dist_matrix.astype(np.float64))
    return model, labels

def _build_flow_payload(
    flight_ids: Sequence[str],
    labels: np.ndarray,
    resampled_paths: Dict[str, np.ndarray],
) -> Tuple[List[Dict[str, object]], List[str]]:
    clusters: Dict[int, List[str]] = {}
    for flight_id, label in zip(flight_ids, labels):
        clusters.setdefault(int(label), []).append(flight_id)

    flows: List[Dict[str, object]] = []
    noise_ids = clusters.get(-1, [])

    print(f"[build-paths] Building flow centroids for {len(clusters) - (1 if -1 in clusters else 0)} clusters...")
    
    for label, members in clusters.items():
        if label == -1:
            continue

        member_paths = [resampled_paths[flight_id] for flight_id in members]
        
        # Use DBA centroid
        centroid = compute_centroid(member_paths)

        flows.append(
            {
                "flow_id": f"flow_{label}",
                "cluster_label": int(label),
                "num_flights": len(members),
                "flight_ids": members,
                "representative_flight_id": members[0],
                "centroid_path": [
                    {"lat": float(lat), "lon": float(lon), "alt": float(alt)}
                    for lat, lon, alt in centroid.tolist()
                ],
            }
        )

    flows.sort(key=lambda item: item["num_flights"], reverse=True)
    return flows, noise_ids

def _run_clustering_layer(
    flight_ids: List[str],
    dist_matrix: np.ndarray,
    resampled_paths: Dict[str, np.ndarray],
    min_samples: int,
    min_cluster_size: int,
    layer_name: str
) -> Dict[str, object]:
    """
    Run clustering for a single configuration.
    """
    print(f"[build-paths] Running {layer_name} clustering (min_samples={min_samples}, min_cluster_size={min_cluster_size})...")
    
    _, labels = _cluster_paths(dist_matrix, min_samples=min_samples, min_cluster_size=min_cluster_size)
    flows, noise_ids = _build_flow_payload(flight_ids, labels, resampled_paths)
    
    print(f"[build-paths] {layer_name}: Found {len(flows)} flows covering {len(flight_ids) - len(noise_ids)} flights.")
    
    return {
        "layer_name": layer_name,
        "hyperparameters": {"min_samples": min_samples, "min_cluster_size": min_cluster_size},
        "flow_count": len(flows),
        "noise_flights": noise_ids,
        "flows": flows,
    }

def run_builder(
    *,
    db_path: Path,
    table: str,
    output: Path,
    limit: Optional[int],
    num_samples: int,
    min_samples_strict: int,
    min_samples_loose: int,
    # eps params are ignored/deprecated but kept in signature if needed, 
    # but we'll rely on min_samples/min_cluster_size
    eps_strict: float = 0.0, 
    eps_loose: float = 0.0,
) -> Dict[str, object]:
    
    repository = FlightRepository(DbConfig(path=db_path, table=table))
    flight_ids, dist_matrix, resampled_paths = _collect_tracks(
        repository, limit=limit, num_samples=num_samples
    )

    # Run Strict Layer (Conservative)
    # Using min_samples_strict for both parameters
    strict_results = _run_clustering_layer(
        flight_ids, dist_matrix, resampled_paths, 
        min_samples=min_samples_strict,
        min_cluster_size=min_samples_strict,
        layer_name="strict"
    )

    # Run Loose Layer (Relaxed)
    loose_results = _run_clustering_layer(
        flight_ids, dist_matrix, resampled_paths, 
        min_samples=min_samples_loose,
        min_cluster_size=min_samples_loose,
        layer_name="loose"
    )

    payload = {
        "db_path": str(db_path.resolve()),
        "table": table,
        "num_samples": num_samples,
        "total_resampled_flights": len(flight_ids),
        "layers": {
            "strict": strict_results,
            "loose": loose_results
        }
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[build-paths] Saved multi-layer path library to {output}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster historical flights into canonical paths (HDBSCAN) using two layers (Strict/Loose)."
        )
    )
    parser.add_argument("--db", type=Path, default=Path("last.db"), help="SQLite source (default: last.db)")
    parser.add_argument(
        "--table", type=str, default="flight_tracks", help="Table name containing ADS-B samples"
    )
    parser.add_argument("--output", type=Path, default=Path("rules/learned_paths.json"))
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on flights to process")
    parser.add_argument("--num-samples", type=int, default=80, help="Resampling length per flight")
    
    # Strict Layer Defaults (Conservative: needs more points to form cluster)
    parser.add_argument("--min-samples-strict", type=int, default=6, help="Strict layer min_samples")
    parser.add_argument("--eps-strict", type=float, default=0.0, help="Ignored (legacy)")

    # Loose Layer Defaults (Relaxed)
    parser.add_argument("--min-samples-loose", type=int, default=4, help="Loose layer min_samples")
    parser.add_argument("--eps-loose", type=float, default=0.0, help="Ignored (legacy)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_builder(
        db_path=args.db,
        table=args.table,
        output=args.output,
        limit=args.limit,
        num_samples=args.num_samples,
        min_samples_strict=args.min_samples_strict,
        min_samples_loose=args.min_samples_loose,
        eps_strict=args.eps_strict,
        eps_loose=args.eps_loose,
    )


if __name__ == "__main__":
    main()
