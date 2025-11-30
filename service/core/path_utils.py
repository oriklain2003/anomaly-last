from __future__ import annotations

from typing import List, Optional, Sequence, Dict

import numpy as np

from core.geodesy import haversine_nm
from core.models import TrackPoint


def _sorted_points(points: Sequence[TrackPoint]) -> List[TrackPoint]:
    """Ensure points are ordered by timestamp."""
    return sorted(points, key=lambda p: p.timestamp)


def resample_track_points(
    points: Sequence[TrackPoint],
    num_samples: int = 120,
) -> Optional[np.ndarray]:
    """
    Interpolate a flight path to a fixed-length sequence of (lat, lon, alt).

    Args:
        points: Original ADS-B samples.
        num_samples: Desired number of waypoints in the normalized path.

    Returns:
        np.ndarray of shape (num_samples, 3) ordered as [lat, lon, alt],
        or None if resampling is not possible (e.g. <2 unique timestamps).
    """
    if len(points) < 2:
        return None

    ordered = _sorted_points(points)
    timestamps = np.array([p.timestamp for p in ordered], dtype=float)
    lats = np.array([p.lat for p in ordered], dtype=float)
    lons = np.array([p.lon for p in ordered], dtype=float)
    alts = np.array([p.alt or 0.0 for p in ordered], dtype=float)

    # Remove duplicate timestamps (cannot interpolate otherwise)
    uniq_ts, uniq_idx = np.unique(timestamps, return_index=True)
    if uniq_ts.size < 2:
        return None

    uniq_lats = lats[uniq_idx]
    uniq_lons = lons[uniq_idx]
    uniq_alts = alts[uniq_idx]

    # Compute cumulative distance in NM
    dists = [0.0]
    for i in range(1, len(uniq_lats)):
        d = haversine_nm(uniq_lats[i-1], uniq_lons[i-1], uniq_lats[i], uniq_lons[i])
        dists.append(d)
    
    cum_dist = np.cumsum(dists)
    total_dist = cum_dist[-1]

    if total_dist == 0.0:
        return None

    target_dist = np.linspace(0.0, total_dist, num_samples)

    interp_lat = np.interp(target_dist, cum_dist, uniq_lats)
    interp_lon = np.interp(target_dist, cum_dist, uniq_lons)
    interp_alt = np.interp(target_dist, cum_dist, uniq_alts)

    return np.stack([interp_lat, interp_lon, interp_alt], axis=1)


def flatten_resampled_path(path: np.ndarray) -> np.ndarray:
    """
    Flatten a resampled path to a 1D vector suitable for clustering.
    """
    return path.reshape(-1)


def mean_path_distance_nm(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """
    Compute the mean great-circle distance between two normalized paths.

    Args:
        path_a: np.ndarray [N, 3]
        path_b: np.ndarray [N, 3]

    Returns:
        Average nautical-mile deviation between corresponding waypoints.
    """
    if path_a.shape != path_b.shape:
        raise ValueError("Paths must have identical shapes for distance comparison")

    total = 0.0
    for (lat_a, lon_a, _), (lat_b, lon_b, _) in zip(path_a, path_b):
        total += haversine_nm(lat_a, lon_a, lat_b, lon_b)
    return total / path_a.shape[0]


def filter_noisy_points(points: Sequence[TrackPoint]) -> List[TrackPoint]:
    """
    Filter out noisy points based on speed, heading jumps, and altitude fluctuations.
    """
    cleaned = []
    ordered = _sorted_points(points)
    
    for i, p in enumerate(ordered):
        # 1. Speed Filter
        if p.gspeed is not None and p.gspeed < 60:
            continue
            
        # 2. Gap Filter (check vs previous kept point)
        if cleaned:
            prev = cleaned[-1]
            dt = p.timestamp - prev.timestamp
            if dt > 20:
                # Split or drop? For now, if gap is huge, we might just treat as separate logic,
                # but simplest for path learning is to just drop or start new. 
                # User says "Remove segments where... dt > 20 sec gap". 
                # We'll assume we keep the point but we might flag it. 
                # Actually if dt > 20, it might be a missing data segment.
                # We'll keep it to avoid breaking continuity unless it's huge.
                continue

            # 3. Heading Error (Jump)
            if p.track is not None and prev.track is not None:
                diff = abs(p.track - prev.track)
                if diff > 180:
                    diff = 360 - diff
                if diff > 110:
                    continue

            # 4. Altitude Fluctuation
            if p.alt is not None and prev.alt is not None and dt > 0 and dt < 60:
                alt_diff = abs(p.alt - prev.alt)
                if alt_diff > 8000:
                    continue

        cleaned.append(p)
        
    return cleaned


def segment_flight(points: Sequence[TrackPoint]) -> Dict[str, List[TrackPoint]]:
    """
    Segment flight into Climb (Takeoff -> FL100), Cruise, Descent (FL100 -> Landing).
    """
    ordered = _sorted_points(points)
    if not ordered:
        return {"climb": [], "cruise": [], "descent": []}
        
    # Find split indices
    climb_end_idx = 0
    descent_start_idx = len(ordered)
    
    # Scan for climb end (first time > 10000)
    for i, p in enumerate(ordered):
        if p.alt and p.alt >= 10000:
            climb_end_idx = i
            break
    else:
        # Never reached 10000
        climb_end_idx = len(ordered) // 2

    # Scan for descent start (last time > 10000)
    for i in range(len(ordered) - 1, -1, -1):
        p = ordered[i]
        if p.alt and p.alt >= 10000:
            descent_start_idx = i + 1 # Include this point in cruise
            break
    else:
         descent_start_idx = len(ordered) // 2
         
    if climb_end_idx > descent_start_idx:
        # Short flight, just split middle
        mid = len(ordered) // 2
        climb_end_idx = mid
        descent_start_idx = mid

    return {
        "climb": ordered[:climb_end_idx],
        "cruise": ordered[climb_end_idx:descent_start_idx],
        "descent": ordered[descent_start_idx:]
    }
