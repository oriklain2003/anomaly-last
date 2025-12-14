"""
Shared utilities for the learning module.

Provides:
- Bounding box filtering
- Clustering helpers (HDBSCAN, DBSCAN)
- DBA centroid computation
- Turn detection
- Resampling utilities
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import TrackPoint
from core.geodesy import haversine_nm, initial_bearing_deg
from core.path_utils import resample_track_points

logger = logging.getLogger(__name__)


# ============================================================================
# Bounding Box
# ============================================================================

@dataclass
class BoundingBox:
    """Geographic bounding box."""
    north: float
    south: float
    east: float
    west: float


def is_in_bbox(lat: float, lon: float, bbox: BoundingBox) -> bool:
    """Check if a point is within the bounding box."""
    return (bbox.south <= lat <= bbox.north) and (bbox.west <= lon <= bbox.east)


# ============================================================================
# Trajectory Distance
# ============================================================================

def trajectory_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute mean haversine distance between two resampled paths.
    
    Args:
        a: Array of shape (N, 3) with [lat, lon, alt]
        b: Array of shape (N, 3) with [lat, lon, alt]
        
    Returns:
        Mean distance in nautical miles
    """
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    
    total_dist = 0.0
    for i in range(length):
        la, lo, _ = a[i]
        lb, lo2, _ = b[i]
        total_dist += haversine_nm(la, lo, lb, lo2)
    
    return total_dist / length


def compute_distance_matrix(paths: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise distance matrix for trajectories.
    
    Args:
        paths: List of resampled trajectory arrays
        
    Returns:
        Symmetric distance matrix of shape (N, N)
    """
    n = len(paths)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            d = trajectory_distance(paths[i], paths[j])
            dist_matrix[i, j] = dist_matrix[j, i] = d
    
    return dist_matrix


# ============================================================================
# Clustering
# ============================================================================

def cluster_trajectories_hdbscan(
    paths: List[np.ndarray],
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> np.ndarray:
    """
    Cluster trajectories using HDBSCAN with precomputed distance matrix.
    
    WARNING: This is slow for large datasets. Use cluster_trajectories_dbscan instead.
    
    Args:
        paths: List of resampled trajectory arrays (N, 3)
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core point
        
    Returns:
        Array of cluster labels (-1 for noise)
    """
    if len(paths) < min_cluster_size:
        return np.full(len(paths), -1)
    
    dist_matrix = compute_distance_matrix(paths)
    
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.0
    )
    
    labels = clusterer.fit_predict(dist_matrix.astype(np.float64))
    return labels


def cluster_trajectories_dbscan(
    paths: List[np.ndarray],
    eps_nm: float = 5.0,
    min_samples: int = 3,
) -> np.ndarray:
    """
    Cluster trajectories using DBSCAN with mean-point distance.
    
    This is MUCH faster than HDBSCAN for large datasets because it uses
    mean trajectory distance instead of computing full N×N distance matrix.
    
    Args:
        paths: List of resampled trajectory arrays (N, 3)
        eps_nm: Maximum mean distance (nm) between trajectories in same cluster
        min_samples: Minimum samples for core point
        
    Returns:
        Array of cluster labels (-1 for noise)
    """
    if len(paths) < min_samples:
        return np.full(len(paths), -1)
    
    # Compute mean point for each trajectory (centroid lat/lon)
    centroids = []
    for path in paths:
        mean_lat = np.mean(path[:, 0])
        mean_lon = np.mean(path[:, 1])
        centroids.append((mean_lat, mean_lon))
    
    # Use DBSCAN on centroids (much faster than full distance matrix)
    # Convert nm to approximate degrees
    eps_deg = eps_nm / 60.0
    
    X = np.array(centroids)
    
    clusterer = DBSCAN(
        eps=eps_deg,
        min_samples=min_samples,
        metric='euclidean'
    )
    
    labels = clusterer.fit_predict(X)
    return labels


def cluster_trajectories_dbscan_precomputed(
    paths: List[np.ndarray],
    eps_nm: float = 5.0,
    min_samples: int = 3,
) -> np.ndarray:
    """
    Cluster trajectories using DBSCAN with precomputed trajectory distance matrix.
    
    More accurate than centroid-based but still O(N²). Use for smaller datasets.
    
    Args:
        paths: List of resampled trajectory arrays (N, 3)
        eps_nm: Maximum mean trajectory distance (nm) in same cluster
        min_samples: Minimum samples for core point
        
    Returns:
        Array of cluster labels (-1 for noise)
    """
    if len(paths) < min_samples:
        return np.full(len(paths), -1)
    
    dist_matrix = compute_distance_matrix(paths)
    
    clusterer = DBSCAN(
        eps=eps_nm,
        min_samples=min_samples,
        metric='precomputed'
    )
    
    labels = clusterer.fit_predict(dist_matrix)
    return labels


def cluster_points_dbscan(
    points: List[Tuple[float, float]],
    eps_nm: float = 2.0,
    min_samples: int = 3
) -> np.ndarray:
    """
    Cluster geographic points using DBSCAN.
    
    Args:
        points: List of (lat, lon) tuples
        eps_nm: Epsilon in nautical miles (converted to degrees approximately)
        min_samples: Minimum samples for core point
        
    Returns:
        Array of cluster labels (-1 for noise)
    """
    if len(points) < min_samples:
        return np.full(len(points), -1)
    
    # Convert nm to approximate degrees (1 degree ~ 60nm at equator)
    eps_deg = eps_nm / 60.0
    
    X = np.array(points)
    
    clusterer = DBSCAN(
        eps=eps_deg,
        min_samples=min_samples,
        metric='euclidean'  # Approximate for small regions
    )
    
    labels = clusterer.fit_predict(X)
    return labels


# ============================================================================
# DTW Barycenter Averaging (DBA)
# ============================================================================

def _align_dtw(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """
    Align sequence s1 to s2 using Dynamic Time Warping.
    
    Args:
        s1: Source sequence (N, 3)
        s2: Target sequence (M, 3)
        
    Returns:
        Aligned sequence with same shape as s2
    """
    n = len(s1)
    m = len(s2)
    
    # DTW cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = haversine_nm(s1[i-1][0], s1[i-1][1], s2[j-1][0], s2[j-1][1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    # Backtrack to get alignment
    i, j = n, m
    matches = {}
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            matches.setdefault(j-1, []).append(s1[i-1])
            
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
            matches.setdefault(0, []).append(s1[i-1])
            i -= 1
        elif j > 0:
            matches.setdefault(j-1, []).append(s1[0])
            j -= 1
    
    # Construct aligned path
    aligned = np.zeros_like(s2)
    for k in range(m):
        pts = matches.get(k, [])
        if pts:
            aligned[k] = np.mean(np.stack(pts), axis=0)
        else:
            aligned[k] = s2[k]
    
    return aligned


def compute_dba_centroid(paths: List[np.ndarray], iterations: int = 3) -> np.ndarray:
    """
    Compute DTW Barycenter Averaging centroid for trajectories.
    
    Args:
        paths: List of trajectory arrays (each N, 3)
        iterations: Number of refinement iterations
        
    Returns:
        Centroid trajectory array
    """
    if not paths:
        raise ValueError("Cannot compute centroid of empty paths")
    
    if len(paths) == 1:
        return paths[0].copy()
    
    # Initialize with mean
    centroid = np.mean(paths, axis=0)
    
    # Iterative refinement
    for _ in range(iterations):
        new_centroid = np.zeros_like(centroid)
        weight = np.zeros((centroid.shape[0], 1))
        
        for p in paths:
            aligned = _align_dtw(p, centroid)
            new_centroid += aligned
            weight += 1
        
        centroid = new_centroid / weight
    
    return centroid


# ============================================================================
# Turn Detection
# ============================================================================

@dataclass
class TurnEvent:
    """Represents a detected turn event."""
    flight_id: str
    start_ts: int
    end_ts: int
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    mid_lat: float
    mid_lon: float
    mid_alt: float
    cumulative_deg: float
    avg_speed: float
    direction: int  # +1 for right, -1 for left


def _signed_heading_delta(h1: float, h0: float) -> float:
    """Compute signed heading change in [-180, 180]."""
    return ((h1 - h0 + 540) % 360) - 180


def detect_turns(
    points: List[TrackPoint],
    min_deg: float = 180.0,
    max_deg: float = 300.0,
    min_duration_s: int = 30,
    max_duration_s: int = 600,
    min_speed_kts: float = 50.0,  # Reduced from 80 - many valid turns at 60-70 kts
    min_alt_ft: float = 1000.0
) -> List[TurnEvent]:
    """
    Detect turns within the specified angle range.
    
    Args:
        points: Sorted list of track points
        min_deg: Minimum cumulative turn angle
        max_deg: Maximum cumulative turn angle
        min_duration_s: Minimum turn duration in seconds
        max_duration_s: Maximum turn duration in seconds
        min_speed_kts: Minimum speed to consider
        min_alt_ft: Minimum altitude to consider
        
    Returns:
        List of detected turn events
    """
    if len(points) < 5:
        return []
    
    turns = []
    sorted_pts = sorted(points, key=lambda p: p.timestamp)
    
    # Scan for turns
    for start_idx in range(len(sorted_pts)):
        start_p = sorted_pts[start_idx]
        
        # Filter by speed and altitude
        if (start_p.gspeed or 0) < min_speed_kts:
            continue
        if (start_p.alt or 0) < min_alt_ft:
            continue
        if start_p.track is None:
            continue
        
        cumulative = 0.0
        direction = None
        speeds = [start_p.gspeed or 0]
        lats = [start_p.lat]
        lons = [start_p.lon]
        alts = [start_p.alt or 0]
        
        prev_heading = start_p.track
        
        for end_idx in range(start_idx + 1, len(sorted_pts)):
            end_p = sorted_pts[end_idx]
            
            duration = end_p.timestamp - start_p.timestamp
            if duration > max_duration_s:
                break
            
            if end_p.track is None:
                continue
            if (end_p.gspeed or 0) < min_speed_kts:
                continue
            if (end_p.alt or 0) < min_alt_ft:
                continue
            
            # Compute heading change
            dh = _signed_heading_delta(end_p.track, prev_heading)
            
            # Establish or validate direction
            if dh != 0:
                sign = 1 if dh > 0 else -1
                if direction is None:
                    direction = sign
                elif sign != direction and abs(dh) > 15:
                    # Direction reversal - stop tracking
                    break
            
            cumulative += abs(dh)
            speeds.append(end_p.gspeed or 0)
            lats.append(end_p.lat)
            lons.append(end_p.lon)
            alts.append(end_p.alt or 0)
            prev_heading = end_p.track
            
            # Check if we found a valid turn
            if duration >= min_duration_s and min_deg <= cumulative <= max_deg:
                mid_idx = len(lats) // 2
                turns.append(TurnEvent(
                    flight_id=start_p.flight_id,
                    start_ts=start_p.timestamp,
                    end_ts=end_p.timestamp,
                    start_lat=start_p.lat,
                    start_lon=start_p.lon,
                    end_lat=end_p.lat,
                    end_lon=end_p.lon,
                    mid_lat=lats[mid_idx],
                    mid_lon=lons[mid_idx],
                    mid_alt=alts[mid_idx],
                    cumulative_deg=cumulative,
                    avg_speed=sum(speeds) / len(speeds),
                    direction=direction or 0
                ))
                break  # Found a turn, move to next start point
            
            # If we've exceeded max_deg, stop
            if cumulative > max_deg:
                break
    
    return turns


# ============================================================================
# Resampling
# ============================================================================

def resample_flight(
    points: List[TrackPoint],
    num_samples: int = 80
) -> Optional[np.ndarray]:
    """
    Resample flight track to fixed number of points.
    
    Args:
        points: List of track points
        num_samples: Number of output samples
        
    Returns:
        Array of shape (num_samples, 3) with [lat, lon, alt], or None if failed
    """
    return resample_track_points(points, num_samples=num_samples)


def extract_segment(
    points: List[TrackPoint],
    from_start: bool = True,
    distance_nm: float = 30.0
) -> List[TrackPoint]:
    """
    Extract a segment from the start or end of a flight.
    
    Args:
        points: Sorted list of track points
        from_start: If True, extract from start; if False, from end
        distance_nm: Length of segment in nautical miles
        
    Returns:
        List of points in the segment
    """
    if len(points) < 2:
        return points
    
    sorted_pts = sorted(points, key=lambda p: p.timestamp)
    
    if not from_start:
        sorted_pts = sorted_pts[::-1]  # Reverse for end extraction
    
    segment = [sorted_pts[0]]
    total_dist = 0.0
    
    for i in range(1, len(sorted_pts)):
        prev = sorted_pts[i - 1]
        curr = sorted_pts[i]
        
        dist = haversine_nm(prev.lat, prev.lon, curr.lat, curr.lon)
        total_dist += dist
        segment.append(curr)
        
        if total_dist >= distance_nm:
            break
    
    if not from_start:
        segment = segment[::-1]  # Reverse back
    
    return segment


def compute_cluster_width(
    paths: List[np.ndarray],
    centroid: np.ndarray
) -> float:
    """
    Compute the width of a cluster based on spread from centroid.
    
    Args:
        paths: List of member trajectories
        centroid: Cluster centroid
        
    Returns:
        Width in nautical miles (2 * std deviation)
    """
    if not paths:
        return 2.0
    
    distances = []
    for path in paths:
        for i in range(min(len(path), len(centroid))):
            dist = haversine_nm(path[i][0], path[i][1], centroid[i][0], centroid[i][1])
            distances.append(dist)
    
    if not distances:
        return 2.0
    
    std = np.std(distances)
    return max(2.0, float(std * 2))  # At least 2nm width

