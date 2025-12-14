from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.geodesy import haversine_nm
from core.models import TrackPoint


def _sorted_points(points: Sequence[TrackPoint]) -> List[TrackPoint]:
    """Ensure points are ordered by timestamp."""
    return sorted(points, key=lambda p: p.timestamp)


def resample_track_points(
    points: Sequence[TrackPoint],
    num_samples: int = 80,
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


def point_to_polyline_distance_nm(
    point: Tuple[float, float],
    polyline: Sequence[Tuple[float, float]],
) -> dict:
    """
    Compute the minimum distance from a point to a polyline and its normalized position.

    Returns:
        {"distance_nm": float, "position": float}
        distance_nm: closest lateral distance to any segment in NM
        position: 0-1 fraction along the path where the closest point lies
    """
    if len(polyline) < 2:
        raise ValueError("Polyline must contain at least two points")

    # Precompute cumulative great-circle lengths for position normalization
    segment_lengths = []
    for a, b in zip(polyline[:-1], polyline[1:]):
        segment_lengths.append(haversine_nm(a[0], a[1], b[0], b[1]))

    total_length = sum(segment_lengths) or 1.0  # prevent division by zero

    # Use a simple equirectangular projection for local distances
    def _to_xy(lat: float, lon: float, ref_lat: float) -> Tuple[float, float]:
        cos_lat = np.cos(np.radians(ref_lat))
        return lat * 60.0, lon * 60.0 * cos_lat  # degrees -> NM approximation

    ref_lat = float(polyline[0][0])
    px, py = _to_xy(point[0], point[1], ref_lat)

    min_dist = float("inf")
    best_pos = 0.0
    cum_length = 0.0

    for idx, (a, b) in enumerate(zip(polyline[:-1], polyline[1:])):
        seg_len = segment_lengths[idx]
        ax, ay = _to_xy(a[0], a[1], ref_lat)
        bx, by = _to_xy(b[0], b[1], ref_lat)

        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        seg_norm_sq = vx * vx + vy * vy

        if seg_norm_sq == 0.0:
            t = 0.0
        else:
            t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_norm_sq))

        proj_x = ax + t * vx
        proj_y = ay + t * vy
        dist_nm = float(np.hypot(px - proj_x, py - proj_y))

        if dist_nm < min_dist:
            min_dist = dist_nm
            best_pos = (cum_length + t * seg_len) / total_length

        cum_length += seg_len

    return {"distance_nm": float(min_dist), "position": float(best_pos)}

