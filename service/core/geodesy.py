from __future__ import annotations

import math
from typing import Tuple

EARTH_RADIUS_KM = 6371.0
NM_PER_KM = 0.539957


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c * NM_PER_KM


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.cos(math.radians(lon2 - lon1))
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def cross_track_distance_nm(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    point: Tuple[float, float],
) -> float:
    """
    Compute cross-track distance from the great-circle path connecting origin and destination.
    Uses a spherical Earth approximation, accurate enough for route-deviation heuristics.
    """
    lat1, lon1 = map(math.radians, origin)
    lat2, lon2 = map(math.radians, destination)
    lat3, lon3 = map(math.radians, point)

    dist13 = angular_distance(lat1, lon1, lat3, lon3)
    if dist13 == 0.0:
        return 0.0

    bearing13 = bearing_rad(lat1, lon1, lat3, lon3)
    bearing12 = bearing_rad(lat1, lon1, lat2, lon2)

    sin_xt = math.sin(dist13) * math.sin(bearing13 - bearing12)
    xt_distance_km = math.asin(max(-1.0, min(1.0, sin_xt))) * EARTH_RADIUS_KM
    return abs(xt_distance_km * NM_PER_KM)


def angular_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return math.atan2(y, x)

