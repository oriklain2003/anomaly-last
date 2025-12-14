from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.config import load_rule_config
from core.geodesy import (
    cross_track_distance_nm,
    haversine_nm,
    initial_bearing_deg,
    is_point_in_polygon,
    create_corridor_polygon,
)
from core.path_utils import resample_track_points, point_to_polyline_distance_nm
from core.models import FlightTrack, RuleContext, RuleResult, TrackPoint

CONFIG = load_rule_config()
RULES = CONFIG.get("rules", {})
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _require_rule_config(rule_name: str) -> Dict[str, Any]:
    cfg = RULES.get(rule_name)
    if cfg is None:
        raise KeyError(f"Missing configuration section for '{rule_name}' in rule_config.json")
    return cfg


EMERGENCY_SQUAWKS = set(CONFIG.get("emergency_squawks", []))

ALTITUDE_CFG = _require_rule_config("altitude_change")
ALTITUDE_CHANGE_FT = float(ALTITUDE_CFG["delta_ft"])
ALTITUDE_WINDOW_SECONDS = int(ALTITUDE_CFG["window_seconds"])
ALTITUDE_MIN_CRUISE_FT = float(ALTITUDE_CFG["min_cruise_ft"])

TURN_CFG = _require_rule_config("abrupt_turn")
TURN_THRESHOLD_DEG = float(TURN_CFG["heading_change_deg"])
TURN_WINDOW_SECONDS = int(TURN_CFG["window_seconds"])
TURN_MIN_SPEED_KTS = float(TURN_CFG["min_speed_kts"])
TURN_ACC_DEG = float(TURN_CFG.get("accumulated_turn_deg", 270))
TURN_ACC_WINDOW = int(TURN_CFG.get("accumulation_window_seconds", 300))

PROXIMITY_CFG = _require_rule_config("proximity")
PROXIMITY_DISTANCE_NM = float(PROXIMITY_CFG["distance_nm"])
PROXIMITY_ALTITUDE_FT = float(PROXIMITY_CFG["altitude_ft"])
PROXIMITY_TIME_WINDOW = int(PROXIMITY_CFG["time_window_seconds"])
PROXIMITY_AIRPORT_EXCLUSION_NM = float(PROXIMITY_CFG.get("airport_exclusion_nm", 0))

ROUTE_DEVIATION_CFG = _require_rule_config("route_deviation")
ROUTE_DEVIATION_NM = float(ROUTE_DEVIATION_CFG["cross_track_nm"])

GO_AROUND_CFG = _require_rule_config("go_around")
GO_AROUND_RADIUS_NM = float(GO_AROUND_CFG["radius_nm"])
GO_AROUND_LOW_ALT_FT = float(GO_AROUND_CFG["min_low_alt_ft"])
GO_AROUND_RECOVERY_FT = float(GO_AROUND_CFG["recovery_ft"])

RETURN_CFG = _require_rule_config("return_to_field")
RETURN_TIME_LIMIT_SECONDS = int(RETURN_CFG["time_limit_seconds"])
RETURN_NEAR_AIRPORT_NM = float(RETURN_CFG["near_airport_nm"])
RETURN_TAKEOFF_ALT_FT = float(RETURN_CFG["takeoff_alt_ft"])
RETURN_LANDING_ALT_FT = float(RETURN_CFG["landing_alt_ft"])
RETURN_MIN_OUTBOUND_NM = float(RETURN_CFG.get("min_outbound_nm", 0))
RETURN_MIN_ELAPSED_SECONDS = int(RETURN_CFG.get("min_elapsed_seconds", 0))

DIVERSION_CFG = _require_rule_config("diversion")
DIVERSION_NEAR_AIRPORT_NM = float(DIVERSION_CFG["near_airport_nm"])

LOW_ALTITUDE_CFG = _require_rule_config("low_altitude")
LOW_ALTITUDE_THRESHOLD_FT = float(LOW_ALTITUDE_CFG["threshold_ft"])
LOW_ALTITUDE_AIRPORT_RADIUS_NM = float(LOW_ALTITUDE_CFG["airport_radius_nm"])

SIGNAL_CFG = _require_rule_config("signal_loss")
SIGNAL_GAP_SECONDS = int(SIGNAL_CFG["gap_seconds"])
SIGNAL_REPEAT_COUNT = int(SIGNAL_CFG["repeat_count"])

UNPLANNED_LANDING_CFG = _require_rule_config("unplanned_israel_landing")
UNPLANNED_LANDING_RADIUS_NM = float(UNPLANNED_LANDING_CFG["near_airport_nm"])

PATH_CFG = _require_rule_config("path_learning")
_path_candidate = Path(PATH_CFG["paths_file"])
PATH_FILE = (_path_candidate if _path_candidate.is_absolute() else (PROJECT_ROOT / _path_candidate)).resolve()
HEATMAP_FILE = Path(PATH_CFG.get("heatmap_file", "rules/flight_heatmap_v2.npy"))
PATH_NUM_SAMPLES = int(PATH_CFG.get("num_samples", 120))
PATH_PRIMARY_RADIUS_NM = float(PATH_CFG.get("primary_radius_nm", 8.0))
PATH_SECONDARY_RADIUS_NM = float(PATH_CFG.get("secondary_radius_nm", 15.0))
HEATMAP_CELL_DEG = float(PATH_CFG.get("heatmap_cell_deg", 0.05))
HEATMAP_THRESHOLD = int(PATH_CFG.get("heatmap_threshold", 5))
MIN_OFF_COURSE_POINTS = int(PATH_CFG.get("min_off_course_points", 15))
EMERGING_DISTANCE_NM = float(PATH_CFG.get("emerging_distance_nm", 12.0))
EMERGING_BUCKET_SIZE = int(PATH_CFG.get("emerging_bucket_size", 5))
EMERGING_SIMILARITY_DEG = int(PATH_CFG.get("emerging_similarity_deg", 30))
DEFAULT_PATH_WIDTH_NM = float(PATH_CFG.get("default_width_nm", 8.0))


_LEARNED_POLYGONS_CACHE = None
_PATH_LIBRARY_CACHE: Optional[Dict[str, Any]] = None
_HEATMAP_CACHE: Optional[Tuple[np.ndarray, Dict[str, Any]]] = None
_LEARNED_TURNS_CACHE: Optional[List[Dict[str, Any]]] = None
_LEARNED_SID_CACHE: Optional[List[Dict[str, Any]]] = None
_LEARNED_STAR_CACHE: Optional[List[Dict[str, Any]]] = None

# Learned behavior configuration (optional - may not exist)
LEARNED_BEHAVIOR_CFG = RULES.get("learned_behavior", {})
_lb_turns_file = Path(LEARNED_BEHAVIOR_CFG.get("turns_file", "rules/learned_turns.json"))
LEARNED_TURNS_FILE = (_lb_turns_file if _lb_turns_file.is_absolute() else (PROJECT_ROOT / _lb_turns_file)).resolve()
_lb_sid_file = Path(LEARNED_BEHAVIOR_CFG.get("sid_file", "rules/learned_sid.json"))
LEARNED_SID_FILE = (_lb_sid_file if _lb_sid_file.is_absolute() else (PROJECT_ROOT / _lb_sid_file)).resolve()
_lb_star_file = Path(LEARNED_BEHAVIOR_CFG.get("star_file", "rules/learned_star.json"))
LEARNED_STAR_FILE = (_lb_star_file if _lb_star_file.is_absolute() else (PROJECT_ROOT / _lb_star_file)).resolve()
TURN_ZONE_TOLERANCE_NM = float(LEARNED_BEHAVIOR_CFG.get("turn_zone_tolerance_nm", 3.0))
SID_STAR_TOLERANCE_NM = float(LEARNED_BEHAVIOR_CFG.get("sid_star_tolerance_nm", 2.5))
MODERATE_TURN_MIN_DEG = float(LEARNED_BEHAVIOR_CFG.get("turn_angle_min_deg", 45.0))
MODERATE_TURN_MAX_DEG = float(LEARNED_BEHAVIOR_CFG.get("turn_angle_max_deg", 300.0))


def _load_learned_turns(refresh: bool = False) -> List[Dict[str, Any]]:
    """Load learned turn zones from JSON file."""
    global _LEARNED_TURNS_CACHE
    if _LEARNED_TURNS_CACHE is not None and not refresh:
        return _LEARNED_TURNS_CACHE
    
    try:
        if LEARNED_TURNS_FILE.exists():
            with open(LEARNED_TURNS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _LEARNED_TURNS_CACHE = data.get("zones", [])
        else:
            _LEARNED_TURNS_CACHE = []
    except Exception:
        _LEARNED_TURNS_CACHE = []
    
    return _LEARNED_TURNS_CACHE


def _load_learned_sid(refresh: bool = False) -> List[Dict[str, Any]]:
    """Load learned SID procedures from JSON file."""
    global _LEARNED_SID_CACHE
    if _LEARNED_SID_CACHE is not None and not refresh:
        return _LEARNED_SID_CACHE
    
    try:
        if LEARNED_SID_FILE.exists():
            with open(LEARNED_SID_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _LEARNED_SID_CACHE = data.get("procedures", [])
        else:
            _LEARNED_SID_CACHE = []
    except Exception:
        _LEARNED_SID_CACHE = []
    
    return _LEARNED_SID_CACHE


def _load_learned_star(refresh: bool = False) -> List[Dict[str, Any]]:
    """Load learned STAR procedures from JSON file."""
    global _LEARNED_STAR_CACHE
    if _LEARNED_STAR_CACHE is not None and not refresh:
        return _LEARNED_STAR_CACHE
    
    try:
        if LEARNED_STAR_FILE.exists():
            with open(LEARNED_STAR_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _LEARNED_STAR_CACHE = data.get("procedures", [])
        else:
            _LEARNED_STAR_CACHE = []
    except Exception:
        _LEARNED_STAR_CACHE = []
    
    return _LEARNED_STAR_CACHE


def _is_on_known_turn_zone(lat: float, lon: float) -> bool:
    """Check if a point is within a known turn zone."""
    turn_zones = _load_learned_turns()
    
    for zone in turn_zones:
        zone_lat = zone.get("lat", 0)
        zone_lon = zone.get("lon", 0)
        zone_radius = zone.get("radius_nm", 2.0)
        
        dist = haversine_nm(lat, lon, zone_lat, zone_lon)
        if dist <= zone_radius + TURN_ZONE_TOLERANCE_NM:
            return True
    
    return False


def _is_on_sid_or_star(lat: float, lon: float) -> bool:
    """Check if a point is on a learned SID or STAR centerline."""
    # Check SIDs
    sids = _load_learned_sid()
    for proc in sids:
        centerline = proc.get("centerline", [])
        width = proc.get("width_nm", 3.0)
        
        if len(centerline) >= 2:
            coords = [(p["lat"], p["lon"]) for p in centerline if "lat" in p and "lon" in p]
            if coords:
                info = point_to_polyline_distance_nm((lat, lon), coords)
                if info["distance_nm"] <= width + SID_STAR_TOLERANCE_NM:
                    return True
    
    # Check STARs
    stars = _load_learned_star()
    for proc in stars:
        centerline = proc.get("centerline", [])
        width = proc.get("width_nm", 3.0)
        
        if len(centerline) >= 2:
            coords = [(p["lat"], p["lon"]) for p in centerline if "lat" in p and "lon" in p]
            if coords:
                info = point_to_polyline_distance_nm((lat, lon), coords)
                if info["distance_nm"] <= width + SID_STAR_TOLERANCE_NM:
                    return True
    
    return False


def _is_on_known_procedure(lat: float, lon: float) -> bool:
    """
    Check if a point is on a known turn zone, SID, or STAR.
    Used to suppress false positives in the turn rule.
    """
    return _is_on_known_turn_zone(lat, lon) or _is_on_sid_or_star(lat, lon)


# New O/D-based paths file
_lb_paths_file = Path(LEARNED_BEHAVIOR_CFG.get("paths_file", "rules/learned_paths.json"))
LEARNED_PATHS_FILE = (_lb_paths_file if _lb_paths_file.is_absolute() else (PROJECT_ROOT / _lb_paths_file)).resolve()
_LEARNED_OD_PATHS_CACHE: Optional[List[Dict[str, Any]]] = None


def _load_learned_od_paths(refresh: bool = False) -> List[Dict[str, Any]]:
    """Load O/D-based learned paths from the new format."""
    global _LEARNED_OD_PATHS_CACHE
    if _LEARNED_OD_PATHS_CACHE is not None and not refresh:
        return _LEARNED_OD_PATHS_CACHE
    
    try:
        if LEARNED_PATHS_FILE.exists():
            with open(LEARNED_PATHS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                paths_raw = data.get("paths", [])
                # Convert to the format expected by existing code
                _LEARNED_OD_PATHS_CACHE = []
                for p in paths_raw:
                    _LEARNED_OD_PATHS_CACHE.append({
                        "id": p.get("id", "unknown"),
                        "type": "od_learned",
                        "origin": p.get("origin"),
                        "destination": p.get("destination"),
                        "centerline": p.get("centerline", []),
                        "width_nm": p.get("width_nm", 4.0),
                        "num_flights": p.get("member_count", 0),
                    })
        else:
            _LEARNED_OD_PATHS_CACHE = []
    except Exception:
        _LEARNED_OD_PATHS_CACHE = []
    
    return _LEARNED_OD_PATHS_CACHE


def _load_path_library(refresh: bool = False) -> Dict[str, Any]:
    """
    Load the path library generated by build_path_library_v2.py.
    """
    global _PATH_LIBRARY_CACHE
    if _PATH_LIBRARY_CACHE is not None and not refresh:
        return _PATH_LIBRARY_CACHE

    try:
        if PATH_FILE.exists():
            with open(PATH_FILE, "r", encoding="utf-8") as f:
                library = json.load(f)
        else:
            library = {}
    except Exception:
        library = {}

    library.setdefault("paths", [])
    library.setdefault("emerging_paths", [])
    library.setdefault("emerging_buckets", [])
    library.setdefault("heatmap", {})

    _PATH_LIBRARY_CACHE = library
    return library


def _save_path_library(library: Dict[str, Any]) -> None:
    """Persist the in-memory path library."""
    global _PATH_LIBRARY_CACHE, _LEARNED_POLYGONS_CACHE
    PATH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH_FILE, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2)
    _PATH_LIBRARY_CACHE = library
    _LEARNED_POLYGONS_CACHE = None  # invalidate polygons


def _get_paths(include_emerging: bool = True, include_od_learned: bool = True) -> List[Dict[str, Any]]:
    """
    Get all paths from various sources.
    
    Args:
        include_emerging: Include emerging/candidate paths
        include_od_learned: Include O/D-based learned paths
        
    Returns:
        List of path dictionaries
    """
    library = _load_path_library()
    paths = list(library.get("paths", []))
    if include_emerging:
        paths += library.get("emerging_paths", [])
    if include_od_learned:
        paths += _load_learned_od_paths()
    return paths


def _get_learned_polygons() -> List[Any]:
    global _LEARNED_POLYGONS_CACHE
    if _LEARNED_POLYGONS_CACHE is not None:
        return _LEARNED_POLYGONS_CACHE

    polygons: List[Any] = []
    for path in _get_paths():
        centerline = path.get("centerline") or []
        coords = [(p["lat"], p["lon"]) for p in centerline if "lat" in p and "lon" in p]
        if len(coords) < 2:
            continue
        radius_nm = float(path.get("width_nm", DEFAULT_PATH_WIDTH_NM))
        poly = create_corridor_polygon(coords, radius_nm)
        if poly:
            polygons.append(poly)

    _LEARNED_POLYGONS_CACHE = polygons
    return polygons


def _load_learned_turns() -> List[Dict[str, Any]]:
    """
    Load learned turn spots from learned_paths.json (layers.turns).
    Each turn has: centroid_lat, centroid_lon, radius_nm, avg_alt, turn_direction, etc.
    """
    global _LEARNED_TURNS_CACHE
    if _LEARNED_TURNS_CACHE is not None:
        return _LEARNED_TURNS_CACHE

    turns: List[Dict[str, Any]] = []
    learned_paths_file = PROJECT_ROOT / "rules" / "learned_paths.json"
    try:
        if learned_paths_file.exists():
            with open(learned_paths_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # turns are stored under layers.turns
            layers = data.get("layers", {})
            turns = layers.get("turns", [])
    except Exception:
        turns = []

    _LEARNED_TURNS_CACHE = turns
    return turns


def _is_point_in_learned_turn(lat: float, lon: float) -> bool:
    """
    Check if a point (lat, lon) falls within any learned turn spot.
    Returns True if the point is within any turn's radius.
    """
    # Radius overrides for specific turn clusters (cluster_id -> new radius in nm)
    RADIUS_OVERRIDES = {
        "LEFT_10": 20,  # Expanded from 10.83nm to cover Amman area turns
    }
    
    # Global buffer to add to all turn radii (nm)
    RADIUS_BUFFER_NM = 0.0
    
    turns = _load_learned_turns()
    for turn in turns:
        centroid_lat = turn.get("centroid_lat")
        centroid_lon = turn.get("centroid_lon")
        cluster_id = turn.get("cluster_id", "")
        
        # Use override if available, otherwise use stored radius + buffer
        if cluster_id in RADIUS_OVERRIDES:
            radius_nm = RADIUS_OVERRIDES[cluster_id]
        else:
            radius_nm = turn.get("radius_nm", 5.0) + RADIUS_BUFFER_NM
        
        if centroid_lat is None or centroid_lon is None:
            continue
        
        dist_nm = haversine_nm(lat, lon, centroid_lat, centroid_lon)
        if dist_nm <= radius_nm:
            return True
    
    return False


def _load_heatmap(refresh: bool = False) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Load cached heatmap grid (flightability)."""
    global _HEATMAP_CACHE
    if _HEATMAP_CACHE is not None and not refresh:
        return _HEATMAP_CACHE

    library = _load_path_library()
    meta = library.get("heatmap", {}) or {}
    heatmap_path = Path(meta.get("heatmap_file", HEATMAP_FILE))

    grid = None
    try:
        if heatmap_path.exists():
            grid = np.load(heatmap_path)
            meta.setdefault("origin", meta.get("origin", [0.0, 0.0]))
            meta.setdefault("cell_size_deg", meta.get("cell_size_deg", HEATMAP_CELL_DEG))
            meta.setdefault("threshold", meta.get("threshold", HEATMAP_THRESHOLD))
            meta.setdefault("shape", list(grid.shape))
    except Exception:
        grid = None

    _HEATMAP_CACHE = (grid, meta)
    return _HEATMAP_CACHE


def _is_in_flightable_region(point: TrackPoint) -> bool:
    grid, meta = _load_heatmap()
    if grid is None:
        return True

    origin_lat, origin_lon = meta.get("origin", [0.0, 0.0])
    cell = float(meta.get("cell_size_deg", HEATMAP_CELL_DEG))
    rows, cols = meta.get("shape", grid.shape)

    r = int((point.lat - origin_lat) / cell)
    c = int((point.lon - origin_lon) / cell)

    if r < 0 or c < 0 or r >= rows or c >= cols:
        return False

    threshold = int(meta.get("threshold", HEATMAP_THRESHOLD))
    return grid[r, c] >= threshold


def _distance_to_path(point: TrackPoint, path_entry: Dict[str, Any]) -> Tuple[float, float]:
    """Return min lateral distance (nm) and normalized position along path (0-1)."""
    centerline = path_entry.get("centerline") or []
    coords = [(p["lat"], p["lon"]) for p in centerline if "lat" in p and "lon" in p]
    if len(coords) < 2:
        return float("inf"), 0.0
    info = point_to_polyline_distance_nm((point.lat, point.lon), coords)
    return float(info["distance_nm"]), float(info["position"])


def _compress_heading_signature(
    points: Sequence[TrackPoint],
    *,
    bin_seconds: int = 10,
    bin_size_deg: int = EMERGING_SIMILARITY_DEG,
) -> Tuple[int, ...]:
    """Build a compact heading signature for emerging path detection."""
    if not points:
        return ()

    ordered = sorted(points, key=lambda p: p.timestamp)
    next_bucket_ts = ordered[0].timestamp + bin_seconds
    bucket_headings: List[float] = []
    signature: List[int] = []
    prev = ordered[0]

    for p in ordered[1:]:
        heading = p.track
        if heading is None:
            heading = initial_bearing_deg(prev.lat, prev.lon, p.lat, p.lon)

        bucket_headings.append(heading % 360.0)

        if p.timestamp >= next_bucket_ts:
            mean_heading = sum(bucket_headings) / len(bucket_headings)
            signature.append(int(mean_heading // bin_size_deg))
            bucket_headings = []
            next_bucket_ts += bin_seconds

        prev = p

    if bucket_headings:
        mean_heading = sum(bucket_headings) / len(bucket_headings)
        signature.append(int(mean_heading // bin_size_deg))

    return tuple(signature)


def _update_emerging_buckets(
    ctx: RuleContext,
    off_path_points: Sequence[TrackPoint],
) -> Optional[Dict[str, Any]]:
    """
    Append the flight to an emerging-path candidate bucket and promote when enough samples arrive.
    """
    if not off_path_points:
        return None

    signature = _compress_heading_signature(off_path_points)
    if not signature:
        return None

    library = _load_path_library()
    buckets = library.setdefault("emerging_buckets", [])

    bucket = next((b for b in buckets if tuple(b.get("signature", ())) == signature), None)
    if bucket is None:
        bucket = {"signature": signature, "count": 0, "flight_ids": []}
        buckets.append(bucket)

    bucket["count"] = int(bucket.get("count", 0)) + 1
    bucket.setdefault("flight_ids", []).append(ctx.track.flight_id)
    bucket["last_updated"] = datetime.utcnow().isoformat() + "Z"

    promoted: Optional[Dict[str, Any]] = None
    if bucket["count"] >= EMERGING_BUCKET_SIZE:
        centerline = resample_track_points(ctx.track.points, num_samples=PATH_NUM_SAMPLES)
        if centerline is not None:
            coords = [(float(lat), float(lon)) for lat, lon, _ in centerline.tolist()]
            width_nm = DEFAULT_PATH_WIDTH_NM
            try:
                dists = [
                    point_to_polyline_distance_nm((p.lat, p.lon), coords)["distance_nm"]
                    for p in ctx.track.points
                ]
                if dists:
                    width_nm = max(float(np.std(dists)), 2.0)
            except Exception:
                pass

            emerging_list = library.setdefault("emerging_paths", [])
            path_id = f"emerging_{len(emerging_list) + 1}"
            promoted = {
                "id": path_id,
                "type": "emerging",
                "width_nm": width_nm,
                "centerline": [{"lat": lat, "lon": lon} for lat, lon in coords],
                "num_flights": bucket["count"],
                "created_from_signature": signature,
            }
            emerging_list.append(promoted)
            buckets.remove(bucket)

    _save_path_library(library)
    return promoted


@dataclass(frozen=True)
class Airport:
    code: str
    name: str
    lat: float
    lon: float
    elevation_ft:Optional[Any] = None

AIRPORT_ENTRIES = CONFIG.get("airports", [])
AIRPORTS: List[Airport] = [Airport(**entry) for entry in AIRPORT_ENTRIES]
AIRPORT_BY_CODE: Dict[str, Airport] = {a.code: a for a in AIRPORTS}


def is_bad_segment(prev: TrackPoint, curr: TrackPoint) -> bool:
    dt = curr.timestamp - prev.timestamp
    if dt <= 0:
        return True

    # 1. Teleport / impossible movement
    max_nm = (curr.gspeed or prev.gspeed or 350) * dt / 3600
    dist_nm = haversine_nm(prev.lat, prev.lon, curr.lat, curr.lon)
    if dist_nm > max_nm * 3:
        return True

    # 2. Impossible heading jump
    if prev.track is not None and curr.track is not None:
        dh = abs(((curr.track - prev.track + 540) % 360) - 180)
        if dh > 80:
            return True

    # 3. FR24 ocean gap -> ignore off-course
    # Far from land + cruise altitude = probably artifact
    nearest_airport, dist_ap = _nearest_airport(curr)
    if (curr.alt or 0) > 15000 and (dist_ap or 999) > 60:
        return True

    # 4. Zero-altitude glitch at high speed
    if (curr.alt or 0) < 200 and (curr.gspeed or 0) > 200:
        return True

    return False


def evaluate_rule(context: RuleContext, rule_id: int) -> RuleResult:
    evaluators = {
        1: _rule_emergency_squawk,
        2: _rule_extreme_altitude_change,
        3: _rule_abrupt_turn,
        4: _rule_dangerous_proximity,
        6: _rule_go_around,
        7: _rule_takeoff_return,
        8: _rule_diversion,
        9: _rule_low_altitude,
        # 10: _rule_signal_loss,
        12: _rule_unplanned_israel_landing,
        11: _rule_off_course,
    }
    evaluator = evaluators.get(rule_id)
    if evaluator is None:
        return RuleResult(rule_id=rule_id, matched=False, summary="Rule not implemented", details={})
    return evaluator(context)


def _rule_emergency_squawk(ctx: RuleContext) -> RuleResult:
    events = [
        {"timestamp": p.timestamp, "squawk": p.squawk}
        for p in ctx.track.sorted_points()
        if p.squawk and p.squawk.strip() in EMERGENCY_SQUAWKS
    ]
    matched = bool(events)
    summary = "Emergency code transmitted" if matched else "No emergency squawk detected"
    return RuleResult(1, matched, summary, {"events": events})


def _rule_extreme_altitude_change(ctx: RuleContext) -> RuleResult:
    points = ctx.track.sorted_points()
    events = []

    # precompute airport distances once
    distances = [_nearest_airport(p)[1] for p in points]

    def is_noise_sequence(start_idx: int, end_idx: int) -> bool:
        """
        Check if points in [start_idx, end_idx] form a noise sequence.
        A noise sequence has suspicious altitude values (like 0) but
        the flight resumes normal altitude after the sequence.
        """
        if start_idx < 0 or end_idx >= len(points) or start_idx >= end_idx:
            return False

        # Check if we have suspicious low altitudes in the sequence
        noise_points = points[start_idx:end_idx + 1]
        if not any(p.alt < 500 for p in noise_points):
            return False

        # Check altitude before the sequence
        if start_idx == 0:
            return False
        prev_alt = points[start_idx - 1].alt

        # Check altitude after the sequence
        if end_idx + 1 >= len(points):
            return False
        next_alt = points[end_idx + 1].alt

        # If before and after are similar (within reasonable range), it's likely noise
        if prev_alt < ALTITUDE_MIN_CRUISE_FT or next_alt < ALTITUDE_MIN_CRUISE_FT:
            return False

        # Check if altitudes before and after are similar (within 5000 ft)
        if abs(next_alt - prev_alt) < 5000:
            # Check if the noise sequence is short (less than 10 points) and far from airports
            sequence_duration = points[end_idx].timestamp - points[start_idx].timestamp
            min_dist = min(distances[max(0, start_idx):min(len(distances), end_idx + 1)])
            if sequence_duration < 300 and (min_dist is None or min_dist > 5):
                return True

        return False

    i = 0
    while i < len(points) - 1:
        prev = points[i]
        curr = points[i + 1]

        dt = curr.timestamp - prev.timestamp
        if dt <= 0 or dt > ALTITUDE_WINDOW_SECONDS:
            i += 1
            continue

        # cruise check
        if prev.alt < ALTITUDE_MIN_CRUISE_FT:
            i += 1
            continue

        # -----------------------------
        # NOISE DETECTION:
        # Check for consecutive noise points (e.g., 4 points with 0 alt)
        # when transitioning from normal altitude
        # -----------------------------
        if curr.alt < 500 and prev.alt >= ALTITUDE_MIN_CRUISE_FT:
            # Look ahead to find the end of a potential noise sequence
            noise_start = i + 1
            noise_end = noise_start
            while noise_end + 1 < len(points) and points[noise_end + 1].alt < 500:
                noise_end += 1

            # If we found a noise sequence (multiple consecutive points), check if it's noise
            if noise_end >= noise_start:
                # Check if we have at least 2 noise points OR single point that recovers quickly
                if (noise_end > noise_start or
                    (noise_end == noise_start and noise_end + 1 < len(points) and
                     points[noise_end + 1].alt >= ALTITUDE_MIN_CRUISE_FT)):
                    if is_noise_sequence(noise_start, noise_end):
                        # Skip the noise sequence and also skip the transition back to normal
                        # by comparing the point before noise to the point after noise
                        i = noise_end + 1
                        if i < len(points) - 1 and abs(points[noise_end + 1].alt - prev.alt) < 5000:
                            # Altitudes before and after noise are similar, skip this transition too
                            i += 1
                        continue

        # Also check if we're transitioning from noise back to normal (safety net for missed cases)
        if prev.alt < 500 and curr.alt >= ALTITUDE_MIN_CRUISE_FT:
            # Look backwards to find if prev is part of a noise sequence
            noise_end = i
            noise_start = i
            while noise_start > 0 and points[noise_start - 1].alt < 500:
                noise_start -= 1

            # Check if this is a noise sequence (even single point)
            if noise_start <= noise_end and noise_start > 0:
                before_noise_alt = points[noise_start - 1].alt
                if (before_noise_alt >= ALTITUDE_MIN_CRUISE_FT and
                    abs(curr.alt - before_noise_alt) < 5000):
                    # This is noise recovering to normal, skip it
                    # Check duration and distance to confirm it's noise
                    if noise_start < noise_end:  # Multiple noise points
                        sequence_duration = points[noise_end].timestamp - points[noise_start].timestamp
                        min_dist = min(distances[max(0, noise_start):min(len(distances), noise_end + 1)])
                        if sequence_duration < 300 and (min_dist is None or min_dist > 5):
                            i += 1
                            continue
                    else:  # Single noise point
                        min_dist = distances[i] if i < len(distances) else None
                        if min_dist is None or min_dist > 5:
                            i += 1
                            continue

        # -----------------------------
        # Glitch Filter: 0 altitude
        # -----------------------------
        # If altitude drops to 0 from a significant altitude, it's a glitch.
        if (curr.alt or 0) <= 0:
             # If we were flying (> 500 ft), a sudden 0 is fake.
             if (prev.alt or 0) > 500:
                 i += 1
                 continue

        # -----------------------------
        # SOFT FILTER 1:
        # Ignore collapses to 0 ft far from airports
        # -----------------------------
        if curr.alt == 0 and distances[i + 1] and distances[i + 1] > 7:
            # noise signature: 35000 → 0 → 35000
            if i + 2 < len(points):
                next_alt = points[i + 2].alt
                if abs(next_alt - prev.alt) < 3000:
                    i += 1
                    continue

        # -----------------------------
        # SOFT FILTER 2:
        # Impossible physics: 0 ft + high speed
        # -----------------------------
        if curr.alt < 200 and curr.gspeed > 200 and distances[i + 1] > 3:
            i += 1
            continue

        delta = curr.alt - prev.alt

        if abs(delta) >= ALTITUDE_CHANGE_FT:
            rate = delta / dt
            events.append({
                "timestamp": curr.timestamp,
                "delta_ft": round(delta, 2),
                "rate_ft_per_s": round(rate, 2),
            })

        i += 1

    matched = bool(events)
    summary = "Detected rapid altitude changes" if matched else "Altitude profile nominal"
    return RuleResult(2, matched, summary, {"events": events})


def _rule_abrupt_turn(ctx: RuleContext) -> RuleResult:
    points = ctx.track.sorted_points()
    events = []

    if len(points) < 4:
        return RuleResult(3, False, "Not enough datapoints", {})

    # -----------------------------
    # 1. Detect abrupt single-point turns (existing logic kept)
    # -----------------------------

    def smooth_heading(i):
        if i == 0 or i == len(points) - 1:
            return points[i].track
        prev_h = points[i - 1].track
        curr_h = points[i].track
        next_h = points[i + 1].track
        if prev_h is None or curr_h is None or next_h is None:
            return curr_h
        return (prev_h + curr_h + next_h) / 3.0

    for i in range(1, len(points)):
        prev = points[i - 1]
        curr = points[i]

        if is_bad_segment(prev, curr):
            continue

        if prev.track is None or curr.track is None:
            continue

        dt = curr.timestamp - prev.timestamp
        if dt <= 0 or dt > TURN_WINDOW_SECONDS:
            continue

        dist_nm = haversine_nm(prev.lat, prev.lon, curr.lat, curr.lon)
        max_possible_nm = (curr.gspeed or 300) * dt / 3600.0
        if dist_nm > max_possible_nm * 3.0:
            continue

        prev_h = smooth_heading(i - 1)
        curr_h = smooth_heading(i)

        if prev_h is None or curr_h is None:
            continue

        diff = _heading_diff(curr_h, prev_h)

        # aerodynamically impossible
        if abs(diff) / dt > 5.0:
            continue

        # ignore if too slow
        if (curr.gspeed or 0.0) < TURN_MIN_SPEED_KTS:
            continue

        if abs(diff) >= TURN_THRESHOLD_DEG:
            # Suppress if within a learned turn spot
            if _is_point_in_learned_turn(curr.lat, curr.lon):
                continue
            
            events.append({
                "timestamp": curr.timestamp,
                "turn_deg": round(diff, 2),
                "dt_s": dt,
                "smoothed_prev": round(prev_h, 2),
                "smoothed_curr": round(curr_h, 2),
            })

    # ==========================================================
    # ========== 1b. Moderate Gradual Turn Detection ===========
    # ==========================================================
    # Detect turns >= MODERATE_TURN_MIN_DEG accumulated over a sliding window
    # that are NOT in learned zones. This catches S-turns and vectoring.
    
    MODERATE_TURN_WINDOW_S = 180  # 3 minute window for accumulated turn
    MODERATE_TURN_MIN_SPEED = 150  # kts - must be moving at reasonable speed
    MODERATE_TURN_MIN_ALT = 2000   # ft - above pattern altitude
    
    def _calc_accumulated_turn(start_idx: int, end_idx: int) -> float:
        """Calculate accumulated heading change between two indices."""
        total = 0.0
        for j in range(start_idx + 1, end_idx + 1):
            if points[j].track is None or points[j-1].track is None:
                continue
            delta = _heading_diff(points[j].track, points[j-1].track)
            total += abs(delta)
        return total
    
    # Track detected moderate turn windows to avoid duplicates
    moderate_turn_windows = []
    
    for i in range(len(points)):
        start_p = points[i]
        if start_p.track is None:
            continue
        if (start_p.gspeed or 0) < MODERATE_TURN_MIN_SPEED:
            continue
        if (start_p.alt or 0) < MODERATE_TURN_MIN_ALT:
            continue
        
        # Look for end point within window
        for j in range(i + 3, len(points)):  # Need at least 3 points for a turn
            end_p = points[j]
            duration = end_p.timestamp - start_p.timestamp
            
            if duration < 30:  # Minimum 30 seconds
                continue
            if duration > MODERATE_TURN_WINDOW_S:
                break
            
            if end_p.track is None:
                continue
            if (end_p.gspeed or 0) < MODERATE_TURN_MIN_SPEED:
                continue
            if (end_p.alt or 0) < MODERATE_TURN_MIN_ALT:
                continue
            
            # Calculate accumulated turn
            acc_turn = _calc_accumulated_turn(i, j)
            
            # Check if this is a moderate turn (>= threshold but < extreme)
            if acc_turn >= MODERATE_TURN_MIN_DEG and acc_turn < TURN_THRESHOLD_DEG:
                # Find midpoint of turn
                mid_idx = (i + j) // 2
                mid_p = points[mid_idx]
                
                # ONLY flag if NOT in a learned turn zone
                # (We don't check SID/STAR here because a turn NOT in a learned
                # turn zone is unusual even if near a procedure path)
                if not _is_point_in_learned_turn(mid_p.lat, mid_p.lon):
                    # Check for overlap with already detected windows
                    overlap = False
                    for (ws, we) in moderate_turn_windows:
                        if not (j < ws or i > we):  # Overlapping
                            overlap = True
                            break
                    
                    if not overlap:
                        moderate_turn_windows.append((i, j))
                        events.append({
                            "type": "moderate_turn_outside_zone",
                            "timestamp": mid_p.timestamp,
                            "start_ts": start_p.timestamp,
                            "end_ts": end_p.timestamp,
                            "duration_s": duration,
                            "accumulated_turn_deg": round(acc_turn, 1),
                            "lat": mid_p.lat,
                            "lon": mid_p.lon,
                            "alt_ft": mid_p.alt,
                            "start_heading": start_p.track,
                            "end_heading": end_p.track,
                        })
                        break  # Found a turn from this start, move on

    # ==========================================================
    # ========== 2. NEW → Clean Holding Pattern Detection =======
    # ==========================================================

    TURN_MIN_DURATION = 45        # must be at least 45 seconds (allows 180 turn)
    TURN_RATE_MAX = 12.0          # deg/sec threshold (increased to 12 as per rule update)
    TURN_MIN_SPEED = 80           # kts
    TURN_MIN_ALT = 1500           # ft
    ACC_THRESHOLD_180 = 220       # half-turn (tightened to avoid known patterns)
    ACC_THRESHOLD_360 = 240       # full orbit (slightly relaxed to capture sparse loops)
    TURN_MAX_ACCEL_KTS_S = 10.0   # max acceleration to reject speed glitches

    def signed_delta(h1, h0):
        """Return signed heading delta in [-180, 180]."""
        return ((h1 - h0 + 540) % 360) - 180

    # Holding-pattern scan
    for start_idx in range(len(points)):
        start_p = points[start_idx]

        if start_p.track is None or (start_p.gspeed or 0) < TURN_MIN_SPEED or (start_p.alt or 0) < TURN_MIN_ALT:
            continue

        # START POINT SANITY CHECK
        # If the start point itself is the result of a massive instantaneous jump from the previous point,
        # it is likely a glitch start and should not be used as the anchor for a turn analysis.
        if start_idx > 0:
            p_prev = points[start_idx - 1]
            dt_prev = start_p.timestamp - p_prev.timestamp
            # Ensure previous point is recent enough to matter
            if 0 < dt_prev < TURN_ACC_WINDOW and p_prev.track is not None:
                dh_prev = signed_delta(start_p.track, p_prev.track)
                rate_prev = abs(dh_prev) / dt_prev
                accel_prev = abs((start_p.gspeed or 0) - (p_prev.gspeed or 0)) / dt_prev

                if rate_prev > TURN_RATE_MAX or accel_prev > TURN_MAX_ACCEL_KTS_S:
                    continue

        cumulative = 0.0
        direction = None

        prev_idx = start_idx
        for end_idx in range(start_idx + 1, len(points)):
            p0 = points[prev_idx]
            p1 = points[end_idx]

            if p1.timestamp - start_p.timestamp > TURN_ACC_WINDOW:
                break

            if p1.track is None:
                continue
            if (p1.gspeed or 0) < TURN_MIN_SPEED:
                continue
            if (p1.alt or 0) < TURN_MIN_ALT:
                continue

            dt = p1.timestamp - p0.timestamp
            if dt <= 0:
                continue
            
            # --- NEW: Ignore points descending within 5 miles of an airport ---
            # Even if above TURN_MIN_ALT, we ignore descent segments near airports
            # as they often involve maneuvering that isn't a "holding pattern" anomaly.
            nearest_ap, dist_ap = _nearest_airport(p1)
            if dist_ap is not None and dist_ap < 5.0:
                # Check if descending
                if (p1.alt or 0) < (p0.alt or 0):
                    continue

            # --- High Density Logic ---
            # If dt >= 10s, we have a gap in the data. We can't reliably compute
            # the heading change across this gap, so we reset prev_idx to this point
            # and continue looking for consecutive close points.
            if dt >= 10.0:
                # Reset to this point - it becomes the new reference for subsequent points
                prev_idx = end_idx
                continue

            # Acceleration check (reject speed glitches)
            # When we encounter a speed glitch, we skip the point but still update prev_idx
            # so subsequent points are compared against a recent reference.
            accel = abs((p1.gspeed or 0) - (p0.gspeed or 0)) / dt
            if accel > TURN_MAX_ACCEL_KTS_S:
                prev_idx = end_idx  # Update reference to avoid cascading failures
                continue

            # Make sure we use the last GOOD point for heading calculation
            # p0 is the previous point (prev_idx). In this loop, prev_idx is updated
            # only at the end of the loop when a point is accepted.
            dh = signed_delta(p1.track, p0.track)
            # turn rate sanity
            if abs(dh) / dt > TURN_RATE_MAX:
                continue

            # TRACK vs BEARING Check (Detect Sensor Failure / Crabbing Glitch)
            # If the reported heading (track) is wildly different from the actual path (bearing),
            # the sensor data is likely invalid.
            # We only check this if moving at reasonable speed (not hovering) and sufficient distance.
            if (p1.gspeed or 0) > 50 and dt > 2:
                bearing = initial_bearing_deg(p0.lat, p0.lon, p1.lat, p1.lon)
                # Calculate difference between reported track and actual bearing
                diff_track_bearing = abs(((p1.track - bearing + 540) % 360) - 180)
                
                # If difference is > 90 degrees, the heading is likely garbage (e.g. sensor stuck or flipped)
                if diff_track_bearing > 90:
                    continue

            # Update previous valid index
            prev_idx = end_idx

            # establish/validate direction
            if dh == 0:
                continue

            sign = 1 if dh > 0 else -1
            if direction is None:
                direction = sign
            elif sign != direction:
                # tolerate tiny opposite blips (≤10 deg) without killing accumulation
                if abs(dh) <= 10:
                    continue
                break

            cumulative += abs(dh)

            duration = p1.timestamp - start_p.timestamp
            if duration < TURN_MIN_DURATION:
                continue

            # detect events
            if cumulative >= ACC_THRESHOLD_360 and duration >= 60:
                # For 360 degree turns (full orbit), we only suppress if very close
                # to an airport (likely a hold for landing sequence).
                # A full 360 is an unusual maneuver that warrants attention even
                # on a known route or in a learned corridor - these corridors
                # represent normal flight paths, not holding pattern areas.
                nearest_ap, dist_ap = _nearest_airport(p1)
                near_airport = dist_ap is not None and dist_ap < 6.0

                if not near_airport:
                    events.append({
                        "type": "holding_pattern",
                        "timestamp": p1.timestamp,
                        "start_ts": start_p.timestamp,
                        "end_ts": p1.timestamp,
                        "duration_s": duration,
                        "cumulative_turn_deg": round(cumulative, 2),
                        "pattern": "360_turn"
                    })
                break

            if cumulative >= ACC_THRESHOLD_180 and duration >= 70:
                polygons = _get_learned_polygons()
                is_suppressed = False
                latlon = (p1.lat, p1.lon)
                for poly in polygons:
                    if is_point_in_polygon(latlon, poly):
                        is_suppressed = True
                        break
                nearest_ap, dist_ap = _nearest_airport(p1)
                near_airport = dist_ap is not None and dist_ap < 6.0
                
                # Check if on known turn zone, SID, or STAR
                on_known_procedure = _is_on_known_procedure(p1.lat, p1.lon)

                if not is_suppressed and not near_airport and not on_known_procedure:
                    events.append({
                        "type": "holding_pattern",
                        "timestamp": p1.timestamp,
                        "start_ts": start_p.timestamp,
                        "end_ts": p1.timestamp,
                        "duration_s": duration,
                        "cumulative_turn_deg": round(cumulative, 2),
                        "pattern": "180_turn"
                    })
                    break
                # If 180 turn is suppressed, continue looking for 360 turn
                # (don't break - the cumulative might reach 360 threshold)

    # ----------------------------------------------------------
    # Fallback: geometric loop detector (path vs displacement)
    # If the above stricter scan missed an obvious loop (e.g. sparse points),
    # use path/disp ratio and cumulative heading to decide.
    # ----------------------------------------------------------
    if not events:
        SPEED_FALLBACK = 80
        ALT_FALLBACK = 1500
        MIN_DURATION = 70          # avoid flagging short orbits in known patterns
        MAX_DURATION = 240
        MIN_HEADING_ACC = 240      # tighten to reduce false positives on known turns
        MIN_PATH_DISP_RATIO = 1.35

        def heading_acc(points_slice):
            acc = 0.0
            for a, b in zip(points_slice, points_slice[1:]):
                if a.track is None or b.track is None:
                    continue
                acc += abs(((b.track - a.track + 540) % 360) - 180)
            return acc

        def path_len(points_slice):
            total = 0.0
            for a, b in zip(points_slice, points_slice[1:]):
                total += haversine_nm(a.lat, a.lon, b.lat, b.lon)
            return total

        event_added = False
        for i in range(len(points)):
            start = points[i]
            if (start.gspeed or 0) < SPEED_FALLBACK or (start.alt or 0) < ALT_FALLBACK:
                continue
            for j in range(i + 2, len(points)):
                end = points[j]
                duration = end.timestamp - start.timestamp
                if duration < MIN_DURATION:
                    continue
                if duration > MAX_DURATION:
                    break
                if (end.gspeed or 0) < SPEED_FALLBACK or (end.alt or 0) < ALT_FALLBACK:
                    continue
                slice_pts = points[i : j + 1]
                disp = haversine_nm(start.lat, start.lon, end.lat, end.lon)
                path = path_len(slice_pts)
                if disp <= 0:
                    continue
                ratio = path / disp
                if ratio < MIN_PATH_DISP_RATIO:
                    continue
                acc = heading_acc(slice_pts)
                if acc < MIN_HEADING_ACC:
                    continue

                # Suppress if within learned corridors, UNLESS the turn is extreme
                # An extreme turn (>500° accumulated or ratio >2.5) is unusual even in
                # a normal corridor and should still be flagged.
                is_extreme_turn = acc >= 500 or ratio >= 2.5
                
                if not is_extreme_turn:
                    polygons = _get_learned_polygons()
                    is_suppressed = False
                    latlon = (end.lat, end.lon)
                    for poly in polygons:
                        if is_point_in_polygon(latlon, poly):
                            is_suppressed = True
                            break
                    if is_suppressed:
                        continue
                    
                    # Check if on known turn zone, SID, or STAR
                    if _is_on_known_procedure(end.lat, end.lon):
                        continue

                events.append({
                    "type": "holding_pattern",
                    "timestamp": end.timestamp,
                    "start_ts": start.timestamp,
                    "end_ts": end.timestamp,
                    "duration_s": duration,
                    "cumulative_turn_deg": round(acc, 2),
                    "pattern": "360_turn_fallback",
                    "path_nm": round(path, 2),
                    "disp_nm": round(disp, 2),
                    "path_disp_ratio": round(ratio, 2),
                })
                event_added = True
                break
            if event_added:
                break

    matched = bool(events)
    summary = "Abrupt heading change or holding pattern observed" if matched else "Heading profile nominal"
    return RuleResult(3, matched, summary, {"events": events})


def _rule_dangerous_proximity(ctx: RuleContext) -> RuleResult:
    points = ctx.track.sorted_points()
    if not points:
        return RuleResult(4, False, "No track data", {})

    # If we don't have a repository, we can't check for other flights
    if ctx.repository is None:
        return RuleResult(4, False, "Skipped: No flight database available", {})

    events = []

    # Pull candidate points
    start = points[0].timestamp - PROXIMITY_TIME_WINDOW
    end = points[-1].timestamp + PROXIMITY_TIME_WINDOW
    
    try:
        nearby_points = [
            p for p in ctx.repository.fetch_points_between(start, end)
            if p.flight_id != ctx.track.flight_id
        ]
    except AttributeError:
         return RuleResult(4, False, "Skipped: Repository error", {})

    for point in points:
        if point.alt < 100:
            continue

        # Skip points near airports (within ~2 miles) - normal traffic patterns
        if PROXIMITY_AIRPORT_EXCLUSION_NM > 0:
            _, airport_dist = _nearest_airport(point)
            if airport_dist <= PROXIMITY_AIRPORT_EXCLUSION_NM:
                continue

        # --- find closest timestamp for each other flight ---
        candidates = [
            other for other in nearby_points
            if abs(other.timestamp - point.timestamp) <= 5   # STRICT time sync
        ]

        for other in candidates:
            if other.alt < 100:
                continue
            
            # Require both aircraft to be above 3000 ft to avoid ground proximity alerts
            if point.alt < 3000 or other.alt < 3000:
                continue

            dist = haversine_nm(point.lat, point.lon, other.lat, other.lon)
            alt_diff = abs(point.alt - other.alt)

            # impossible values → skip
            if dist < 0.5 and alt_diff < 200 and point.alt < 1000:
                continue

            # heading sanity (optional)
            # if point.track is not None and other.track is not None:
            #     if abs(point.track - other.track) > 130:
            #         continue

            if dist <= PROXIMITY_DISTANCE_NM and alt_diff <= PROXIMITY_ALTITUDE_FT:
                events.append({
                    "timestamp": point.timestamp,
                    "other_flight": other.flight_id,
                    "other_callsign": other.callsign,
                    "distance_nm": round(dist, 2),
                    "altitude_diff_ft": round(alt_diff, 1),
                })
                break

    matched = bool(events)
    summary = "Proximity alert triggered" if matched else "No proximity conflicts"
    return RuleResult(4, matched, summary, {"events": events})


RUNWAY_HEADINGS = {
    "LCRA": [100, 280],   # RAF Akrotiri
    "ALJAWZAH": [135, 315],
    "HEGR": [160, 340],   # El Gora Airport
    "LLBG": [76, 256],    # Ben Gurion
    "LLHA": [155, 335],   # Haifa
    "LLER": [9, 189],     # Ramon Intl
    "LLSD": [140, 320],   # Sde Dov
    "LLBS": [143, 323],   # Beersheba
    "LLET": [86, 266],    # Eilat
    "LLOV": [30, 210],    # Ovda
    "LLNV": [100, 280],   # Nevatim AFB
    "LLMG": [80, 260],    # Megiddo
    "LLHZ": [160, 340],   # Herzliya
    "OLBA": [155, 335],   # Beirut
    "OLKA": [90, 270],    # Rayak AB
    "OJAI": [75, 255],    # Queen Alia Intl
    "OJAM": [61, 241],    # Marka Intl
    "OJAQ": [20, 200],    # Aqaba
    "OJMF": [150, 330],   # Mafraq AB
    "OJJR": [113, 293],   # Jerash
    "OJMN": [142, 322],   # Ma'an
    "OSDI": [50, 230],    # Damascus Intl
    "OSKL": [75, 255],    # Al Qusayr
    "OSAP": [35, 215],    # An Nasiriya AB
}

def _heading_diff(h1, h2):
    """Smallest circular difference between two headings."""
    diff = abs(h1 - h2) % 360
    return diff if diff <= 180 else 360 - diff

def _is_runway_aligned(point, airport_code, tolerance=30):
    """Check if heading is aligned with any runway direction."""
    if airport_code not in RUNWAY_HEADINGS:
        return True  # fallback, don't block detection

    for rh in RUNWAY_HEADINGS[airport_code]:
        if _heading_diff(point.track, rh) <= tolerance:
            return True
    return False


def _rule_go_around(ctx: RuleContext) -> RuleResult:
    events = []

    for airport in AIRPORTS:
        segments = _points_near_airport(ctx.track.sorted_points(), airport, GO_AROUND_RADIUS_NM)
        if len(segments) < 3:
            continue

        # Sort segments by timestamp for neighbor checking (should be sorted already, but ensure it)
        segments_by_time = sorted(segments, key=lambda p: p.timestamp)
        
        # Sort by altitude to find candidates for lowest
        sorted_by_alt = sorted(segments, key=lambda p: p.alt)
        
        lowest = None
        MAX_VS_FT_SEC = 200.0  # ~12000 fpm, filter impossible vertical moves

        for candidate in sorted_by_alt:
            idx = segments_by_time.index(candidate)
            is_glitch = False
            
            # Check previous (descent into point)
            if idx > 0:
                prev = segments_by_time[idx - 1]
                dt = candidate.timestamp - prev.timestamp
                dy = abs(candidate.alt - prev.alt)
                if dt > 0 and (dy / dt) > MAX_VS_FT_SEC:
                    is_glitch = True
            
            # Check next (climb out of point)
            if not is_glitch and idx < len(segments_by_time) - 1:
                next_p = segments_by_time[idx + 1]
                dt = next_p.timestamp - candidate.timestamp
                dy = abs(next_p.alt - candidate.alt)
                if dt > 0 and (dy / dt) > MAX_VS_FT_SEC:
                    is_glitch = True
            
            if not is_glitch:
                lowest = candidate
                break
        
        if lowest is None:
            continue

        # Filter impossible altitudes (glitches) and use AGL for threshold
        elevation = airport.elevation_ft or 0
        agl = lowest.alt - elevation

        # 1. Sanity check: if >200ft below airport, it's a data error (e.g. 0ft report at 2400ft airport)
        if agl < -200:
            continue

        # 2. Check low altitude relative to ground (AGL)
        # Add small buffer for baro/ADS‑B and airport elevation mismatches
        LOW_ALT_BUFFER_FT = 150.0
        if agl > GO_AROUND_LOW_ALT_FT + LOW_ALT_BUFFER_FT:
            continue

        # NEW: require runway alignment at lowest point
        if not _is_runway_aligned(lowest, airport.code):
            continue

        # NEW: require clear descent BEFORE the lowest point
        before_low = [p for p in segments if p.timestamp < lowest.timestamp]
        if not before_low:
            continue

        descent_amount = min(p.alt for p in before_low) - lowest.alt
        if descent_amount < 300:  # require descent trend of at least 300 ft
            continue

        # climb after low point
        after_low = [p for p in segments if p.timestamp > lowest.timestamp]
        if not after_low:
            continue

        max_climb = max(p.alt for p in after_low) - lowest.alt
        if max_climb >= GO_AROUND_RECOVERY_FT:
            events.append(
                {
                    "airport": airport.code,
                    "timestamp": lowest.timestamp,
                    "min_alt_ft": round(lowest.alt, 1),
                    "recovered_ft": round(max_climb, 1),
                    "descent_into_low_ft": round(descent_amount, 1),
                    "aligned_with_runway": True
                }
            )

    matched = bool(events)
    summary = "Go-around detected" if matched else "No go-around patterns"
    return RuleResult(6, matched, summary, {"events": events})


def _rule_takeoff_return(ctx: RuleContext) -> RuleResult:
    points = ctx.track.sorted_points()
    if len(points) < 4:
        return RuleResult(7, False, "Insufficient points", {})

    origin_airport, origin_dist = _nearest_airport(points[0])
    if origin_airport is None or origin_dist > RETURN_NEAR_AIRPORT_NM:
        return RuleResult(7, False, "Origin airport unknown", {})

    origin_elev = (origin_airport.elevation_ft or 0)

    takeoff_point = next((p for p in points if (p.alt or 0) >= origin_elev + RETURN_TAKEOFF_ALT_FT), None)
    if takeoff_point is None:
        return RuleResult(7, False, "Flight never departed", {})

    # Require that the aircraft actually traveled outbound before considering a return.
    max_outbound_nm = max(
        haversine_nm(p.lat, p.lon, origin_airport.lat, origin_airport.lon)
        for p in points
        if p.timestamp >= takeoff_point.timestamp
    )
    if max_outbound_nm < RETURN_MIN_OUTBOUND_NM:
        return RuleResult(7, False, "No meaningful outbound leg", {"max_outbound_nm": max_outbound_nm})

    for point in points:
        if point.timestamp <= takeoff_point.timestamp:
            continue
        distance_home = haversine_nm(point.lat, point.lon, origin_airport.lat, origin_airport.lon)
        
        # Check landing (using AGL)
        if (point.alt or 0) < origin_elev + RETURN_LANDING_ALT_FT and distance_home <= RETURN_NEAR_AIRPORT_NM:
            dt = point.timestamp - takeoff_point.timestamp
            if dt <= RETURN_TIME_LIMIT_SECONDS and dt >= RETURN_MIN_ELAPSED_SECONDS:
                info = {
                    "airport": origin_airport.code,
                    "takeoff_ts": takeoff_point.timestamp,
                    "landing_ts": point.timestamp,
                    "elapsed_s": dt,
                    "max_outbound_nm": max_outbound_nm,
                }
                return RuleResult(7, True, "Return-to-field detected", info)
    return RuleResult(7, False, "No immediate return detected", {})


def _rule_diversion(ctx: RuleContext) -> RuleResult:
    metadata = ctx.metadata
    if metadata is None or metadata.planned_destination is None:
        return RuleResult(8, False, "No planned destination provided", {})

    planned = AIRPORT_BY_CODE.get(metadata.planned_destination.upper())
    if planned is None:
        return RuleResult(8, False, "Planned destination not in airport list", {})

    last_point = ctx.track.sorted_points()[-1] if ctx.track.points else None
    if last_point is None:
        return RuleResult(8, False, "No track data", {})

    actual_airport, actual_dist = _nearest_airport(last_point)
    if actual_airport is None or actual_dist > DIVERSION_NEAR_AIRPORT_NM:
        return RuleResult(8, True, "Flight ended away from any known airport", {"distance_to_airport_nm": actual_dist})

    matched = actual_airport.code != planned.code
    summary = "Flight diverted to alternate airport" if matched else "Flight landed at planned destination"
    details = {
        "planned": planned.code,
        "actual": actual_airport.code,
        "distance_nm": round(actual_dist, 2),
    }
    return RuleResult(8, matched, summary, details)


def _rule_low_altitude(ctx: RuleContext) -> RuleResult:
    points = ctx.track.sorted_points()
    events = []
    last_alt = None
    last_ts = None

    for p in points:
        alt = p.alt or 0.0
        speed = p.gspeed or 0.0
        vs = p.vspeed or 0.0

        nearest, dist = _nearest_airport(p)

        # -----------------------
        # 0. Skip ascending / climb-out
        # -----------------------

        # A. If climbing > 300 ft/min → normal ascent
        if vs > 300:
            last_alt = alt
            last_ts = p.timestamp
            continue

        # B. If was just on ground (<50 ft) in last 60 seconds → skip
        if last_alt is not None and last_alt < 50:
            dt = p.timestamp - last_ts
            if dt < 60:   # 1 minute from takeoff
                last_alt = alt
                last_ts = p.timestamp
                continue

        # C. If within 25 NM of airport and climbing → normal
        if nearest and dist < 25 and vs > 0:
            last_alt = alt
            last_ts = p.timestamp
            continue

        # -----------------------
        # HARD SANITY CHECKS
        # -----------------------

        if alt < 200 and (dist is None or dist > 15):
            last_alt = alt
            last_ts = p.timestamp
            continue

        if alt < 800 and speed > 200 and (dist is None or dist > 10):
            last_alt = alt
            last_ts = p.timestamp
            continue

        # Sudden impossible descent
        if last_alt is not None and last_ts is not None:
            dt = p.timestamp - last_ts
            if dt > 0:
                rate = (last_alt - alt) / dt
            else:
                rate = 0.0
            
            if rate > 100 and (dist is None or dist > 10):
                last_alt = alt
                last_ts = p.timestamp
                continue

        # -----------------------
        # LOW ALTITUDE CONFIRMATION
        # -----------------------

        if alt < LOW_ALTITUDE_THRESHOLD_FT:
            idx = points.index(p)
            if idx + 1 < len(points):
                if (points[idx + 1].alt or 0.0) >= LOW_ALTITUDE_THRESHOLD_FT:
                    # single flicker
                    last_alt = alt
                    last_ts = p.timestamp
                    continue

            # Check standard airport radius
            if nearest and dist <= LOW_ALTITUDE_AIRPORT_RADIUS_NM:
                last_alt = alt
                last_ts = p.timestamp
                continue

            # -----------------------
            # APPROACH PATTERN DETECTION:
            # If descending towards an airport, use larger radius (35-40 NM)
            # -----------------------
            if nearest and dist and dist <= 40 and vs < 0:  # Descending
                # Check if heading is roughly towards the airport
                if p.track is not None:
                    bearing_to_airport = initial_bearing_deg(p.lat, p.lon, nearest.lat, nearest.lon)
                    heading_diff = _heading_diff(bearing_to_airport, p.track)
                    
                    # Allow heading within 45 degrees of airport direction
                    if abs(heading_diff) <= 45:
                        # Check if in a continuous descent pattern
                        # Look back at last few points to confirm descent pattern
                        descent_confirmed = False
                        
                        # Check if altitude is decreasing (descent pattern)
                        if idx >= 1:
                            prev_alt_1 = points[idx - 1].alt or 0.0
                            if prev_alt_1 > alt:  # Altitude is decreasing
                                descent_confirmed = True
                                # Check one more point back if available
                                if idx >= 2:
                                    prev_alt_2 = points[idx - 2].alt or 0.0
                                    if prev_alt_2 > prev_alt_1:  # Continued descent
                                        descent_confirmed = True
                        
                        # Also check if distance to airport is decreasing
                        if idx > 0:
                            prev_point = points[idx - 1]
                            prev_dist = haversine_nm(prev_point.lat, prev_point.lon, nearest.lat, nearest.lon)
                            if dist < prev_dist:  # Getting closer to airport
                                descent_confirmed = True
                        
                        # Speed check: approach speeds are typically 100-180 kts
                        # For descending flights near airports, allow legitimate approach patterns
                        if 90 <= speed <= 200 and descent_confirmed:
                            # This looks like a legitimate approach pattern
                            last_alt = alt
                            last_ts = p.timestamp
                            continue

            # -----------------------
            # REAL EVENT
            # -----------------------
            events.append({
                "timestamp": p.timestamp,
                "alt_ft": round(alt, 1),
                "speed_kts": speed,
                "distance_to_airport_nm": round(dist or -1, 1),
                "vspeed_fpm": vs,
            })

        last_alt = alt
        last_ts = p.timestamp

    matched = bool(events)
    summary = "Low altitude detected outside protected zones" if matched else "Altitude remained above minima"
    return RuleResult(9, matched, summary, {"events": events})


def _rule_signal_loss(ctx: RuleContext) -> RuleResult:
    points = ctx.track.sorted_points()
    if len(points) < 2:
        return RuleResult(10, False, "Insufficient points", {})

    gaps = []
    prev = points[0]
    for curr in points[1:]:
        dt = curr.timestamp - prev.timestamp
        
        # Check if on ground (AGL < 300)
        # We need nearest airport to know ground level
        prev_airport, prev_dist = _nearest_airport(prev)
        curr_airport, curr_dist = _nearest_airport(curr)
        
        prev_elev = (prev_airport.elevation_ft if prev_airport and prev_dist < 10 else 0) or 0
        curr_elev = (curr_airport.elevation_ft if curr_airport and curr_dist < 10 else 0) or 0
        
        prev_agl = (prev.alt or 0) - prev_elev
        curr_agl = (curr.alt or 0) - curr_elev
        
        if prev_agl < 300 or curr_agl < 300:
            prev = curr
            continue
            
        if dt >= SIGNAL_GAP_SECONDS:
            gaps.append({"start_ts": prev.timestamp, "end_ts": curr.timestamp, "gap_s": dt})
        prev = curr

    matched = len(gaps) >= SIGNAL_REPEAT_COUNT
    return RuleResult(10, matched, "Signal loss" if matched else "Nominal", {"gaps": gaps})


def _rule_unplanned_israel_landing(ctx: RuleContext) -> RuleResult:
    metadata = ctx.metadata

    if metadata is None or not metadata.planned_destination:
        return RuleResult(12, False, "Missing planned destination", {})

    planned = metadata.planned_destination.upper()

    points = ctx.track.sorted_points()
    if not points:
        return RuleResult(12, False, "No track data", {})

    last_point = points[-1]

    # Find actual nearest airport (landing airport)
    actual_airport, actual_dist = _nearest_airport(last_point)

    if actual_airport is None or actual_dist > UNPLANNED_LANDING_RADIUS_NM:
        return RuleResult(12, False, "Flight did not land at a known airport", {})

    actual = actual_airport.code

    # -----------------------------
    # Core Logic:
    # landed somewhere different than the plan → anomaly
    # -----------------------------
    if planned != actual:
        return RuleResult(
            12,
            True,
            f"Flight landed at {actual} instead of planned {planned}",
            {
                "planned": planned,
                "actual": actual,
                "distance_nm": round(actual_dist, 2),
                "type": "wrong_landing_airport",
            }
        )

    # -----------------------------
    # Normal
    # -----------------------------
    return RuleResult(
        12,
        False,
        "Flight landed at planned destination",
        {
            "planned": planned,
            "actual": actual,
            "distance_nm": round(actual_dist, 2),
        }
    )


def _rule_off_course(ctx: RuleContext) -> RuleResult:
    """
    Path adherence using primary/secondary/emerging paths + heatmap.
    - On path if within path width.
    - Wrong region if entering a low-activity heatmap cell.
    - Emerging detector buckets far-off trajectories.
    """

    points = ctx.track.sorted_points()
    if not points:
        return RuleResult(11, False, "No track data", {})

    paths = _get_paths()
    if not paths:
        return RuleResult(11, False, "No path library loaded", {})

    on_path: List[Dict[str, Any]] = []
    off_path: List[Dict[str, Any]] = []
    wrong_region: List[Dict[str, Any]] = []
    far_points: List[TrackPoint] = []
    assignments: Dict[str, int] = defaultdict(int)

    for idx, p in enumerate(points):
        if idx > 0 and is_bad_segment(points[idx - 1], p):
            continue
        if (p.alt or 0) <= 4000:
            continue

        best: Optional[Tuple[str, float, float, float, str]] = None  # id, dist, pos, width, type
        for path in paths:
            dist_nm, pos = _distance_to_path(p, path)
            width = float(path.get("width_nm", DEFAULT_PATH_WIDTH_NM))
            if best is None or dist_nm < best[1]:
                best = (path.get("id", "unknown"), dist_nm, pos, width, path.get("type", "primary"))

        if best is None:
            continue

        path_id, dist_nm, pos, width_nm, path_type = best
        if dist_nm <= width_nm:
            assignments[path_id] += 1
            on_path.append(
                {
                    "timestamp": p.timestamp,
                    "path_id": path_id,
                    "distance_nm": round(dist_nm, 2),
                    "position": round(pos, 3),
                    "type": path_type,
                }
            )
        else:
            off_record = {
                "timestamp": p.timestamp,
                "distance_nm": round(dist_nm, 2),
                "lat": p.lat,
                "lon": p.lon,
            }
            off_path.append(off_record)
            if dist_nm >= EMERGING_DISTANCE_NM:
                far_points.append(p)
            if not _is_in_flightable_region(p):
                wrong_region.append(off_record)

    promoted = None
    if far_points:
        promoted = _update_emerging_buckets(ctx, far_points)

    matched = len(off_path) >= MIN_OFF_COURSE_POINTS or len(wrong_region) > 0
    summary = "Flight deviated from known paths" if matched else "Flight stayed within known corridors"
    if wrong_region:
        summary = "Entered low-activity region"

    details = {
        "on_path_points": len(on_path),
        "off_path_points": len(off_path),
        "wrong_region_points": len(wrong_region),
        "assignments": dict(assignments),
        "samples": {"on_path": on_path[:50], "off_path": off_path[:50], "wrong_region": wrong_region[:50]},
        "emerging_promoted": promoted["id"] if promoted else None,
        "threshold_points": MIN_OFF_COURSE_POINTS,
    }

    return RuleResult(11, matched, summary, details)


def _points_near_airport(points: Sequence[TrackPoint], airport: Airport, radius_nm: float) -> List[TrackPoint]:
    return [p for p in points if haversine_nm(p.lat, p.lon, airport.lat, airport.lon) <= radius_nm]


def _nearest_airport(point: TrackPoint) -> Tuple[Optional[Airport], float]:
    best_airport: Optional[Airport] = None
    best_distance: float = float('inf')
    for airport in AIRPORTS:
        distance = haversine_nm(point.lat, point.lon, airport.lat, airport.lon)
        if distance < best_distance:
            best_distance = distance
            best_airport = airport
    return best_airport, best_distance


def _pairwise(points: Sequence[TrackPoint]):
    iterator = iter(points)
    prev = next(iterator, None)
    for current in iterator:
        if prev is not None:
            yield prev, current
        prev = current


def has_point_above_altitude(track: FlightTrack, altitude_ft: float = 5000.0) -> bool:
    """
    Check if a flight has at least one point above the specified altitude.
    """
    c = 0
    for point in track.points:
        if point.alt > altitude_ft:
            if c > 4:
                return True
            else:
                c += 1
    return False
