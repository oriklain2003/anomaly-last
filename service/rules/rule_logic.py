from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.config import load_rule_config
from core.geodesy import cross_track_distance_nm, haversine_nm, initial_bearing_deg
from core.models import FlightTrack, RuleContext, RuleResult, TrackPoint

CONFIG = load_rule_config()
RULES = CONFIG.get("rules", {})


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

DIVERSION_CFG = _require_rule_config("diversion")
DIVERSION_NEAR_AIRPORT_NM = float(DIVERSION_CFG["near_airport_nm"])

LOW_ALTITUDE_CFG = _require_rule_config("low_altitude")
LOW_ALTITUDE_THRESHOLD_FT = float(LOW_ALTITUDE_CFG["threshold_ft"])
LOW_ALTITUDE_AIRPORT_RADIUS_NM = float(LOW_ALTITUDE_CFG["airport_radius_nm"])

SIGNAL_CFG = _require_rule_config("signal_loss")
SIGNAL_GAP_SECONDS = int(SIGNAL_CFG["gap_seconds"])
SIGNAL_REPEAT_COUNT = int(SIGNAL_CFG["repeat_count"])


@dataclass(frozen=True)
class Airport:
    code: str
    name: str
    lat: float
    lon: float


AIRPORT_ENTRIES = CONFIG.get("airports", [])
AIRPORTS: List[Airport] = [Airport(**entry) for entry in AIRPORT_ENTRIES]
AIRPORT_BY_CODE: Dict[str, Airport] = {a.code: a for a in AIRPORTS}


def evaluate_rule(context: RuleContext, rule_id: int) -> RuleResult:
    evaluators = {
        1: _rule_emergency_squawk,
        2: _rule_extreme_altitude_change,
        3: _rule_abrupt_turn,
        4: _rule_dangerous_proximity,
        # 5: _rule_route_deviation,
        6: _rule_go_around,
        7: _rule_takeoff_return,
        8: _rule_diversion,
        9: _rule_low_altitude,
        # 10: _rule_signal_loss,
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

    # Need at least 4 points to smooth and validate
    if len(points) < 4:
        return RuleResult(3, False, "Not enough datapoints", {})

    def smooth_heading(i):
        """3-point moving average smoothing"""
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

        if prev.track is None or curr.track is None:
            continue

        dt = curr.timestamp - prev.timestamp
        if dt <= 0 or dt > TURN_WINDOW_SECONDS:
            continue

        # Reject impossible position jumps
        dist_nm = haversine_nm(prev.lat, prev.lon, curr.lat, curr.lon)
        max_possible_nm = (curr.gspeed or 300) * dt / 3600.0
        if dist_nm > max_possible_nm * 3.0:
            # FR24/ADSB glitch → skip
            continue

        # Smooth headings
        prev_h = smooth_heading(i - 1)
        curr_h = smooth_heading(i)

        if prev_h is None or curr_h is None:
            continue

        diff = _heading_diff(curr_h, prev_h)

        # Reject aerodynamically impossible turn rate
        # Typical max turn: ~3°/sec (5°/sec extreme)
        if abs(diff) / dt > 5.0:  # deg/s
            continue

        # Still require high speed (turning matters only then)
        if (curr.gspeed or 0.0) < TURN_MIN_SPEED_KTS:
            continue

        # Require multiple consecutive points showing same turn direction
        if i + 2 < len(points):
            n1 = _heading_diff(smooth_heading(i + 1), curr_h)
            n2 = _heading_diff(smooth_heading(i + 2), smooth_heading(i + 1))
            if not (abs(n1) > 20 and abs(n2) > 20):
                continue

        if abs(diff) >= TURN_THRESHOLD_DEG:
            events.append({
                "timestamp": curr.timestamp,
                "turn_deg": round(diff, 2),
                "dt_s": dt,
                "smoothed_prev": round(prev_h, 2),
                "smoothed_curr": round(curr_h, 2),
                "distance_nm": round(dist_nm, 3),
                "turn_rate_deg_s": round(abs(diff) / dt, 2)
            })

    # ---------------------------------------
    # NEW: Cumulative Turn / Holding Pattern
    # ---------------------------------------
    # Scan for accumulated heading changes over a longer window (e.g. 360 deg in 5 mins)
    
    # Optimization: Only check every Nth point to save cycles, or just check all.
    # Since N is small (<1000 usually), checking all is fine.

    matched_intervals = []

    for start_idx in range(len(points)):
        start_p = points[start_idx]
        cumulative_turn = 0.0

        for end_idx in range(start_idx + 1, len(points)):
            curr_p = points[end_idx]
            prev_p = points[end_idx - 1]

            # Skip missing data
            if curr_p.track is None or prev_p.track is None:
                continue

            # ------------------------------------
            # NEW: Reject taxi / ground noise
            # ------------------------------------
            if (prev_p.gspeed or 0) < 80 or (curr_p.gspeed or 0) < 80:
                continue

            # ------------------------------------
            # NEW: Reject low altitude phases
            # ------------------------------------
            if (curr_p.alt or 0) < 500:
                continue

            # Time window check
            if curr_p.timestamp - start_p.timestamp > TURN_ACC_WINDOW:
                break

            dt = curr_p.timestamp - prev_p.timestamp
            if dt <= 0:
                continue

            # Heading delta
            d_h = _heading_diff(curr_p.track, prev_p.track)

            # ------------------------------------
            # NEW: Reject impossible turn rates
            # ------------------------------------
            turn_rate = abs(d_h) / dt
            if turn_rate > 5.0:  # deg/sec
                continue

            cumulative_turn += d_h

            # ------------------------------------
            # NEW: Require minimum duration to avoid noise
            # ------------------------------------
            if curr_p.timestamp - start_p.timestamp < 45:
                continue

            # Threshold for holding/orbit detection
            if abs(cumulative_turn) >= TURN_ACC_DEG:

                # Prevent duplicates
                is_duplicate = False
                for m in matched_intervals:
                    if (
                            start_p.timestamp >= m["start_ts"] and
                            curr_p.timestamp <= m["end_ts"]
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    evt = {
                        "timestamp": curr_p.timestamp,
                        "type": "holding_pattern",
                        "start_ts": start_p.timestamp,
                        "end_ts": curr_p.timestamp,
                        "duration_s": curr_p.timestamp - start_p.timestamp,
                        "cumulative_turn_deg": round(cumulative_turn, 2)
                    }
                    events.append(evt)
                    matched_intervals.append(evt)

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

        # --- find closest timestamp for each other flight ---
        candidates = [
            other for other in nearby_points
            if abs(other.timestamp - point.timestamp) <= 5   # STRICT time sync
        ]

        for other in candidates:
            if other.alt < 100:
                continue

            dist = haversine_nm(point.lat, point.lon, other.lat, other.lon)
            alt_diff = abs(point.alt - other.alt)

            # impossible values → skip
            if dist < 0.5 and alt_diff < 200 and point.alt < 1000:
                continue

            # heading sanity (optional)
            if point.track is not None and other.track is not None:
                if abs(point.track - other.track) > 130:
                    continue

            if dist <= PROXIMITY_DISTANCE_NM and alt_diff <= PROXIMITY_ALTITUDE_FT:
                events.append({
                    "timestamp": point.timestamp,
                    "other_flight": other.flight_id,
                    "distance_nm": round(dist, 2),
                    "altitude_diff_ft": round(alt_diff, 1),
                })
                break

    matched = bool(events)
    summary = "Proximity alert triggered" if matched else "No proximity conflicts"
    return RuleResult(4, matched, summary, {"events": events})
# def _rule_route_deviation(ctx: RuleContext) -> RuleResult:
#     points = ctx.track.sorted_points()
#     if len(points) < 3:
#         return RuleResult(5, False, "Insufficient points for route analysis", {})
#
#     origin = (points[0].lat, points[0].lon)
#     destination = (points[-1].lat, points[-1].lon)
#
#     deviations = [
#         cross_track_distance_nm(origin, destination, (p.lat, p.lon)) for p in points[1:-1]
#     ]
#     max_dev = max(deviations) if deviations else 0.0
#     matched = max_dev >= ROUTE_DEVIATION_NM
#     summary = (
#         f"Cross-track deviation {max_dev:.1f} NM exceeds threshold"
#         if matched
#         else "Route deviation within acceptable bounds"
#     )
#     return RuleResult(5, matched, summary, {"max_deviation_nm": round(max_dev, 2)})


RUNWAY_HEADINGS = {
"LCRA": [100, 280],   # RAF Akrotiri (Runway 10/28)

"ALJAWZAH": [135, 315],
"HEGR": [160, 340],   # El Gora Airport (Runway 16/34)



    # --------------------------
    # 🇮🇱 ISRAEL
    # --------------------------
    "LLBG": [76, 256],       # Ben Gurion (Runway 12/30 ~ 116/296 also exists but rarely used)
    "LLHA": [155, 335],      # Haifa (Runway 16/34)
    "LLER": [9, 189],        # Ramon Intl (Runway 01/19)
    "LLSD": [140, 320],      # Sde Dov (closed but heading is correct)
    "LLBS": [143, 323],      # Beersheba / Teyman (Runway 14/32)
    "LLET": [86, 266],       # Eilat (old airport, runway 08/26)
    "LLOV": [30, 210],       # Ovda (Runway 03/21)
    "LLNV": [100, 280],      # Nevatim AFB (Runway 10/28)
    "LLMG": [80, 260],       # Megiddo (Runway 08/26)
    "LLHZ": [160, 340],      # Herzliya (Runway 16/34)

    # --------------------------
    # 🇱🇧 LEBANON
    # --------------------------
    "OLBA": [155, 335],      # Beirut Rafic Hariri (Runway 16/34)
    "OLKA": [90, 270],       # Rayak AB (Runway 09/27)

    # --------------------------
    # 🇯🇴 JORDAN
    # --------------------------
    "OJAI": [75, 255],       # Queen Alia Intl (Runway 08/26)
    "OJAM": [61, 241],       # Marka Intl (Runway 06/24 ~ 058/238)
    "OJAQ": [20, 200],       # Aqaba King Hussein (Runway 02/20)
    "OJMF": [150, 330],      # Mafraq AB (Runway 15/33)
    "OJJR": [113, 293],      # Jerash (Runway 11/29)
    "OJMN": [142, 322],      # Ma'an (Runway 14/32)

    # --------------------------
    # 🇸🇾 SYRIA (within your bounding box)
    # --------------------------
    "OSDI": [50, 230],       # Damascus Intl (Runway 05/23)
    "OSKL": [75, 255],       # Al Qusayr (Runway 08/26)
    "OSAP": [35, 215],       # An Nasiriya AB (Runway 04/22)
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

        # lowest altitude point
        lowest = min(segments, key=lambda p: p.alt)
        if lowest.alt > GO_AROUND_LOW_ALT_FT:
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

    takeoff_point = next((p for p in points if p.alt >= RETURN_TAKEOFF_ALT_FT), None)
    if takeoff_point is None:
        return RuleResult(7, False, "Flight never departed", {})

    for point in points:
        if point.timestamp <= takeoff_point.timestamp:
            continue
        distance_home = haversine_nm(point.lat, point.lon, origin_airport.lat, origin_airport.lon)
        if point.alt < RETURN_LANDING_ALT_FT and distance_home <= RETURN_NEAR_AIRPORT_NM:
            dt = point.timestamp - takeoff_point.timestamp
            if dt <= RETURN_TIME_LIMIT_SECONDS:
                info = {
                    "airport": origin_airport.code,
                    "takeoff_ts": takeoff_point.timestamp,
                    "landing_ts": point.timestamp,
                    "elapsed_s": dt,
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

        # Ignore gaps when the aircraft is low/ground mode
        if prev.alt < 300 or curr.alt < 300:
            prev = curr
            continue

        # Real in-flight signal loss
        if dt >= SIGNAL_GAP_SECONDS:
            gaps.append({
                "start_ts": prev.timestamp,
                "end_ts": curr.timestamp,
                "gap_s": dt
            })

        prev = curr

    matched = len(gaps) >= SIGNAL_REPEAT_COUNT
    summary = "In-flight signal loss observed" if matched else "Signal continuity nominal"

    return RuleResult(10, matched, summary, {"gaps": gaps})


def _points_near_airport(points: Sequence[TrackPoint], airport: Airport, radius_nm: float) -> List[TrackPoint]:
    return [p for p in points if haversine_nm(p.lat, p.lon, airport.lat, airport.lon) <= radius_nm]


def _nearest_airport(point: TrackPoint) -> Tuple[Optional[Airport], Optional[float]]:
    best_airport: Optional[Airport] = None
    best_distance: Optional[float] = None
    for airport in AIRPORTS:
        distance = haversine_nm(point.lat, point.lon, airport.lat, airport.lon)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_airport = airport
    return best_airport, best_distance


def _heading_diff(current: float, previous: float) -> float:
    diff = (current - previous + 180) % 360 - 180
    return diff


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
    
    Args:
        track: The flight track to check
        altitude_ft: The altitude threshold in feet (default: 5000.0)
        
    Returns:
        True if any point in the track is above the threshold, False otherwise
    """
    c = 0
    for point in track.points:
        if point.alt > altitude_ft:
            if c > 4:
                return True
            else:
                c += 1
    return False

