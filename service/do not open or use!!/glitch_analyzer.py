from typing import List, Dict, Any
import statistics

# Import from core using relative or absolute path logic
import sys
import os
sys.path.append(os.getcwd())

from core.models import FlightTrack, TrackPoint

def analyze_glitches(flight: FlightTrack) -> Dict[str, Any]:
    """
    Analyzes a flight track for glitches (spikes, frozen positions, gaps).
    Returns metrics: spike_ratio, freeze_ratio, missing_ratio, glitch_score.
    """
    points = flight.sorted_points()
    if not points:
        return {
            "spike_ratio": 0.0,
            "freeze_ratio": 0.0,
            "missing_ratio": 0.0,
            "glitch_score": 0.0,
            "details": []
        }

    total_points = len(points)
    if total_points < 2:
        return {
            "spike_ratio": 0.0,
            "freeze_ratio": 0.0,
            "missing_ratio": 0.0,
            "glitch_score": 0.0,
            "details": []
        }

    # Extract speeds for median calculation (ignore Nones)
    speeds = [p.gspeed for p in points if p.gspeed is not None]
    median_speed = statistics.median(speeds) if speeds else 0
    
    # Safety check for median speed to avoid division by zero or tiny numbers
    if median_speed < 50: 
        median_speed = 50 # assume at least 50kts for reasonable flight

    spike_count = 0
    freeze_count = 0
    gap_count = 0 
    severe_glitch_found = False

    # Track consecutive frozen points
    current_freeze_run = 1
    
    for i in range(1, total_points):
        p1 = points[i-1]
        p2 = points[i]
        
        dt = p2.timestamp - p1.timestamp
        
        # 3. Missing-data gaps
        if dt > 60:
            gap_count += 1
            if dt > 300: # 5 min gap is severe
                severe_glitch_found = True
            
        # 1. Speed / heading spikes
        is_spike = False
        
        # Check groundspeed
        speed_val = p2.gspeed if p2.gspeed is not None else 0
        if speed_val > 1200:
            is_spike = True
            severe_glitch_found = True # Mach 2+ is definitely a glitch for commercial
        elif median_speed > 0 and speed_val > 2.5 * median_speed: # Increased multiplier slightly to 2.5x to be safe
            is_spike = True
            
        # Check heading change
        if not is_spike and dt < 5 and dt > 0:
            if p1.track is not None and p2.track is not None:
                diff = abs(p2.track - p1.track)
                if diff > 180:
                    diff = 360 - diff
                if diff > 90:
                    is_spike = True
        
        if is_spike:
            spike_count += 1

        # 2. Frozen positions
        if (abs(p1.lat - p2.lat) < 0.00001 and 
            abs(p1.lon - p2.lon) < 0.00001 and 
            abs(p1.alt - p2.alt) < 1):
            current_freeze_run += 1
        else:
            if current_freeze_run >= 3:
                freeze_count += current_freeze_run
            current_freeze_run = 1
            
    # Handle last freeze run
    if current_freeze_run >= 3:
        freeze_count += current_freeze_run

    # --- SCORING LOGIC V2 (More Sensitive) ---
    
    # 1. Ratios
    spike_ratio = spike_count / total_points
    freeze_ratio = freeze_count / total_points
    missing_ratio = gap_count / (total_points - 1)
    
    # 2. Weighted Score
    # We boost the impact of spikes significantly. 
    # Even 2-3% spikes is huge data quality issue.
    
    # Sigmoid-like impact: 5% spikes should be nearly 1.0 score
    score_spike = min(1.0, spike_ratio * 20) # 0.05 * 20 = 1.0
    score_freeze = min(1.0, freeze_ratio * 5) # 0.2 * 5 = 1.0
    score_gap = min(1.0, missing_ratio * 10) # 0.1 * 10 = 1.0
    
    # Combined
    glitch_score = max(score_spike, score_freeze, score_gap)
    
    # Override if severe glitch found
    if severe_glitch_found:
        glitch_score = max(glitch_score, 0.8)

    return {
        "spike_ratio": round(spike_ratio, 4),
        "freeze_ratio": round(freeze_ratio, 4),
        "missing_ratio": round(missing_ratio, 4),
        "glitch_score": round(glitch_score, 4),
        "metrics": {
            "spike_count": spike_count,
            "freeze_count": freeze_count,
            "gap_count": gap_count,
            "median_speed": median_speed,
            "severe_glitch": severe_glitch_found
        }
    }
