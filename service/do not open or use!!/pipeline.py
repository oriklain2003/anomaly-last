import sys
import os
import logging
from typing import Dict, Any

# Fix DLL load error on Windows by importing torch first
try:
    import torch
except ImportError:
    pass
except OSError:
    pass

# Add root to path to import existing modules
root_path = str(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from anomaly_pipeline import AnomalyPipeline
from core.models import FlightTrack
from new_service.glitch_analyzer import analyze_glitches
from new_service.llm_layer import analyze_anomaly_with_llm

logger = logging.getLogger(__name__)

class EnhancedPipeline:
    def __init__(self):
        self.base_pipeline = AnomalyPipeline()

    def analyze(self, flight: FlightTrack) -> Dict[str, Any]:
        # 1. Run existing pipeline
        base_result = self.base_pipeline.analyze(flight)
        
        # If not an anomaly (by summary decision), return base result
        # Note: AnomalyPipeline.analyze returns a dict with a "summary" key.
        # summary: { is_anomaly: bool, ... }
        
        summary = base_result.get("summary", {})
        is_anomaly = summary.get("is_anomaly", False)
        
        if not is_anomaly:
            return base_result

        # 2. It's an anomaly! Run Glitch Analyzer
        glitch_report = analyze_glitches(flight)
        
        # 3. Prepare data for LLM
        # Summarize flight data (start, end, bounds, few points)
        points = flight.sorted_points()
        summary_points = []
        full_points_data = [] # For CSV generation
        
        if points:
            # Full points for CSV (raw data)
            for p in points:
                full_points_data.append({
                    "ts": p.timestamp,
                    "lat": p.lat,
                    "lon": p.lon,
                    "alt": p.alt,
                    "spd": p.gspeed,
                    "hdg": p.track,
                    "vspd": p.vspeed
                })

            # Sampled points for JSON context (keep small)
            step = max(1, len(points) // 20)
            for i in range(0, len(points), step):
                p = points[i]
                summary_points.append({
                    "ts": p.timestamp,
                    "lat": p.lat,
                    "lon": p.lon,
                    "alt": p.alt,
                    "spd": p.gspeed,
                    "hdg": p.track
                })
                
        flight_summary = {
            "flight_id": flight.flight_id,
            "point_count": len(points),
            "duration": (points[-1].timestamp - points[0].timestamp) if points else 0,
            "sampled_trajectory": summary_points,
            "full_trajectory": full_points_data # New field for CSV generation
        }

        # 4. Run LLM Layer
        llm_result = analyze_anomaly_with_llm(
            anomaly_report=base_result,
            flight_data_summary=flight_summary,
            glitch_analysis=glitch_report
        )
        
        # 5. Combine Results
        combined_result = base_result.copy()
        combined_result["glitch_analysis"] = glitch_report
        combined_result["llm_analysis"] = llm_result
        
        return combined_result

# Singleton for easy access
_enhanced_pipeline = None

def get_enhanced_pipeline():
    global _enhanced_pipeline
    if _enhanced_pipeline is None:
        _enhanced_pipeline = EnhancedPipeline()
    return _enhanced_pipeline
