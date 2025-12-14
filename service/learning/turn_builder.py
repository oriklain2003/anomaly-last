"""
Turn builder - learns known turn zones from normal flight data.

Approach:
1. Scan all flights for turn events (180-300 degrees cumulative)
2. Extract turn midpoint location, altitude, speed, direction
3. Run DBSCAN on (lat, lon) to find turn hotspots
4. Output cluster centers as "known turn zones"
5. Output: rules/learned_turns.json
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from learning.data_loader import FlightDataLoader
from learning.utils import (
    cluster_points_dbscan,
    detect_turns,
    TurnEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class LearnedTurnZone:
    """Represents a learned turn zone."""
    id: int
    lat: float
    lon: float
    radius_nm: float
    avg_alt_ft: float
    angle_range_deg: List[float]  # [min, max]
    avg_speed_kts: float
    member_count: int
    directions: Dict[str, int]  # {"left": N, "right": M}


class TurnBuilder:
    """
    Builds a turn zone library by detecting and clustering turns.
    """
    
    def __init__(
        self,
        data_loader: FlightDataLoader,
        min_turn_deg: float = 180.0,
        max_turn_deg: float = 300.0,
        min_duration_s: int = 30,
        max_duration_s: int = 600,
        min_speed_kts: float = 50.0,  # Reduced from 80 - many valid turns at 60-70 kts
        min_alt_ft: float = 1000.0,
        cluster_eps_nm: float = 3.0,
        cluster_min_samples: int = 3,
        min_zone_radius_nm: float = 0.5,  # Filter out zones that are too small
        max_zone_radius_nm: float = 15.0  # Filter out zones that are too spread out
    ):
        """
        Initialize the turn builder.
        
        Args:
            data_loader: Data loader instance
            min_turn_deg: Minimum cumulative turn angle
            max_turn_deg: Maximum cumulative turn angle
            min_duration_s: Minimum turn duration in seconds
            max_duration_s: Maximum turn duration in seconds
            min_speed_kts: Minimum speed to consider (reduced to 50kts)
            min_alt_ft: Minimum altitude to consider
            cluster_eps_nm: DBSCAN epsilon in nautical miles
            cluster_min_samples: DBSCAN min_samples
            min_zone_radius_nm: Minimum zone radius to keep (filter tiny zones)
            max_zone_radius_nm: Maximum zone radius to keep (filter spread out zones)
        """
        self.data_loader = data_loader
        self.min_turn_deg = min_turn_deg
        self.max_turn_deg = max_turn_deg
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.min_speed_kts = min_speed_kts
        self.min_alt_ft = min_alt_ft
        self.cluster_eps_nm = cluster_eps_nm
        self.cluster_min_samples = cluster_min_samples
        self.min_zone_radius_nm = min_zone_radius_nm
        self.max_zone_radius_nm = max_zone_radius_nm
    
    def _extract_all_turns(self, flights) -> List[TurnEvent]:
        """Extract turns from all flights."""
        all_turns = []
        
        for flight in flights:
            points = flight.sorted_points()
            turns = detect_turns(
                points,
                min_deg=self.min_turn_deg,
                max_deg=self.max_turn_deg,
                min_duration_s=self.min_duration_s,
                max_duration_s=self.max_duration_s,
                min_speed_kts=self.min_speed_kts,
                min_alt_ft=self.min_alt_ft
            )
            all_turns.extend(turns)
        
        return all_turns
    
    def _cluster_turns(self, turns: List[TurnEvent]) -> List[LearnedTurnZone]:
        """Cluster turns by location and compute zone statistics."""
        if not turns:
            return []
        
        # Extract midpoints for clustering
        points = [(t.mid_lat, t.mid_lon) for t in turns]
        
        # Cluster
        labels = cluster_points_dbscan(
            points,
            eps_nm=self.cluster_eps_nm,
            min_samples=self.cluster_min_samples
        )
        
        # Group turns by cluster
        clusters: Dict[int, List[TurnEvent]] = defaultdict(list)
        for turn, label in zip(turns, labels):
            if label >= 0:  # Skip noise
                clusters[label].append(turn)
        
        # Build zones
        zones = []
        filtered_count = 0
        
        for cluster_id, cluster_turns in clusters.items():
            # Compute center (mean lat/lon)
            lats = [t.mid_lat for t in cluster_turns]
            lons = [t.mid_lon for t in cluster_turns]
            center_lat = np.mean(lats)
            center_lon = np.mean(lons)
            
            # Compute radius (max distance from center)
            from core.geodesy import haversine_nm
            distances = [
                haversine_nm(center_lat, center_lon, t.mid_lat, t.mid_lon)
                for t in cluster_turns
            ]
            radius = max(distances) + 0.5  # Add buffer
            radius = max(radius, 1.0)  # Minimum 1nm
            
            # Filter zones with inconsistent radius (too small or too large)
            if radius < self.min_zone_radius_nm:
                filtered_count += 1
                continue
            if radius > self.max_zone_radius_nm:
                filtered_count += 1
                continue
            
            # Compute statistics
            alts = [t.mid_alt for t in cluster_turns]
            speeds = [t.avg_speed for t in cluster_turns]
            angles = [t.cumulative_deg for t in cluster_turns]
            
            # Count directions
            left_count = sum(1 for t in cluster_turns if t.direction < 0)
            right_count = sum(1 for t in cluster_turns if t.direction > 0)
            
            zones.append(LearnedTurnZone(
                id=cluster_id,
                lat=float(center_lat),
                lon=float(center_lon),
                radius_nm=float(radius),
                avg_alt_ft=float(np.mean(alts)),
                angle_range_deg=[float(min(angles)), float(max(angles))],
                avg_speed_kts=float(np.mean(speeds)),
                member_count=len(cluster_turns),
                directions={"left": left_count, "right": right_count}
            ))
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} zones (too small/large radius)")
        
        return zones
    
    def build(self, progress_callback=None) -> List[LearnedTurnZone]:
        """
        Build the turn zone library from all normal flights.
        
        Args:
            progress_callback: Optional callback(message: str) for progress
            
        Returns:
            List of learned turn zones
        """
        def log_progress(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)
        
        log_progress("Loading normal flights for turn detection...")
        flights = list(self.data_loader.iter_normal_flights(min_points=20))
        log_progress(f"Loaded {len(flights)} flights")
        
        log_progress(f"Detecting turns ({self.min_turn_deg}-{self.max_turn_deg} degrees)...")
        all_turns = self._extract_all_turns(flights)
        log_progress(f"Found {len(all_turns)} turn events")
        
        log_progress("Clustering turns by location...")
        zones = self._cluster_turns(all_turns)
        log_progress(f"Identified {len(zones)} turn zones")
        
        return zones
    
    def save(self, zones: List[LearnedTurnZone], output_path: Path) -> None:
        """
        Save the turn zone library to JSON.
        
        Args:
            zones: List of learned turn zones
            output_path: Output file path
        """
        output = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "parameters": {
                "min_turn_deg": self.min_turn_deg,
                "max_turn_deg": self.max_turn_deg,
                "cluster_eps_nm": self.cluster_eps_nm
            },
            "total_zones": len(zones),
            "zones": [
                {
                    "id": int(z.id),
                    "lat": round(z.lat, 6),
                    "lon": round(z.lon, 6),
                    "radius_nm": round(z.radius_nm, 2),
                    "avg_alt_ft": round(z.avg_alt_ft, 0),
                    "angle_range_deg": [round(a, 1) for a in z.angle_range_deg],
                    "avg_speed_kts": round(z.avg_speed_kts, 1),
                    "member_count": z.member_count,
                    "directions": z.directions
                }
                for z in zones
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved turn zone library to {output_path}")


def run_turn_builder(
    research_db: Path,
    feedback_db: Optional[Path] = None,
    training_db: Optional[Path] = None,
    last_db: Optional[Path] = None,
    output_path: Path = Path("rules/learned_turns.json"),
    **kwargs
) -> List[LearnedTurnZone]:
    """
    Run the turn builder pipeline.
    
    Args:
        research_db: Path to research.db
        feedback_db: Path to feedback.db (optional)
        training_db: Path to training_dataset.db (optional)
        last_db: Path to last.db (optional, primary data source)
        output_path: Output file path
        **kwargs: Additional arguments for TurnBuilder
        
    Returns:
        List of learned turn zones
    """
    loader = FlightDataLoader(
        research_db=research_db,
        feedback_db=feedback_db,
        training_db=training_db,
        last_db=last_db
    )
    
    builder = TurnBuilder(data_loader=loader, **kwargs)
    zones = builder.build()
    builder.save(zones, output_path)
    
    return zones


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Build turn zone library from flight data")
    parser.add_argument("--research-db", type=Path, default=Path("research.db"))
    parser.add_argument("--feedback-db", type=Path, default=Path("training_ops/feedback.db"))
    parser.add_argument("--training-db", type=Path, default=Path("training_ops/training_dataset.db"))
    parser.add_argument("--output", type=Path, default=Path("rules/learned_turns.json"))
    parser.add_argument("--min-turn-deg", type=float, default=180.0)
    parser.add_argument("--max-turn-deg", type=float, default=300.0)
    parser.add_argument("--cluster-eps", type=float, default=3.0)
    parser.add_argument("--min-samples", type=int, default=3)
    
    args = parser.parse_args()
    
    run_turn_builder(
        research_db=args.research_db,
        feedback_db=args.feedback_db,
        training_db=args.training_db,
        output_path=args.output,
        min_turn_deg=args.min_turn_deg,
        max_turn_deg=args.max_turn_deg,
        cluster_eps_nm=args.cluster_eps,
        cluster_min_samples=args.min_samples
    )

