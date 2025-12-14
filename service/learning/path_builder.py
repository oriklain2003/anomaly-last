"""
Path builder - learns flight paths clustered by origin/destination airport pairs.

Approach:
1. Group flights by (origin_airport, destination_airport) pair
2. For each O/D pair with enough flights:
   - Resample trajectories to fixed points
   - Run HDBSCAN clustering
   - Compute DBA centroid per cluster
3. Output: rules/learned_paths.json
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import FlightTrack
from learning.data_loader import FlightDataLoader, DEFAULT_BBOX
from learning.utils import (
    cluster_trajectories_dbscan,
    compute_dba_centroid,
    compute_cluster_width,
    resample_flight,
)

logger = logging.getLogger(__name__)


@dataclass
class LearnedPath:
    """Represents a learned flight path."""
    id: str
    origin: Optional[str]
    destination: Optional[str]
    centerline: List[Dict[str, float]]  # [{"lat": ..., "lon": ..., "alt": ...}, ...]
    width_nm: float
    member_count: int
    member_flights: List[str]


class PathBuilder:
    """
    Builds a path library by clustering flights by origin/destination pairs.
    """
    
    def __init__(
        self,
        data_loader: FlightDataLoader,
        num_samples: int = 40,  # Reduced from 80 for faster processing
        min_flights_per_od: int = 5,
        min_cluster_size: int = 3,
        min_samples: int = 3,
        cluster_eps_nm: float = 5.0,  # DBSCAN epsilon in nm
        od_threshold_nm: float = 10.0,
        require_both_od: bool = True  # Require both origin AND destination
    ):
        """
        Initialize the path builder.
        
        Args:
            data_loader: Data loader instance
            num_samples: Number of resampling points per trajectory
            min_flights_per_od: Minimum flights to consider an O/D pair
            min_cluster_size: DBSCAN min cluster size
            min_samples: DBSCAN min_samples for core point
            cluster_eps_nm: DBSCAN epsilon in nautical miles
            od_threshold_nm: Distance threshold for O/D airport detection
            require_both_od: If True, skip flights without both origin AND destination
        """
        self.data_loader = data_loader
        self.num_samples = num_samples
        self.min_flights_per_od = min_flights_per_od
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_eps_nm = cluster_eps_nm
        self.od_threshold_nm = od_threshold_nm
        self.require_both_od = require_both_od
    
    def _group_by_od(
        self,
        flights: List[FlightTrack]
    ) -> Dict[Tuple[Optional[str], Optional[str]], List[FlightTrack]]:
        """Group flights by origin/destination airport pair."""
        groups: Dict[Tuple[Optional[str], Optional[str]], List[FlightTrack]] = defaultdict(list)
        skipped_incomplete = 0
        
        for flight in flights:
            origin, dest = self.data_loader.get_origin_destination(
                flight, 
                threshold_nm=self.od_threshold_nm
            )
            
            # Filter invalid O/D pairs if require_both_od is True
            if self.require_both_od:
                if origin is None or dest is None:
                    skipped_incomplete += 1
                    continue
            
            groups[(origin, dest)].append(flight)
        
        if skipped_incomplete > 0:
            logger.info(f"Skipped {skipped_incomplete} flights without complete O/D pair")
        
        return groups
    
    def _build_paths_for_od(
        self,
        origin: Optional[str],
        dest: Optional[str],
        flights: List[FlightTrack]
    ) -> List[LearnedPath]:
        """Build paths for a single O/D pair."""
        paths_out = []
        
        # Resample all flights
        resampled = []
        flight_ids = []
        
        for flight in flights:
            arr = resample_flight(flight.points, num_samples=self.num_samples)
            if arr is not None:
                resampled.append(arr)
                flight_ids.append(flight.flight_id)
        
        if len(resampled) < self.min_cluster_size:
            return paths_out
        
        # Cluster using DBSCAN (much faster than HDBSCAN)
        labels = cluster_trajectories_dbscan(
            resampled,
            eps_nm=self.cluster_eps_nm,
            min_samples=self.min_samples
        )
        
        # Build path for each cluster
        clusters: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            if label >= 0:  # Skip noise
                clusters[label].append(idx)
        
        for cluster_id, indices in clusters.items():
            member_paths = [resampled[i] for i in indices]
            member_ids = [flight_ids[i] for i in indices]
            
            # Compute centroid
            centroid = compute_dba_centroid(member_paths)
            
            # Compute width
            width = compute_cluster_width(member_paths, centroid)
            
            # Create path ID
            origin_str = origin or "UNK"
            dest_str = dest or "UNK"
            path_id = f"{origin_str}_{dest_str}_{cluster_id}"
            
            # Convert centroid to JSON-friendly format
            centerline = []
            for lat, lon, alt in centroid.tolist():
                centerline.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "alt": float(alt)
                })
            
            paths_out.append(LearnedPath(
                id=path_id,
                origin=origin,
                destination=dest,
                centerline=centerline,
                width_nm=width,
                member_count=len(member_ids),
                member_flights=member_ids
            ))
        
        return paths_out
    
    def build(self, progress_callback=None) -> List[LearnedPath]:
        """
        Build the path library from all normal flights.
        
        Args:
            progress_callback: Optional callback(message: str) for progress
            
        Returns:
            List of learned paths
        """
        def log_progress(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)
        
        log_progress("Loading normal flights...")
        flights = list(self.data_loader.iter_normal_flights(min_points=20))
        log_progress(f"Loaded {len(flights)} flights")
        
        log_progress("Grouping flights by origin/destination...")
        od_groups = self._group_by_od(flights)
        log_progress(f"Found {len(od_groups)} O/D pairs")
        
        all_paths = []
        processed = 0
        
        for (origin, dest), group_flights in od_groups.items():
            if len(group_flights) < self.min_flights_per_od:
                continue
            
            log_progress(f"Processing {origin or 'UNK'} -> {dest or 'UNK'} ({len(group_flights)} flights)")
            
            paths = self._build_paths_for_od(origin, dest, group_flights)
            all_paths.extend(paths)
            processed += 1
        
        log_progress(f"Built {len(all_paths)} paths from {processed} O/D pairs")
        return all_paths
    
    def save(self, paths: List[LearnedPath], output_path: Path) -> None:
        """
        Save the path library to JSON.
        
        Args:
            paths: List of learned paths
            output_path: Output file path
        """
        output = {
            "version": "3.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_paths": len(paths),
            "paths": [
                {
                    "id": p.id,
                    "origin": p.origin,
                    "destination": p.destination,
                    "centerline": p.centerline,
                    "width_nm": round(p.width_nm, 2),
                    "member_count": int(p.member_count),
                    "member_flights": p.member_flights[:10]  # Limit for file size
                }
                for p in paths
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved path library to {output_path}")


def run_path_builder(
    research_db: Path,
    feedback_db: Optional[Path] = None,
    training_db: Optional[Path] = None,
    last_db: Optional[Path] = None,
    output_path: Path = Path("rules/learned_paths.json"),
    **kwargs
) -> List[LearnedPath]:
    """
    Run the path builder pipeline.
    
    Args:
        research_db: Path to research.db
        feedback_db: Path to feedback.db (optional)
        training_db: Path to training_dataset.db (optional)
        last_db: Path to last.db (optional, primary data source)
        output_path: Output file path
        **kwargs: Additional arguments for PathBuilder
        
    Returns:
        List of learned paths
    """
    loader = FlightDataLoader(
        research_db=research_db,
        feedback_db=feedback_db,
        training_db=training_db,
        last_db=last_db
    )
    
    builder = PathBuilder(data_loader=loader, **kwargs)
    paths = builder.build()
    builder.save(paths, output_path)
    
    return paths


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Build path library from flight data")
    parser.add_argument("--research-db", type=Path, default=Path("research.db"))
    parser.add_argument("--feedback-db", type=Path, default=Path("training_ops/feedback.db"))
    parser.add_argument("--training-db", type=Path, default=Path("training_ops/training_dataset.db"))
    parser.add_argument("--output", type=Path, default=Path("rules/learned_paths.json"))
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--min-flights", type=int, default=5)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    
    args = parser.parse_args()
    
    run_path_builder(
        research_db=args.research_db,
        feedback_db=args.feedback_db,
        training_db=args.training_db,
        output_path=args.output,
        num_samples=args.num_samples,
        min_flights_per_od=args.min_flights,
        min_cluster_size=args.min_cluster_size
    )

