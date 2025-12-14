"""
SID/STAR builder - learns standard departure and arrival procedures.

Approach:
- SID: First 30nm of departing flights (climbing from airport)
- STAR: Last 40nm of arriving flights (descending to airport)
- Group by airport, cluster with HDBSCAN, compute centroids
- Output: rules/learned_sid.json, rules/learned_star.json
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

from core.models import FlightTrack, TrackPoint
from core.geodesy import haversine_nm
from learning.data_loader import FlightDataLoader, AIRPORTS
from learning.utils import (
    cluster_trajectories_dbscan,
    compute_dba_centroid,
    compute_cluster_width,
    resample_flight,
    extract_segment,
)

logger = logging.getLogger(__name__)


@dataclass
class LearnedProcedure:
    """Represents a learned SID or STAR procedure."""
    id: str
    airport: str
    procedure_type: str  # "SID" or "STAR"
    centerline: List[Dict[str, float]]  # [{"lat": ..., "lon": ..., "alt": ...}, ...]
    width_nm: float
    member_count: int
    member_flights: List[str]


class SIDSTARBuilder:
    """
    Builds SID and STAR libraries by clustering departure/arrival tracks.
    """
    
    def __init__(
        self,
        data_loader: FlightDataLoader,
        sid_distance_nm: float = 30.0,
        star_distance_nm: float = 40.0,
        airport_threshold_nm: float = 5.0,
        num_samples: int = 40,
        min_flights_per_airport: int = 5,
        min_cluster_size: int = 3,
        min_samples: int = 3,
        cluster_eps_nm: float = 3.0,  # DBSCAN epsilon
        min_climb_rate_fpm: float = 200.0,
        min_descent_rate_fpm: float = -200.0
    ):
        """
        Initialize the SID/STAR builder.
        
        Args:
            data_loader: Data loader instance
            sid_distance_nm: Length of SID segment to extract
            star_distance_nm: Length of STAR segment to extract
            airport_threshold_nm: Max distance from airport to consider departure/arrival
            num_samples: Number of resampling points per segment
            min_flights_per_airport: Minimum flights to cluster for an airport
            min_cluster_size: DBSCAN min cluster size
            min_samples: DBSCAN min_samples for core point
            cluster_eps_nm: DBSCAN epsilon in nautical miles
            min_climb_rate_fpm: Minimum climb rate for departure detection
            min_descent_rate_fpm: Maximum (most negative) descent rate for arrival
        """
        self.data_loader = data_loader
        self.sid_distance_nm = sid_distance_nm
        self.star_distance_nm = star_distance_nm
        self.airport_threshold_nm = airport_threshold_nm
        self.num_samples = num_samples
        self.min_flights_per_airport = min_flights_per_airport
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_eps_nm = cluster_eps_nm
        self.min_climb_rate_fpm = min_climb_rate_fpm
        self.min_descent_rate_fpm = min_descent_rate_fpm
    
    def _is_departing(self, points: List[TrackPoint]) -> bool:
        """Check if flight is departing (climbing from start)."""
        if len(points) < 5:
            return False
        
        # Check first few points for climb
        sorted_pts = sorted(points, key=lambda p: p.timestamp)
        early_pts = sorted_pts[:min(10, len(sorted_pts))]
        
        # Check average vertical speed
        vs_sum = 0
        vs_count = 0
        for p in early_pts:
            if p.vspeed is not None:
                vs_sum += p.vspeed
                vs_count += 1
        
        if vs_count > 0:
            avg_vs = vs_sum / vs_count
            return avg_vs > self.min_climb_rate_fpm
        
        # Fallback: check altitude trend
        if len(early_pts) >= 2:
            alt_diff = early_pts[-1].alt - early_pts[0].alt
            time_diff = early_pts[-1].timestamp - early_pts[0].timestamp
            if time_diff > 0:
                rate = (alt_diff / time_diff) * 60  # ft per minute
                return rate > self.min_climb_rate_fpm
        
        return False
    
    def _is_arriving(self, points: List[TrackPoint]) -> bool:
        """Check if flight is arriving (descending to end)."""
        if len(points) < 5:
            return False
        
        # Check last few points for descent
        sorted_pts = sorted(points, key=lambda p: p.timestamp)
        late_pts = sorted_pts[-min(10, len(sorted_pts)):]
        
        # Check average vertical speed
        vs_sum = 0
        vs_count = 0
        for p in late_pts:
            if p.vspeed is not None:
                vs_sum += p.vspeed
                vs_count += 1
        
        if vs_count > 0:
            avg_vs = vs_sum / vs_count
            return avg_vs < self.min_descent_rate_fpm
        
        # Fallback: check altitude trend
        if len(late_pts) >= 2:
            alt_diff = late_pts[-1].alt - late_pts[0].alt
            time_diff = late_pts[-1].timestamp - late_pts[0].timestamp
            if time_diff > 0:
                rate = (alt_diff / time_diff) * 60  # ft per minute
                return rate < self.min_descent_rate_fpm
        
        return False
    
    def _get_departure_airport(self, flight: FlightTrack) -> Optional[str]:
        """Get departure airport if flight starts near one and is climbing."""
        points = flight.sorted_points()
        if not points:
            return None
        
        first_point = points[0]
        airport = self.data_loader.get_airport_for_point(
            first_point.lat, first_point.lon,
            threshold_nm=self.airport_threshold_nm
        )
        
        if airport and self._is_departing(points):
            return airport
        return None
    
    def _get_arrival_airport(self, flight: FlightTrack) -> Optional[str]:
        """Get arrival airport if flight ends near one and is descending."""
        points = flight.sorted_points()
        if not points:
            return None
        
        last_point = points[-1]
        airport = self.data_loader.get_airport_for_point(
            last_point.lat, last_point.lon,
            threshold_nm=self.airport_threshold_nm
        )
        
        if airport and self._is_arriving(points):
            return airport
        return None
    
    def _build_procedures(
        self,
        airport: str,
        flights: List[FlightTrack],
        is_sid: bool
    ) -> List[LearnedProcedure]:
        """Build SID or STAR procedures for an airport."""
        procedures = []
        proc_type = "SID" if is_sid else "STAR"
        distance = self.sid_distance_nm if is_sid else self.star_distance_nm
        
        # Extract segments
        segments = []
        flight_ids = []
        
        for flight in flights:
            points = flight.sorted_points()
            segment = extract_segment(
                points,
                from_start=is_sid,
                distance_nm=distance
            )
            
            if len(segment) >= 5:
                arr = resample_flight(segment, num_samples=self.num_samples)
                if arr is not None:
                    segments.append(arr)
                    flight_ids.append(flight.flight_id)
        
        if len(segments) < self.min_cluster_size:
            return procedures
        
        # Cluster using DBSCAN (faster than HDBSCAN)
        labels = cluster_trajectories_dbscan(
            segments,
            eps_nm=self.cluster_eps_nm,
            min_samples=self.min_samples
        )
        
        # Build procedure for each cluster
        clusters: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            if label >= 0:
                clusters[label].append(idx)
        
        for cluster_id, indices in clusters.items():
            member_segments = [segments[i] for i in indices]
            member_ids = [flight_ids[i] for i in indices]
            
            # Compute centroid
            centroid = compute_dba_centroid(member_segments)
            
            # Compute width
            width = compute_cluster_width(member_segments, centroid)
            
            # Create procedure ID
            proc_id = f"{airport}_{proc_type}_{cluster_id}"
            
            # Convert centroid to JSON format
            centerline = []
            for lat, lon, alt in centroid.tolist():
                centerline.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "alt": float(alt)
                })
            
            procedures.append(LearnedProcedure(
                id=proc_id,
                airport=airport,
                procedure_type=proc_type,
                centerline=centerline,
                width_nm=width,
                member_count=len(member_ids),
                member_flights=member_ids
            ))
        
        return procedures
    
    def build(self, progress_callback=None) -> Tuple[List[LearnedProcedure], List[LearnedProcedure]]:
        """
        Build SID and STAR libraries from all normal flights.
        
        Args:
            progress_callback: Optional callback(message: str) for progress
            
        Returns:
            Tuple of (SID procedures, STAR procedures)
        """
        def log_progress(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)
        
        log_progress("Loading normal flights for SID/STAR detection...")
        flights = list(self.data_loader.iter_normal_flights(min_points=20))
        log_progress(f"Loaded {len(flights)} flights")
        
        # Group by departure airport
        departures: Dict[str, List[FlightTrack]] = defaultdict(list)
        arrivals: Dict[str, List[FlightTrack]] = defaultdict(list)
        
        log_progress("Identifying departures and arrivals...")
        for flight in flights:
            dep_airport = self._get_departure_airport(flight)
            if dep_airport:
                departures[dep_airport].append(flight)
            
            arr_airport = self._get_arrival_airport(flight)
            if arr_airport:
                arrivals[arr_airport].append(flight)
        
        log_progress(f"Found departures from {len(departures)} airports")
        log_progress(f"Found arrivals to {len(arrivals)} airports")
        
        # Build SIDs
        all_sids = []
        for airport, flights in departures.items():
            if len(flights) >= self.min_flights_per_airport:
                log_progress(f"Building SIDs for {airport} ({len(flights)} flights)")
                sids = self._build_procedures(airport, flights, is_sid=True)
                all_sids.extend(sids)
        
        log_progress(f"Built {len(all_sids)} SID procedures")
        
        # Build STARs
        all_stars = []
        for airport, flights in arrivals.items():
            if len(flights) >= self.min_flights_per_airport:
                log_progress(f"Building STARs for {airport} ({len(flights)} flights)")
                stars = self._build_procedures(airport, flights, is_sid=False)
                all_stars.extend(stars)
        
        log_progress(f"Built {len(all_stars)} STAR procedures")
        
        return all_sids, all_stars
    
    def save(
        self,
        sids: List[LearnedProcedure],
        stars: List[LearnedProcedure],
        sid_output_path: Path,
        star_output_path: Path
    ) -> None:
        """
        Save SID and STAR libraries to JSON.
        
        Args:
            sids: List of SID procedures
            stars: List of STAR procedures
            sid_output_path: Output file for SIDs
            star_output_path: Output file for STARs
        """
        # Save SIDs
        sid_output = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "type": "SID",
            "total_procedures": len(sids),
            "procedures": [
                {
                    "id": p.id,
                    "airport": p.airport,
                    "type": p.procedure_type,
                    "centerline": p.centerline,
                    "width_nm": round(p.width_nm, 2),
                    "member_count": int(p.member_count),
                    "member_flights": p.member_flights[:10]
                }
                for p in sids
            ]
        }
        
        sid_output_path.parent.mkdir(parents=True, exist_ok=True)
        with sid_output_path.open("w", encoding="utf-8") as f:
            json.dump(sid_output, f, indent=2)
        logger.info(f"Saved SID library to {sid_output_path}")
        
        # Save STARs
        star_output = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "type": "STAR",
            "total_procedures": len(stars),
            "procedures": [
                {
                    "id": p.id,
                    "airport": p.airport,
                    "type": p.procedure_type,
                    "centerline": p.centerline,
                    "width_nm": round(p.width_nm, 2),
                    "member_count": int(p.member_count),
                    "member_flights": p.member_flights[:10]
                }
                for p in stars
            ]
        }
        
        star_output_path.parent.mkdir(parents=True, exist_ok=True)
        with star_output_path.open("w", encoding="utf-8") as f:
            json.dump(star_output, f, indent=2)
        logger.info(f"Saved STAR library to {star_output_path}")


def run_sid_star_builder(
    research_db: Path,
    feedback_db: Optional[Path] = None,
    training_db: Optional[Path] = None,
    last_db: Optional[Path] = None,
    sid_output_path: Path = Path("rules/learned_sid.json"),
    star_output_path: Path = Path("rules/learned_star.json"),
    **kwargs
) -> Tuple[List[LearnedProcedure], List[LearnedProcedure]]:
    """
    Run the SID/STAR builder pipeline.
    
    Args:
        research_db: Path to research.db
        feedback_db: Path to feedback.db (optional)
        training_db: Path to training_dataset.db (optional)
        last_db: Path to last.db (optional, primary data source)
        sid_output_path: Output file for SIDs
        star_output_path: Output file for STARs
        **kwargs: Additional arguments for SIDSTARBuilder
        
    Returns:
        Tuple of (SID procedures, STAR procedures)
    """
    loader = FlightDataLoader(
        research_db=research_db,
        feedback_db=feedback_db,
        training_db=training_db,
        last_db=last_db
    )
    
    builder = SIDSTARBuilder(data_loader=loader, **kwargs)
    sids, stars = builder.build()
    builder.save(sids, stars, sid_output_path, star_output_path)
    
    return sids, stars


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Build SID/STAR libraries from flight data")
    parser.add_argument("--research-db", type=Path, default=Path("research.db"))
    parser.add_argument("--feedback-db", type=Path, default=Path("training_ops/feedback.db"))
    parser.add_argument("--training-db", type=Path, default=Path("training_ops/training_dataset.db"))
    parser.add_argument("--sid-output", type=Path, default=Path("rules/learned_sid.json"))
    parser.add_argument("--star-output", type=Path, default=Path("rules/learned_star.json"))
    parser.add_argument("--sid-distance", type=float, default=30.0)
    parser.add_argument("--star-distance", type=float, default=40.0)
    parser.add_argument("--min-flights", type=int, default=5)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    
    args = parser.parse_args()
    
    run_sid_star_builder(
        research_db=args.research_db,
        feedback_db=args.feedback_db,
        training_db=args.training_db,
        sid_output_path=args.sid_output,
        star_output_path=args.star_output,
        sid_distance_nm=args.sid_distance,
        star_distance_nm=args.star_distance,
        min_flights_per_airport=args.min_flights,
        min_cluster_size=args.min_cluster_size
    )

