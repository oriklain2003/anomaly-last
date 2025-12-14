"""
Unified data loader for learning from research.db and feedback.db.

Loads normal (non-anomaly) flights from:
- research.db: normal_tracks table
- feedback.db + training_dataset.db: flights marked as non-anomaly by users
- last.db: flight_tracks table
- training_ops/consolidated.db: flight_tracks table
"""

from __future__ import annotations

import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import FlightTrack, TrackPoint
from core.config import TRAIN_NORTH, TRAIN_SOUTH, TRAIN_EAST, TRAIN_WEST, load_rule_config
from core.geodesy import haversine_nm

logger = logging.getLogger(__name__)

# Callsign prefixes to filter out (Israeli military/government)
FILTERED_CALLSIGN_PREFIXES = ("4XA", "4XB", "4XC", "4XD")

# Low altitude threshold for O/D detection (ft)
LOW_ALTITUDE_THRESHOLD_FT = 5000.0  # Increased to catch more landing/takeoff segments

# Airport proximity threshold for O/D detection (nm)
AIRPORT_PROXIMITY_NM = 15.0

# Speed threshold for landing/takeoff detection (kts)
LANDING_TAKEOFF_SPEED_KTS = 200.0

# Number of points to check for O/D detection
OD_SEARCH_POINTS = 20  # Increased from 5 to handle flights entering bbox late


@dataclass
class BoundingBox:
    """Geographic bounding box for filtering flights."""
    north: float
    south: float
    east: float
    west: float
    
    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is within the bounding box."""
        return (self.south <= lat <= self.north) and (self.west <= lon <= self.east)


# Default bounding box from config
DEFAULT_BBOX = BoundingBox(
    north=TRAIN_NORTH,
    south=TRAIN_SOUTH,
    east=TRAIN_EAST,
    west=TRAIN_WEST
)


@dataclass
class Airport:
    """Airport definition for O/D detection."""
    code: str
    name: str
    lat: float
    lon: float
    elevation_ft: Optional[float] = None


def _load_airports() -> List[Airport]:
    """Load airports from rule_config.json."""
    config = load_rule_config()
    airports = []
    for entry in config.get("airports", []):
        airports.append(Airport(
            code=entry["code"],
            name=entry["name"],
            lat=entry["lat"],
            lon=entry["lon"],
            elevation_ft=entry.get("elevation_ft")
        ))
    return airports


AIRPORTS = _load_airports()


class FlightDataLoader:
    """
    Unified loader for multiple data sources (non-anomaly flights only).
    
    Features:
    - Filters out flights with callsigns starting with 4XA, 4XB, 4XC, 4XD
    - Only keeps track points within the bounding box
    - Determines O/D airports based on first/last point altitude and proximity
    """
    
    def __init__(
        self,
        research_db: Optional[Path] = None,
        feedback_db: Optional[Path] = None,
        training_db: Optional[Path] = None,
        last_db: Optional[Path] = None,
        consolidated_db: Optional[Path] = None,
        bbox: Optional[BoundingBox] = None
    ):
        """
        Initialize the data loader.
        
        Args:
            research_db: Path to realtime/research.db (normal_tracks table)
            feedback_db: Path to feedback.db (user_feedback table)
            training_db: Path to training_dataset.db (flight_tracks table)
            last_db: Path to last.db (flight_tracks table)
            consolidated_db: Path to consolidated.db (flight_tracks table)
            bbox: Bounding box for filtering (defaults to Levant region)
        """
        self.research_db = research_db or Path("realtime/research.db")
        self.feedback_db = feedback_db or Path("training_ops/feedback.db")
        self.training_db = training_db or Path("training_ops/training_dataset.db")
        self.last_db = last_db or Path("last.db")
        self.consolidated_db = consolidated_db or Path("training_ops/consolidated.db")
        self.bbox = bbox or DEFAULT_BBOX
        self._seen_flight_ids: Set[str] = set()
    
    def _row_to_point(self, row: tuple, flight_id: str) -> TrackPoint:
        """Convert a database row to TrackPoint."""
        # Expect: flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source
        if len(row) == 11:
            fid, ts, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source = row
        elif len(row) == 10:
            # Without flight_id
            ts, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source = row
            fid = flight_id
        else:
            # Handle variations
            fid = flight_id
            ts, lat, lon, alt = row[0], row[1], row[2], row[3]
            gspeed = row[4] if len(row) > 4 else None
            vspeed = row[5] if len(row) > 5 else None
            track = row[6] if len(row) > 6 else None
            squawk = row[7] if len(row) > 7 else None
            callsign = row[8] if len(row) > 8 else None
            source = row[9] if len(row) > 9 else None
            
        return TrackPoint(
            flight_id=fid or flight_id,
            timestamp=int(ts),
            lat=float(lat),
            lon=float(lon),
            alt=float(alt or 0),
            gspeed=float(gspeed) if gspeed is not None else None,
            vspeed=float(vspeed) if vspeed is not None else None,
            track=float(track) if track is not None else None,
            squawk=str(squawk) if squawk else None,
            callsign=callsign,
            source=source
        )
    
    def _filter_points_to_bbox(self, points: List[TrackPoint]) -> List[TrackPoint]:
        """Filter points to only those within the bounding box."""
        return [p for p in points if self.bbox.contains(p.lat, p.lon)]
    
    def _is_filtered_callsign(self, callsign: Optional[str]) -> bool:
        """Check if callsign should be filtered out."""
        if not callsign:
            return False
        cs_upper = callsign.strip().upper()
        return any(cs_upper.startswith(prefix) for prefix in FILTERED_CALLSIGN_PREFIXES)
    
    def _build_callsign_filter_sql(self) -> str:
        """Build SQL WHERE clause to filter out unwanted callsigns."""
        conditions = []
        for prefix in FILTERED_CALLSIGN_PREFIXES:
            conditions.append(f"callsign NOT LIKE '{prefix}%'")
        return " AND ".join(conditions)
    
    def _iter_from_table(
        self,
        db_path: Path,
        table_name: str,
        db_label: str
    ) -> Iterator[FlightTrack]:
        """
        Generic iterator for a database table.
        
        Fetches flights with points in the bounding box, filters callsigns,
        and only keeps points within the bbox.
        """
        if not db_path.exists():
            logger.warning(f"{db_label} not found: {db_path}")
            return
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Build query to get flight_ids that have at least one point in bbox
            # and don't have filtered callsigns
            callsign_filter = self._build_callsign_filter_sql()
            
            # First, get all flight_ids with points in the bounding box
            query = f"""
                SELECT DISTINCT flight_id 
                FROM {table_name}
                WHERE lat BETWEEN ? AND ?
                  AND lon BETWEEN ? AND ?
                  AND ({callsign_filter} OR callsign IS NULL OR callsign = '')
            """
            
            cursor.execute(query, (
                self.bbox.south, self.bbox.north,
                self.bbox.west, self.bbox.east
            ))
            flight_ids = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"Found {len(flight_ids)} flights in {db_label} {table_name} within bbox")
            
            processed = 0
            for flight_id in flight_ids:
                if flight_id in self._seen_flight_ids:
                    continue
                
                # Fetch only points within the bounding box
                cursor.execute(f"""
                    SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source
                    FROM {table_name}
                    WHERE flight_id = ?
                      AND lat BETWEEN ? AND ?
                      AND lon BETWEEN ? AND ?
                    ORDER BY timestamp ASC
                """, (flight_id, self.bbox.south, self.bbox.north, self.bbox.west, self.bbox.east))
                
                rows = cursor.fetchall()
                if not rows:
                    continue
                
                # Check callsign from first row
                first_row = rows[0]
                callsign = first_row[9] if len(first_row) > 9 else None
                if self._is_filtered_callsign(callsign):
                    continue
                
                points = [self._row_to_point(row, flight_id) for row in rows]
                
                if points:
                    self._seen_flight_ids.add(flight_id)
                    processed += 1
                    yield FlightTrack(flight_id=flight_id, points=points)
                    
                    if processed % 500 == 0:
                        logger.info(f"Processed {processed} flights from {db_label}...")
            
            conn.close()
            logger.info(f"Loaded {processed} flights from {db_label}")
            
        except Exception as e:
            logger.error(f"Error reading {db_label}: {e}")
    
    def _iter_research_normal(self) -> Iterator[FlightTrack]:
        """Iterate over normal flights from research.db."""
        yield from self._iter_from_table(
            self.research_db, "normal_tracks", "research.db"
        )
    
    def _iter_consolidated_normal(self) -> Iterator[FlightTrack]:
        """Iterate over normal flights from consolidated.db."""
        yield from self._iter_from_table(
            self.consolidated_db, "flight_tracks", "consolidated.db"
        )
    
    def _iter_last_db_normal(self) -> Iterator[FlightTrack]:
        """Iterate over normal flights from last.db."""
        yield from self._iter_from_table(
            self.last_db, "flight_tracks", "last.db"
        )
    
    def _get_non_anomaly_flight_ids(self) -> Set[str]:
        """Get flight IDs marked as non-anomaly in feedback.db."""
        if not self.feedback_db.exists():
            logger.warning(f"Feedback DB not found: {self.feedback_db}")
            return set()
        
        try:
            conn = sqlite3.connect(str(self.feedback_db))
            cursor = conn.cursor()
            
            # user_label = 0 means non-anomaly
            cursor.execute("SELECT DISTINCT flight_id FROM user_feedback WHERE user_label = 0")
            flight_ids = {row[0] for row in cursor.fetchall()}
            
            conn.close()
            logger.info(f"Found {len(flight_ids)} non-anomaly flights in feedback.db")
            return flight_ids
        except Exception as e:
            logger.error(f"Error reading feedback.db: {e}")
            return set()
    
    def _iter_feedback_normal(self) -> Iterator[FlightTrack]:
        """Iterate over non-anomaly flights from feedback/training DBs."""
        non_anomaly_ids = self._get_non_anomaly_flight_ids()
        
        if not non_anomaly_ids:
            return
        
        if not self.training_db.exists():
            logger.warning(f"Training DB not found: {self.training_db}")
            return
        
        try:
            conn = sqlite3.connect(str(self.training_db))
            cursor = conn.cursor()
            
            for flight_id in non_anomaly_ids:
                if flight_id in self._seen_flight_ids:
                    continue
                
                # Fetch points within bounding box
                cursor.execute("""
                    SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source
                    FROM flight_tracks
                    WHERE flight_id = ?
                      AND lat BETWEEN ? AND ?
                      AND lon BETWEEN ? AND ?
                    ORDER BY timestamp ASC
                """, (flight_id, self.bbox.south, self.bbox.north, self.bbox.west, self.bbox.east))
                
                rows = cursor.fetchall()
                if not rows:
                    continue
                
                # Check callsign
                first_row = rows[0]
                callsign = first_row[9] if len(first_row) > 9 else None
                if self._is_filtered_callsign(callsign):
                    continue
                
                points = [self._row_to_point(row, flight_id) for row in rows]
                
                if points:
                    self._seen_flight_ids.add(flight_id)
                    yield FlightTrack(flight_id=flight_id, points=points)
            
            conn.close()
        except Exception as e:
            logger.error(f"Error reading training_dataset.db: {e}")
    
    def iter_normal_flights(self, min_points: int = 20) -> Iterator[FlightTrack]:
        """
        Iterate over all normal flights from all data sources.
        
        Args:
            min_points: Minimum number of points required for a valid flight
                       (default 20 to filter out garbage flights with few points)
            
        Yields:
            FlightTrack objects for each normal flight within the bounding box
        """
        self._seen_flight_ids.clear()
        count = 0
        
        # Priority order: consolidated.db (most curated) -> research.db -> last.db -> feedback
        
        # 1. Consolidated DB (has both normal and processed data)
        for flight in self._iter_consolidated_normal():
            if len(flight.points) >= min_points:
                count += 1
                yield flight
        
        # 2. Research DB (realtime/research.db - normal_tracks)
        for flight in self._iter_research_normal():
            if len(flight.points) >= min_points:
                count += 1
                yield flight
        
        # 3. Last DB (last.db - flight_tracks)
        for flight in self._iter_last_db_normal():
            if len(flight.points) >= min_points:
                count += 1
                yield flight
        
        # 4. Feedback non-anomaly flights
        for flight in self._iter_feedback_normal():
            if len(flight.points) >= min_points:
                count += 1
                yield flight
        
        logger.info(f"Total normal flights loaded: {count}")
    
    def get_airport_for_point(
        self,
        lat: float,
        lon: float,
        threshold_nm: float = AIRPORT_PROXIMITY_NM
    ) -> Optional[str]:
        """
        Get the nearest airport code if within threshold distance.
        
        Args:
            lat: Latitude
            lon: Longitude
            threshold_nm: Maximum distance in nautical miles
            
        Returns:
            Airport code if within threshold, None otherwise
        """
        best_code = None
        best_dist = float('inf')
        
        for airport in AIRPORTS:
            dist = haversine_nm(lat, lon, airport.lat, airport.lon)
            if dist < best_dist and dist <= threshold_nm:
                best_dist = dist
                best_code = airport.code
        
        return best_code
    
    def get_origin_destination(
        self,
        flight: FlightTrack,
        threshold_nm: float = AIRPORT_PROXIMITY_NM
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine origin and destination airports for a flight.
        
        Logic:
        1. Sort points by timestamp
        2. For origin: Check first OD_SEARCH_POINTS points for low altitude (< 5000ft)
           AND low speed (< 200kts) near an airport (< 15nm)
        3. For destination: Check last OD_SEARCH_POINTS points similarly
        4. Fallback: Simple proximity check on first/last point
        
        Args:
            flight: The flight track
            threshold_nm: Max distance from airport to consider it as O/D
            
        Returns:
            Tuple of (origin_code, destination_code), either may be None
        """
        points = flight.sorted_points()
        if len(points) < 2:
            return None, None
        
        origin = None
        destination = None
        
        # Check first OD_SEARCH_POINTS points for origin (increased from 5)
        search_range = min(OD_SEARCH_POINTS, len(points))
        for i in range(search_range):
            p = points[i]
            
            # Check if at low altitude AND low speed (landing/takeoff segment)
            is_low_alt = p.alt <= LOW_ALTITUDE_THRESHOLD_FT
            is_low_speed = (p.gspeed or 999) < LANDING_TAKEOFF_SPEED_KTS
            
            if is_low_alt and is_low_speed:
                airport = self.get_airport_for_point(p.lat, p.lon, threshold_nm)
                if airport:
                    # Verify altitude is reasonable for this airport
                    airport_obj = next((a for a in AIRPORTS if a.code == airport), None)
                    if airport_obj:
                        airport_elev = airport_obj.elevation_ft or 0
                        # Allow up to altitude threshold above airport elevation
                        if p.alt <= airport_elev + LOW_ALTITUDE_THRESHOLD_FT:
                            origin = airport
                            break
            elif is_low_alt:
                # Low altitude but no speed data - still check airport
                airport = self.get_airport_for_point(p.lat, p.lon, threshold_nm)
                if airport:
                    airport_obj = next((a for a in AIRPORTS if a.code == airport), None)
                    if airport_obj:
                        airport_elev = airport_obj.elevation_ft or 0
                        if p.alt <= airport_elev + LOW_ALTITUDE_THRESHOLD_FT:
                            origin = airport
                            break
        
        # Fallback: If no origin found, check first point proximity only
        if not origin:
            first_point = points[0]
            origin = self.get_airport_for_point(first_point.lat, first_point.lon, threshold_nm)
        
        # Check last OD_SEARCH_POINTS points for destination
        search_start = max(0, len(points) - OD_SEARCH_POINTS)
        for i in range(len(points) - 1, search_start - 1, -1):
            p = points[i]
            
            # Check if at low altitude AND low speed (landing segment)
            is_low_alt = p.alt <= LOW_ALTITUDE_THRESHOLD_FT
            is_low_speed = (p.gspeed or 999) < LANDING_TAKEOFF_SPEED_KTS
            
            if is_low_alt and is_low_speed:
                airport = self.get_airport_for_point(p.lat, p.lon, threshold_nm)
                if airport:
                    # Verify altitude is reasonable for this airport
                    airport_obj = next((a for a in AIRPORTS if a.code == airport), None)
                    if airport_obj:
                        airport_elev = airport_obj.elevation_ft or 0
                        if p.alt <= airport_elev + LOW_ALTITUDE_THRESHOLD_FT:
                            destination = airport
                            break
            elif is_low_alt:
                # Low altitude but no speed data - still check airport
                airport = self.get_airport_for_point(p.lat, p.lon, threshold_nm)
                if airport:
                    airport_obj = next((a for a in AIRPORTS if a.code == airport), None)
                    if airport_obj:
                        airport_elev = airport_obj.elevation_ft or 0
                        if p.alt <= airport_elev + LOW_ALTITUDE_THRESHOLD_FT:
                            destination = airport
                            break
        
        # Fallback: If no destination found, check last point proximity only
        if not destination:
            last_point = points[-1]
            destination = self.get_airport_for_point(last_point.lat, last_point.lon, threshold_nm)
        
        return origin, destination
