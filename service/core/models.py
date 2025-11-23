from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrackPoint:
    flight_id: str
    timestamp: int
    lat: float
    lon: float
    alt: float
    gspeed: Optional[float] = None
    vspeed: Optional[float] = None
    track: Optional[float] = None
    squawk: Optional[str] = None
    callsign: Optional[str] = None
    source: Optional[str] = None


@dataclass
class FlightTrack:
    flight_id: str
    points: List[TrackPoint] = field(default_factory=list)

    def sorted_points(self) -> List[TrackPoint]:
        return sorted(self.points, key=lambda p: p.timestamp)


@dataclass
class FlightMetadata:
    planned_destination: Optional[str] = None
    planned_route: Optional[List[List[float]]] = None  # [[lat, lon], ...]


@dataclass
class RuleContext:
    track: FlightTrack
    metadata: Optional[FlightMetadata]
    repository: "FlightRepository"


@dataclass
class RuleResult:
    rule_id: int
    matched: bool
    summary: str
    details: Dict[str, object]

