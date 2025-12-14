"""
Learning module for flight behavior pattern extraction.

This module provides tools for learning normal flight patterns from historical data:
- Path learning: Cluster flights by origin/destination airport pairs
- Turn learning: Identify known turn zones (180-300 degree turns)
- SID/STAR learning: Extract standard departure/arrival procedures

Usage:
    python -m learning.run_all --research-db research.db --output-dir rules/
"""

from .data_loader import FlightDataLoader
from .utils import (
    is_in_bbox,
    cluster_trajectories_hdbscan,
    cluster_points_dbscan,
    compute_dba_centroid,
    detect_turns,
)
from .path_builder import PathBuilder
from .turn_builder import TurnBuilder
from .sid_star_builder import SIDSTARBuilder

__all__ = [
    "FlightDataLoader",
    "is_in_bbox",
    "cluster_trajectories_hdbscan",
    "cluster_points_dbscan",
    "compute_dba_centroid",
    "detect_turns",
    "PathBuilder",
    "TurnBuilder",
    "SIDSTARBuilder",
]

