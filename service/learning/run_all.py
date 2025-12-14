"""
Orchestrator script to run all learning builders.

Usage:
    python -m learning.run_all --research-db research.db --output-dir rules/
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure parent directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from learning.path_builder import run_path_builder
from learning.turn_builder import run_turn_builder
from learning.sid_star_builder import run_sid_star_builder

logger = logging.getLogger(__name__)


def run_all(
    research_db: Path,
    feedback_db: Path,
    training_db: Path,
    last_db: Path,
    output_dir: Path,
    skip_paths: bool = False,
    skip_turns: bool = False,
    skip_sid_star: bool = False,
    **kwargs
) -> dict:
    """
    Run all learning builders.
    
    Args:
        research_db: Path to research.db (realtime/research.db)
        feedback_db: Path to feedback.db
        training_db: Path to training_dataset.db
        last_db: Path to last.db (primary data source)
        output_dir: Output directory for JSON files
        skip_paths: Skip path building
        skip_turns: Skip turn building
        skip_sid_star: Skip SID/STAR building
        **kwargs: Additional arguments passed to individual builders
        
    Returns:
        Summary dict with counts from each builder
    """
    start_time = datetime.now()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "start_time": start_time.isoformat(),
        "research_db": str(research_db),
        "feedback_db": str(feedback_db),
        "training_db": str(training_db),
        "last_db": str(last_db),
        "output_dir": str(output_dir),
        "paths": None,
        "turns": None,
        "sids": None,
        "stars": None,
    }
    
    # Build paths
    if not skip_paths:
        logger.info("=" * 60)
        logger.info("BUILDING PATH LIBRARY")
        logger.info("=" * 60)
        try:
            paths = run_path_builder(
                research_db=research_db,
                feedback_db=feedback_db,
                training_db=training_db,
                last_db=last_db,
                output_path=output_dir / "learned_paths.json",
                num_samples=kwargs.get("path_num_samples", 80),
                min_flights_per_od=kwargs.get("path_min_flights", 5),
                min_cluster_size=kwargs.get("path_min_cluster_size", 3),
                require_both_od=kwargs.get("require_both_od", True),
            )
            summary["paths"] = len(paths)
            logger.info(f"Path building complete: {len(paths)} paths")
        except Exception as e:
            logger.error(f"Path building failed: {e}")
            summary["paths"] = f"ERROR: {e}"
    
    # Build turns
    if not skip_turns:
        logger.info("=" * 60)
        logger.info("BUILDING TURN ZONE LIBRARY")
        logger.info("=" * 60)
        try:
            turns = run_turn_builder(
                research_db=research_db,
                feedback_db=feedback_db,
                training_db=training_db,
                last_db=last_db,
                output_path=output_dir / "learned_turns.json",
                min_turn_deg=kwargs.get("turn_min_deg", 180.0),
                max_turn_deg=kwargs.get("turn_max_deg", 300.0),
                cluster_eps_nm=kwargs.get("turn_cluster_eps", 3.0),
                cluster_min_samples=kwargs.get("turn_min_samples", 3),
            )
            summary["turns"] = len(turns)
            logger.info(f"Turn building complete: {len(turns)} zones")
        except Exception as e:
            logger.error(f"Turn building failed: {e}")
            summary["turns"] = f"ERROR: {e}"
    
    # Build SID/STAR
    if not skip_sid_star:
        logger.info("=" * 60)
        logger.info("BUILDING SID/STAR LIBRARIES")
        logger.info("=" * 60)
        try:
            sids, stars = run_sid_star_builder(
                research_db=research_db,
                feedback_db=feedback_db,
                training_db=training_db,
                last_db=last_db,
                sid_output_path=output_dir / "learned_sid.json",
                star_output_path=output_dir / "learned_star.json",
                sid_distance_nm=kwargs.get("sid_distance", 30.0),
                star_distance_nm=kwargs.get("star_distance", 40.0),
                min_flights_per_airport=kwargs.get("sid_star_min_flights", 5),
                min_cluster_size=kwargs.get("sid_star_min_cluster_size", 3),
            )
            summary["sids"] = len(sids)
            summary["stars"] = len(stars)
            logger.info(f"SID/STAR building complete: {len(sids)} SIDs, {len(stars)} STARs")
        except Exception as e:
            logger.error(f"SID/STAR building failed: {e}")
            summary["sids"] = f"ERROR: {e}"
            summary["stars"] = f"ERROR: {e}"
    
    end_time = datetime.now()
    summary["end_time"] = end_time.isoformat()
    summary["duration_seconds"] = (end_time - start_time).total_seconds()
    
    logger.info("=" * 60)
    logger.info("LEARNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {summary['duration_seconds']:.1f} seconds")
    logger.info(f"Paths: {summary['paths']}")
    logger.info(f"Turn Zones: {summary['turns']}")
    logger.info(f"SIDs: {summary['sids']}")
    logger.info(f"STARs: {summary['stars']}")
    
    return summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run all learning builders to generate path, turn, SID, and STAR libraries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Database arguments
    parser.add_argument(
        "--research-db", type=Path, default=Path("realtime/research.db"),
        help="Path to research.db (contains normal_tracks)"
    )
    parser.add_argument(
        "--feedback-db", type=Path, default=Path("training_ops/feedback.db"),
        help="Path to feedback.db"
    )
    parser.add_argument(
        "--training-db", type=Path, default=Path("training_ops/training_dataset.db"),
        help="Path to training_dataset.db"
    )
    parser.add_argument(
        "--last-db", type=Path, default=Path("last.db"),
        help="Path to last.db (primary data source with flight_tracks)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("rules"),
        help="Output directory for JSON files"
    )
    
    # Skip flags
    parser.add_argument("--skip-paths", action="store_true", help="Skip path building")
    parser.add_argument("--skip-turns", action="store_true", help="Skip turn building")
    parser.add_argument("--skip-sid-star", action="store_true", help="Skip SID/STAR building")
    
    # Path builder options
    parser.add_argument("--path-num-samples", type=int, default=80, help="Path resampling points")
    parser.add_argument("--path-min-flights", type=int, default=5, help="Min flights per O/D pair")
    parser.add_argument("--path-min-cluster-size", type=int, default=3, help="Path HDBSCAN min_cluster_size")
    parser.add_argument("--include-partial-od", action="store_true", help="Include flights without clear O/D")
    
    # Turn builder options
    parser.add_argument("--turn-min-deg", type=float, default=180.0, help="Minimum turn angle")
    parser.add_argument("--turn-max-deg", type=float, default=300.0, help="Maximum turn angle")
    parser.add_argument("--turn-cluster-eps", type=float, default=3.0, help="Turn DBSCAN eps (nm)")
    parser.add_argument("--turn-min-samples", type=int, default=3, help="Turn DBSCAN min_samples")
    
    # SID/STAR builder options
    parser.add_argument("--sid-distance", type=float, default=30.0, help="SID segment length (nm)")
    parser.add_argument("--star-distance", type=float, default=40.0, help="STAR segment length (nm)")
    parser.add_argument("--sid-star-min-flights", type=int, default=5, help="Min flights per airport")
    parser.add_argument("--sid-star-min-cluster-size", type=int, default=3, help="SID/STAR HDBSCAN min_cluster_size")
    
    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("learning.log")
        ]
    )
    
    # Run all builders
    summary = run_all(
        research_db=args.research_db,
        feedback_db=args.feedback_db,
        training_db=args.training_db,
        last_db=args.last_db,
        output_dir=args.output_dir,
        skip_paths=args.skip_paths,
        skip_turns=args.skip_turns,
        skip_sid_star=args.skip_sid_star,
        path_num_samples=args.path_num_samples,
        path_min_flights=args.path_min_flights,
        path_min_cluster_size=args.path_min_cluster_size,
        require_both_od=not args.include_partial_od,
        turn_min_deg=args.turn_min_deg,
        turn_max_deg=args.turn_max_deg,
        turn_cluster_eps=args.turn_cluster_eps,
        turn_min_samples=args.turn_min_samples,
        sid_distance=args.sid_distance,
        star_distance=args.star_distance,
        sid_star_min_flights=args.sid_star_min_flights,
        sid_star_min_cluster_size=args.sid_star_min_cluster_size,
    )
    
    return 0 if all(
        not isinstance(v, str) or not v.startswith("ERROR")
        for v in [summary["paths"], summary["turns"], summary["sids"], summary["stars"]]
        if v is not None
    ) else 1


if __name__ == "__main__":
    sys.exit(main())

