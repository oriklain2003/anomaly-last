"""
Performance profiler for flight anomaly detection.
Tests specific flight to identify bottlenecks in turn and path rules.
"""
import sqlite3
import json
import time
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, Any, List

# Setup path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.models import FlightTrack, TrackPoint, RuleContext
from rules.rule_logic import (
    _rule_abrupt_turn, _rule_off_course,
    _is_point_in_learned_turn, _get_paths, _distance_to_path,
    _load_learned_turns, _get_learned_polygons, _is_on_known_procedure,
    haversine_nm
)

SERVICE_DIR = Path(__file__).parent / "service"
PRESENT_DB = SERVICE_DIR / "present_anomalies.db"


def load_flight_from_present_db(flight_id: str) -> FlightTrack:
    """Load a flight from the present_anomalies.db."""
    conn = sqlite3.connect(str(PRESENT_DB))
    cursor = conn.cursor()
    
    # Check table structure
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables in DB: {[t[0] for t in tables]}")
    
    # Try to get flight data
    cursor.execute("SELECT * FROM anomalies WHERE flight_id LIKE ?", (f"%{flight_id}%",))
    rows = cursor.fetchall()
    
    if not rows:
        raise ValueError(f"Flight {flight_id} not found in present_anomalies.db")
    
    # Get column names
    columns = [desc[0] for desc in cursor.description]
    print(f"Columns: {columns}")
    
    row = rows[0]
    data = dict(zip(columns, row))
    print(f"Flight ID: {data.get('flight_id')}")
    print(f"Callsign: {data.get('callsign')}")
    
    # Try to get track points from the analysis_result JSON
    analysis_result = data.get('analysis_result')
    if analysis_result:
        result_data = json.loads(analysis_result)
        flight_path = result_data.get('summary', {}).get('flight_path', [])
        print(f"Found {len(flight_path)} points in flight_path")
    
    conn.close()
    
    # We need the raw track points, let's check if there's a separate table
    conn = sqlite3.connect(str(PRESENT_DB))
    cursor = conn.cursor()
    
    # Check for track_points or similar table
    for table in [t[0] for t in tables]:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        print(f"\nTable {table}: {[c[1] for c in cols]}")
    
    conn.close()
    return None


def load_flight_from_cache_db(flight_id: str) -> FlightTrack:
    """Load flight from flight_cache.db."""
    cache_db = Path(__file__).parent / "flight_cache.db"
    
    if not cache_db.exists():
        print(f"Cache DB not found at {cache_db}")
        return None
    
    conn = sqlite3.connect(str(cache_db))
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM flights WHERE flight_id LIKE ?", (f"%{flight_id}%",))
    rows = cursor.fetchall()
    
    if not rows:
        print(f"Flight {flight_id} not found in cache")
        return None
    
    columns = [desc[0] for desc in cursor.description]
    row = rows[0]
    data = dict(zip(columns, row))
    
    flight_data = json.loads(data['data'])
    print(f"Found flight: {flight_data.get('flight_id')}")
    print(f"Points: {len(flight_data.get('points', []))}")
    
    # Convert to FlightTrack
    points = []
    for p in flight_data.get('points', []):
        points.append(TrackPoint(
            flight_id=flight_data['flight_id'],
            timestamp=p['timestamp'],
            lat=p['lat'],
            lon=p['lon'],
            alt=p.get('alt', 0),
            gspeed=p.get('gspeed'),
            vspeed=p.get('vspeed'),
            track=p.get('track'),
            squawk=p.get('squawk'),
            callsign=p.get('callsign'),
        ))
    
    conn.close()
    return FlightTrack(flight_id=flight_data['flight_id'], points=points)


def load_flight_from_last_db(flight_id: str) -> FlightTrack:
    """Load flight from last.db."""
    last_db = Path(__file__).parent / "last.db"
    
    if not last_db.exists():
        print(f"last.db not found at {last_db}")
        return None
    
    conn = sqlite3.connect(str(last_db))
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"Tables in last.db: {tables}")
    
    # Try track_points table
    if 'track_points' in tables:
        cursor.execute("SELECT * FROM track_points WHERE flight_id LIKE ? LIMIT 5", (f"%{flight_id}%",))
        sample = cursor.fetchall()
        if sample:
            columns = [desc[0] for desc in cursor.description]
            print(f"Sample columns: {columns}")
            
            # Get all points for this flight
            full_flight_id = sample[0][columns.index('flight_id')]
            cursor.execute("SELECT * FROM track_points WHERE flight_id = ?", (full_flight_id,))
            all_points = cursor.fetchall()
            print(f"Found {len(all_points)} points for flight {full_flight_id}")
            
            points = []
            for row in all_points:
                data = dict(zip(columns, row))
                points.append(TrackPoint(
                    flight_id=data['flight_id'],
                    timestamp=data['timestamp'],
                    lat=data['lat'],
                    lon=data['lon'],
                    alt=data.get('alt', 0) or 0,
                    gspeed=data.get('gspeed'),
                    vspeed=data.get('vspeed'),
                    track=data.get('track'),
                    squawk=data.get('squawk'),
                    callsign=data.get('callsign'),
                ))
            
            conn.close()
            return FlightTrack(flight_id=full_flight_id, points=points)
    
    conn.close()
    return None


def profile_turn_rule(flight: FlightTrack):
    """Profile the turn rule specifically."""
    ctx = RuleContext(track=flight, metadata=None, repository=None)
    
    print("\n" + "="*60)
    print("PROFILING TURN RULE (Rule 3)")
    print("="*60)
    
    # Time individual components
    points = flight.sorted_points()
    print(f"Total points: {len(points)}")
    
    # Profile learned turns loading
    t0 = time.time()
    turns = _load_learned_turns()
    print(f"Load learned turns: {time.time() - t0:.4f}s ({len(turns)} turns)")
    
    # Profile is_point_in_learned_turn calls
    t0 = time.time()
    count = 0
    for p in points[:100]:  # Sample 100 points
        if _is_point_in_learned_turn(p.lat, p.lon):
            count += 1
    elapsed = time.time() - t0
    print(f"Check 100 points in learned turns: {elapsed:.4f}s ({elapsed/100*1000:.2f}ms per point, {count} matches)")
    
    # Profile _is_on_known_procedure
    t0 = time.time()
    count = 0
    for p in points[:100]:
        if _is_on_known_procedure(p.lat, p.lon):
            count += 1
    elapsed = time.time() - t0
    print(f"Check 100 points on known procedure: {elapsed:.4f}s ({elapsed/100*1000:.2f}ms per point, {count} matches)")
    
    # Full turn rule with profiler
    print("\nFull turn rule profiling:")
    profiler = cProfile.Profile()
    profiler.enable()
    
    t0 = time.time()
    result = _rule_abrupt_turn(ctx)
    elapsed = time.time() - t0
    
    profiler.disable()
    
    print(f"Total turn rule time: {elapsed:.4f}s")
    print(f"Matched: {result.matched}")
    print(f"Events: {len(result.details.get('events', []))}")
    
    # Print top time consumers
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("\nTop 20 time-consuming functions:")
    print(s.getvalue())


def profile_path_rule(flight: FlightTrack):
    """Profile the path/off-course rule specifically."""
    ctx = RuleContext(track=flight, metadata=None, repository=None)
    
    print("\n" + "="*60)
    print("PROFILING PATH RULE (Rule 11)")
    print("="*60)
    
    points = flight.sorted_points()
    print(f"Total points: {len(points)}")
    
    # Profile paths loading
    t0 = time.time()
    paths = _get_paths()
    print(f"Load paths: {time.time() - t0:.4f}s ({len(paths)} paths)")
    
    # Profile polygon loading
    t0 = time.time()
    polygons = _get_learned_polygons()
    print(f"Load polygons: {time.time() - t0:.4f}s ({len(polygons)} polygons)")
    
    # Profile single point distance calculation
    t0 = time.time()
    sample_point = points[len(points)//2]  # Middle point
    for path in paths:
        _distance_to_path(sample_point, path)
    elapsed = time.time() - t0
    print(f"Calculate distance to all {len(paths)} paths for 1 point: {elapsed:.4f}s")
    
    # Profile 100 points
    t0 = time.time()
    for p in points[:100]:
        for path in paths:
            _distance_to_path(p, path)
    elapsed = time.time() - t0
    print(f"Calculate distance to all paths for 100 points: {elapsed:.4f}s ({elapsed/100*1000:.2f}ms per point)")
    
    # Full path rule with profiler
    print("\nFull path rule profiling:")
    profiler = cProfile.Profile()
    profiler.enable()
    
    t0 = time.time()
    result = _rule_off_course(ctx)
    elapsed = time.time() - t0
    
    profiler.disable()
    
    print(f"Total path rule time: {elapsed:.4f}s")
    print(f"Matched: {result.matched}")
    print(f"On-path points: {result.details.get('on_path_points', 0)}")
    print(f"Off-path points: {result.details.get('off_path_points', 0)}")
    
    # Print top time consumers
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("\nTop 20 time-consuming functions:")
    print(s.getvalue())


def analyze_complexity(flight: FlightTrack):
    """Analyze the algorithmic complexity issues."""
    points = flight.sorted_points()
    n = len(points)
    
    print("\n" + "="*60)
    print("COMPLEXITY ANALYSIS")
    print("="*60)
    
    paths = _get_paths()
    turns = _load_learned_turns()
    
    print(f"\nInput sizes:")
    print(f"  - Flight points (n): {n}")
    print(f"  - Paths (p): {len(paths)}")
    print(f"  - Learned turns (t): {len(turns)}")
    
    print(f"\nComplexity estimates:")
    print(f"  - Turn rule moderate detection: O(n²) = {n*n:,} iterations")
    print(f"  - Turn rule holding pattern: O(n²) = {n*n:,} iterations")
    print(f"  - Each point × learned turns: O(n×t) = {n*len(turns):,} haversine calls")
    print(f"  - Path rule: O(n×p×segments) where avg segments ≈ 100")
    
    # Calculate average segments per path
    total_segments = 0
    for path in paths:
        centerline = path.get("centerline", [])
        total_segments += max(0, len(centerline) - 1)
    avg_segments = total_segments / len(paths) if paths else 0
    
    print(f"  - Actual avg segments per path: {avg_segments:.1f}")
    print(f"  - Path complexity estimate: {n * len(paths) * avg_segments:,.0f} segment checks")
    
    return n, len(paths), len(turns), avg_segments


def main():
    flight_id = "3b85ce16"
    
    print("="*60)
    print(f"PERFORMANCE PROFILER FOR FLIGHT: {flight_id}")
    print("="*60)
    
    # Try loading from different sources
    flight = None
    
    print("\n1. Trying last.db...")
    flight = load_flight_from_last_db(flight_id)
    
    if not flight:
        print("\n2. Trying flight_cache.db...")
        flight = load_flight_from_cache_db(flight_id)
    
    if not flight:
        print("\n3. Trying present_anomalies.db...")
        load_flight_from_present_db(flight_id)
    
    if not flight:
        print("\nERROR: Could not load flight from any database!")
        return
    
    print(f"\nLoaded flight with {len(flight.points)} points")
    
    # Run complexity analysis
    analyze_complexity(flight)
    
    # Profile turn rule
    profile_turn_rule(flight)
    
    # Profile path rule
    profile_path_rule(flight)
    
    print("\n" + "="*60)
    print("OPTIMIZATION SUGGESTIONS")
    print("="*60)
    print("""
Based on the profiling results, here are potential optimizations:

1. TURN RULE OPTIMIZATIONS:
   - Add spatial indexing for learned turns (R-tree or grid-based)
   - Skip moderate turn detection loop if flight is entirely on known procedures
   - Early exit from holding pattern detection if point is near airport
   - Cache _is_point_in_learned_turn results for nearby points

2. PATH RULE OPTIMIZATIONS:
   - Pre-filter paths by bounding box before detailed distance calculation
   - Use spatial indexing for path segments
   - Skip points that are clearly within a corridor (use simplified check first)
   - Batch distance calculations using vectorized numpy operations

3. GENERAL OPTIMIZATIONS:
   - Implement point subsampling for very long flights (>500 points)
   - Use early termination when anomaly is already confirmed
   - Cache haversine calculations for repeated point pairs
""")


if __name__ == "__main__":
    main()

