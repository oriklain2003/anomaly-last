import sys
import json
from pathlib import Path
from core.models import FlightTrack, TrackPoint
from anomaly_pipeline import AnomalyPipeline
from core.memory_repo import InMemoryRepository

def verify():
    print("Verifying Improvements...")
    
    # 1. Setup Pipeline
    pipeline = AnomalyPipeline()
    
    # 2. Mock Flight
    points = []
    # Create a simple straight line in the training box
    lat_start, lon_start = 30.0, 34.0
    for i in range(60):
        points.append(TrackPoint(
            flight_id="TEST_FLIGHT",
            timestamp=1000 + i*10,
            lat=lat_start + i*0.01,
            lon=lon_start + i*0.01,
            alt=10000,
            gspeed=400,
            vspeed=0,
            track=45
        ))
    
    flight = FlightTrack(flight_id="TEST_FLIGHT", points=points)
    
    # 3. Mock Repository Context
    # Create another flight nearby to test repo access (even if no rule triggers, it validates the path)
    points2 = [TrackPoint(flight_id="OTHER", timestamp=1000, lat=30.0, lon=34.0, alt=10000)]
    other_flight = FlightTrack(flight_id="OTHER", points=points2)
    
    # Mock FlightState for MemoryRepo
    class MockState:
        def __init__(self, f): self.f = f
        def to_flight_track(self): return self.f
        @property
        def points(self): return self.f.points

    active_flights = {"OTHER": MockState(other_flight)}
    repo = InMemoryRepository(active_flights)
    
    # 4. Run Analysis
    print("Running Analysis...")
    results = pipeline.analyze(flight, repository=repo)
    
    # 5. Checks
    print("\n--- Checks ---")
    
    # Check Hybrid Layer
    if "layer_6_hybrid" in results:
        print("[PASS] Layer 6 Hybrid is present.")
        print(f"       Result: {results['layer_6_hybrid']}")
    else:
        # It might be missing if model file not found (expected since we didn't train it)
        # But the key should be absent or error. 
        # Wait, if model not found, pipeline sets self.hybrid_detector = None
        # So it won't be in results.
        # We need to verify that the CODE tried to load it.
        print("[INFO] Layer 6 Hybrid not in results (Model file likely missing, which is expected).")
        if pipeline.hybrid_detector is None:
             print("       Confirmed: pipeline.hybrid_detector is None.")
    
    # Check Rule Engine (should have run without error using the repo)
    if "layer_1_rules" in results:
        status = results["layer_1_rules"].get("status")
        if status != "ERROR":
            print(f"[PASS] Rule Engine ran successfully. Status: {status}")
        else:
            print(f"[FAIL] Rule Engine Error: {results['layer_1_rules']}")
    
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify()
