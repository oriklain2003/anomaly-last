import sys
import os
from pathlib import Path
import unittest

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from core.models import FlightTrack, TrackPoint, RuleContext
from rules.rule_logic import _rule_abrupt_turn

class TestRuleAbruptTurn(unittest.TestCase):

    def create_point(self, i, timestamp, lat, lon, alt, heading, speed=300):
        return TrackPoint(
            flight_id="TEST001",
            timestamp=timestamp,
            lat=lat,
            lon=lon,
            alt=alt,
            gspeed=speed,
            track=heading,
            squawk="1200",
            source="test"
        )

    def test_impossible_heading_change_ignored(self):
        """
        Test that an impossible heading change (e.g., 180 degrees in 1 second)
        is ignored and does not trigger a sharp turn or holding pattern rule.
        """
        points = []
        start_lat = 32.0
        start_lon = 34.0
        
        # 1. Straight flight
        for i in range(5):
            p = self.create_point(i, 1000 + i, start_lat, start_lon + (i * 0.001), 30000, 90)
            points.append(p)
            
        # 2. GLITCH: Jump to 270 (180 deg change) in 1 sec
        p_glitch = self.create_point(5, 1005, start_lat, start_lon + (5 * 0.001), 30000, 270)
        points.append(p_glitch)
        
        # 3. Resume Straight flight
        for i in range(6, 11):
            p = self.create_point(i, 1000 + i, start_lat, start_lon + (i * 0.001), 30000, 90)
            points.append(p)

        track = FlightTrack(flight_id="TEST001", points=points)
        ctx = RuleContext(track=track, metadata=None, repository=None)
        
        result = _rule_abrupt_turn(ctx)
        
        self.assertFalse(result.matched, "Should not match abrupt turn for impossible heading glitch")
        self.assertEqual(len(result.details["events"]), 0, "Should have 0 events")

    def test_valid_holding_pattern(self):
        """
        Test that a valid holding pattern (circle) IS detected.
        This ensures we haven't broken the detection logic while ignoring glitches.
        """
        points = []
        start_ts = 1000
        start_lat = 32.0
        start_lon = 34.0
        
        # Create a full 360 degree turn
        # 36 steps of 10 degrees = 360 degrees.
        # Time step 5 seconds. Rate = 2 deg/s (< 5 deg/s limit).
        # Total time = 36 * 5 = 180 seconds (< 300s window).
        # Cumulative turn should be 360 (> 270 threshold).
        
        for i in range(40):
            heading = (i * 10) % 360
            # Circle trajectory approximation not strictly needed for heading logic,
            # but let's just vary lat/lon slightly to avoid "stationary" checks if any.
            # Actually rule checks speed > 80.
            
            p = self.create_point(i, start_ts + (i * 5), start_lat, start_lon, 30000, heading, speed=250)
            points.append(p)
            
        track = FlightTrack(flight_id="TEST002", points=points)
        ctx = RuleContext(track=track, metadata=None, repository=None)
        
        result = _rule_abrupt_turn(ctx)
        
        self.assertTrue(result.matched, "Should detect valid holding pattern")
        found_holding = False
        for evt in result.details["events"]:
            if evt.get("type") == "holding_pattern":
                found_holding = True
                break
        self.assertTrue(found_holding, "Should report holding_pattern event")

if __name__ == '__main__':
    unittest.main()
