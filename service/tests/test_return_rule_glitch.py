
import unittest
from core.models import FlightTrack, TrackPoint, RuleContext
from rules.rule_logic import _rule_takeoff_return

# Mock airport config if needed, but rule_logic loads it from config. 
# We rely on OJAI being in the config.

class TestReturnRuleGlitch(unittest.TestCase):
    def test_return_glitch_ignored(self):
        # OJAI coords
        lat = 31.722556
        lon = 35.993214
        
        # Create a track
        # 1. Ground points
        # 2. Takeoff climb
        # 3. Glitch to 0
        # 4. Recovery
        
        points = []
        base_ts = 1000
        
        # Ground
        for i in range(5):
            points.append(TrackPoint(
                flight_id="test",
                timestamp=base_ts + i*10,
                lat=lat,
                lon=lon,
                alt=0
            ))
            
        # Climb to 5000
        # takeoff threshold is ~3900 (2395 + 1500)
        for i in range(5):
            points.append(TrackPoint(
                flight_id="test",
                timestamp=base_ts + 100 + i*10,
                lat=lat, 
                lon=lon,
                alt=3000 + i*500 # 3000, 3500, 4000, 4500, 5000
            ))
            
        # Glitch point
        points.append(TrackPoint(
            flight_id="test",
            timestamp=base_ts + 200,
            lat=lat,
            lon=lon,
            alt=0 # Glitch!
        ))
        
        # Next point back high
        points.append(TrackPoint(
            flight_id="test",
            timestamp=base_ts + 210,
            lat=lat,
            lon=lon,
            alt=5000
        ))
        
        track = FlightTrack(flight_id="test", points=points)
        ctx = RuleContext(track=track, metadata=None, repository=None)
        
        result = _rule_takeoff_return(ctx)
        
        # Should be False because the low point was a glitch
        self.assertFalse(result.matched, "Should ignore single point glitch")
        
    def test_return_true_landing(self):
        # OJAI coords
        lat = 31.722556
        lon = 35.993214
        
        points = []
        base_ts = 1000
        
        # Ground
        for i in range(5):
            points.append(TrackPoint(
                flight_id="test",
                timestamp=base_ts + i*10,
                lat=lat,
                lon=lon,
                alt=0
            ))
            
        # Climb above 3900
        points.append(TrackPoint(
            flight_id="test",
            timestamp=base_ts + 100,
            lat=lat,
            lon=lon,
            alt=4500
        ))
        
        # Return and land (sustained)
        points.append(TrackPoint(
            flight_id="test",
            timestamp=base_ts + 200,
            lat=lat,
            lon=lon,
            alt=500
        ))
        points.append(TrackPoint(
            flight_id="test",
            timestamp=base_ts + 210,
            lat=lat,
            lon=lon,
            alt=400
        ))
        
        track = FlightTrack(flight_id="test", points=points)
        ctx = RuleContext(track=track, metadata=None, repository=None)
        
        result = _rule_takeoff_return(ctx)
        print(f"Result: {result}")
        
        self.assertTrue(result.matched, "Should detect actual return")

if __name__ == '__main__':
    unittest.main()

