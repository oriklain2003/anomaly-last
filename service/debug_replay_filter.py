
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from fr24sdk.client import Client
from core.models import FlightTrack, TrackPoint
from datetime import datetime

API_TOKEN = "019aca50-8288-7260-94b5-6d82fbeb351c|dC21vuw2bsf2Y43qAlrBKb7iSM9ibqSDT50x3giN763b577b"

def fetch_full_track(client, flight_id):
    try:
        tracks = client.flight_tracks.get(flight_id=[flight_id])
        data_list = tracks.model_dump()["data"]
        if not data_list:
            print("No data found for flight_id")
            return None
            
        flight_data = data_list[0]
        fr24_id = flight_data["fr24_id"]
        track_points = flight_data["tracks"]
        
        points = []
        for tp in track_points:
            ts_val = tp["timestamp"]
            if isinstance(ts_val, str):
                ts = int(datetime.fromisoformat(ts_val.replace("Z", "+00:00")).timestamp())
            else:
                ts = int(ts_val)
            
            points.append(TrackPoint(
                flight_id=fr24_id,
                timestamp=ts,
                lat=float(tp["lat"]),
                lon=float(tp["lon"]),
                alt=float(tp["alt"]),
                gspeed=float(tp["gspeed"]) if tp.get("gspeed") is not None else None,
                vspeed=float(tp["vspeed"]) if tp.get("vspeed") is not None else None,
                track=float(tp["track"]) if tp.get("track") is not None else None,
                squawk=str(tp["squawk"]) if tp.get("squawk") else None,
                callsign=tp.get("callsign"),
                source=tp.get("source"),
            ))
        
        return FlightTrack(flight_id=fr24_id, points=points)
        
    except Exception as e:
        print(f"Error fetching: {e}")
        return None

def check_filter(track):
    ignored_prefixes = ["4XA", "4XB", "4XC"]
    callsign_check = None
    for p in track.points:
        if p.callsign and p.callsign.strip():
            callsign_check = p.callsign.strip().upper()
            break
    
    print(f"Detected Calllsign: '{callsign_check}'")
    
    if callsign_check and any(callsign_check.startswith(prefix) for prefix in ignored_prefixes):
        print(f"Skipping {track.flight_id} (filtered callsign {callsign_check})")
        return True
    else:
        print(f"Not Skipped")
        return False

if __name__ == "__main__":
    client = Client(api_token=API_TOKEN)
    flight_id = "3b4f891b"
    print(f"Fetching {flight_id}...")
    track = fetch_full_track(client, flight_id)
    if track:
        print(f"Points: {len(track.points)}")
        # Print first few callsigns
        for i, p in enumerate(track.points[:5]):
            print(f"Point {i} callsign: '{p.callsign}'")
            
        check_filter(track)
    else:
        print("Track not found")



