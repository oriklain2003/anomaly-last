import folium
from datetime import datetime
from fr24sdk.client import Client
import sys
import os

def get_flight_track(flight_id):
    print(f"Fetching track for flight ID: {flight_id}...")
    
    # Initialize client with the token from playground.py
    client = Client(api_token="019a9d54-678a-73da-a927-1f7a60b27a8f|xouEuixtJFLIlwNKajdW0mFjV4sguJsf5eByvIYTde7f4b26")
    
    try:
        # Fetch flight tracks
        tracks = client.flight_tracks.get(flight_id=[flight_id])
        
        # Check if we received data
        data = tracks.model_dump().get("data", [])
        if not data:
            print(f"No data found for flight ID {flight_id}")
            return None
            
        flight_data = data[0]
        raw_points = flight_data.get("tracks", [])
        
        if not raw_points:
            print("No track points found in flight data.")
            return None

        points = []
        for tp in raw_points:
            # Parse ISO timestamp to unix timestamp (handling Z for UTC)
            ts_str = tp["timestamp"].replace("Z", "+00:00")
            try:
                ts = int(datetime.fromisoformat(ts_str).timestamp())
            except ValueError:
                # Fallback if format is different
                ts = 0
            
            points.append({
                "lat": float(tp["lat"]),
                "lon": float(tp["lon"]),
                "alt": float(tp["alt"]),
                "ts": ts,
                "gspeed": tp.get("gspeed"),
                "vspeed": tp.get("vspeed"),
                "track": tp.get("track"),
                "callsign": tp.get("callsign")
            })
            
        print(f"Retrieved {len(points)} track points.")
        return points
        
    except Exception as e:
        print(f"Error fetching flight data: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_map(points, flight_id, output_file=None):
    if not points:
        print("No points to plot.")
        return

    if output_file is None:
        output_file = f"map_{flight_id}.html"

    # Calculate center of the map
    avg_lat = sum(p["lat"] for p in points) / len(points)
    avg_lon = sum(p["lon"] for p in points) / len(points)
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6)
    
    # Add a line connecting the points
    line_points = [[p["lat"], p["lon"]] for p in points]
    folium.PolyLine(line_points, color="red", weight=2.5, opacity=0.8).add_to(m)

    # Add points
    for p in points:
        try:
            pretty_time = datetime.fromtimestamp(p["ts"]).strftime("%Y-%m-%d %H:%M:%S")
        except:
            pretty_time = str(p["ts"])
            
        tooltip_text = (
            f"<b>Time:</b> {pretty_time}<br>"
            f"<b>Alt:</b> {p['alt']} ft<br>"
            f"<b>Speed:</b> {p['gspeed']} kts<br>"
            f"<b>Lat:</b> {p['lat']}<br>"
            f"<b>Lon:</b> {p['lon']}"
        )
        
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=3,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.9,
            tooltip=tooltip_text
        ).add_to(m)

    m.save(output_file)
    print(f"Map saved to {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # Allow passing flight ID as command line argument
    if len(sys.argv) > 1:
        f_id = sys.argv[1]
    else:
        f_id = input("Enter flight ID (e.g., 3bc6854c): ").strip()
    
    if f_id:
        track_data = get_flight_track(f_id)
        if track_data:
            create_map(track_data, f_id)

