import requests
import time

# Flightradar24 API setup
API_TOKEN = "019a7948-61c3-72d5-9442-a62e4fea2bef|khA5LNvXdamyXIloiFGiOXkrdgvaB0iZf7sYQwcL33b49ed1"  # Replace with your FR24 API token
BASE_URL = "https://fr24api.flightradar24.com/api"

# Authentication headers (Bearer token and API version)
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Accept-Version": "v1",
    "User-Agent": "Mozilla/5.0"  # example UA to avoid Cloudflare blocks
}

# Define the bounding box (N, S, W, E)
north = 34.597042
south = 28.536275
west  = 32.299805
east  = 37.397461
bounds_param = f"{north},{south},{west},{east}"

# Time range: last 365 days from today
end_time = int(time.time())                        # current time in UNIX epoch (seconds)
start_time = end_time - 365 * 24 * 3600            # 365 days ago in epoch seconds

# Interval between queries (in seconds)
interval = 600  # e.g. 600s = 10 minutes. Adjust for desired resolution.

flight_ids = set()  # to collect unique flight IDs seen in the area

# Loop over the time range, querying flights in the bounding box
timestamp = start_time
print("Scanning for flights in bounding box...")
while timestamp <= end_time:
    # Build request to historic positions (light) endpoint for this timestamp
    url = f"{BASE_URL}/historic/flight-positions/light"
    params = {
        "bounds": bounds_param,
        "timestamp": timestamp
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        # Handle potential errors (e.g., if rate limit exceeded or other issue)
        print(f"Warning: request at time {timestamp} returned status {response.status_code}")
        # Implement exponential backoff or wait if needed
        time.sleep(5)
        timestamp += interval
        continue

    data = response.json()
    # The JSON structure will include a list of flights (likely under a key, e.g., "data" or similar).
    # Each flight entry should contain a unique flight identifier (fr24_id) and position info.
    flights = data.get("data", [])  # assuming the flights list is in data['data']
    for flight in flights:
        flight_id = flight.get("fr24_id") or flight.get("id")
        if flight_id:
            flight_ids.add(flight_id)
    # Respect rate limit: ensure not more than ~30 calls/min (2 sec per call)
    time.sleep(2)
    timestamp += interval

print(f"Found {len(flight_ids)} unique flights in the area over last 365 days.")

# Now retrieve full tracks for each flight and extract the segment inside the box
tracks_in_box = {}  # dictionary to hold the in-box track segment for each flight ID

for fid in flight_ids:
    # Request the full track for this flight ID
    track_url = f"{BASE_URL}/historic/flight-positions/full"
    params = {"flight": fid}
    response = requests.get(track_url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Warning: track request for flight {fid} returned {response.status_code}")
        continue
    track_data = response.json()
    # Assume track_data contains flight info including a list of position points.
    # The structure might have positions under a key like "positions" or "track".
    positions = track_data.get("positions") or track_data.get("track") or track_data.get("data", [])
    # Filter positions to get segment inside the bounding box
    in_box_segment = []
    inside = False
    for pos in positions:
        lat = pos.get("lat") or pos.get("latitude")
        lon = pos.get("lon") or pos.get("longitude")
        if lat is None or lon is None:
            continue  # skip if no coordinate
        # Check if this position is inside the bounding box
        in_box = (south <= lat <= north) and (west <= lon <= east)
        if in_box and not inside:
            # Flight is entering the box at this point
            inside = True
        if inside:
            in_box_segment.append(pos)
            if not in_box:
                # Flight has just exited the box; stop collecting further points
                inside = False
                # Remove the last position because it's outside the box
                in_box_segment.pop()
                break
    if in_box_segment:
        tracks_in_box[fid] = in_box_segment

print(f"Retrieved tracks for {len(tracks_in_box)} flights that crossed the area.")

# Example: print summary of one track segment
if tracks_in_box:
    example_id, segment = next(iter(tracks_in_box.items()))
    entry_time = segment[0].get("timestamp")
    exit_time = segment[-1].get("timestamp")
    print(f"Flight {example_id} entered the box at {entry_time} and exited/landed at {exit_time}.")
    print(f"Segment inside box has {len(segment)} position points.")
