import requests
import time
import sqlite3

# Flightradar24 API setup
API_TOKEN = "019a7948-61c3-72d5-9442-a62e4fea2bef|khA5LNvXdamyXIloiFGiOXkrdgvaB0iZf7sYQwcL33b49ed1"  # Replace with your real token
BASE_URL = "https://fr24api.flightradar24.com/api"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Accept-Version": "v1",
    "User-Agent": "Mozilla/5.0"
}

# Bounding box (N, S, W, E)
north = 34.597042
south = 28.536275
west = 32.299805
east = 37.397461
BATCH_SIZE = 20   # <--- YOU CAN CHANGE THIS ANYTIME

bounds_param = f"{north},{south},{west},{east}"

# Time range: last 365 days
end_time = int(time.time())
start_time = end_time - 60 * 24 * 3600
interval = 60 * 60 * 6  # every 10 minutes

# SQLite setup
conn = sqlite3.connect("flight_tracks2.db")
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS flight_tracks (
        flight_id TEXT,
        timestamp INTEGER,
        lat REAL,
        lon REAL,
        alt REAL,
        heading REAL
    )
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_flight_time ON flight_tracks (flight_id, timestamp)")
conn.commit()

# Step 1: Collect flight IDs seen in bounding box
flight_ids = set()
timestamp = start_time
print("Scanning for flights in bounding box...")

while timestamp <= end_time:
    url = f"{BASE_URL}/historic/flight-positions/light"
    params = {"bounds": bounds_param, "timestamp": int(timestamp)}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Warning: {timestamp} => {response.status_code}")
        time.sleep(5)
        timestamp += interval
        continue

    data = response.json()
    flights = data.get("data", [])
    for flight in flights:
        flight_id = flight.get("fr24_id") or flight.get("id")
        if flight_id:
            flight_ids.add(flight_id)

    time.sleep(5)
    timestamp += interval

print(f"Found {len(flight_ids)} unique flights in the area.")
from fr24sdk.client import Client

client = Client(api_token="019a981c-bc41-732d-a99f-994db065bb98|2mDFqSEvKLKoxJX2rCScnvTa3DKiEPmKzTltYMaI69f780a0")
tracks = client.flight_tracks.get(flight_id=['3d218487'])
# Step 2: Fetch and save full tracks into SQLite
from datetime import datetime


print(f"Fetching full tracks using fr24sdk in bulks of {BATCH_SIZE}...")

# Convert set → sorted list
flight_ids_list = list(flight_ids)

# Process in chunks
for i in range(0, len(flight_ids_list), BATCH_SIZE):

    batch = flight_ids_list[i:i + BATCH_SIZE]

    print(f"Fetching batch {i//BATCH_SIZE + 1} containing {len(batch)} flights...")

    try:
        track_responses = client.flight_tracks.get(flight_id=batch)
    except Exception as e:
        print(f"Batch error: {e}")
        continue

    # track_responses = [FlightTracks(...), FlightTracks(...)]
    for ft in track_responses.data:
        fid = ft.fr24_id
        points = ft.tracks  # list[FlightTrackPoint]

        in_box_rows = []

        for p in points:

            lat = p.lat
            lon = p.lon
            if lat is None or lon is None:
                continue

            # bounding box filter
            if not (south <= lat <= north and west <= lon <= east):
                continue

            # convert ISO date -> epoch seconds
            try:
                ts = int(datetime.strptime(p.timestamp, "%Y-%m-%dT%H:%M:%SZ").timestamp())
            except:
                continue

            # Build row
            in_box_rows.append({
                "flight_id": fid,
                "timestamp": ts,
                "lat": lat,
                "lon": lon,
                "alt": p.alt,
                "heading": p.track
            })

        # Save batch of rows for this flight
        if in_box_rows:
            cur.executemany("""
                INSERT INTO flight_tracks (flight_id, timestamp, lat, lon, alt, heading)
                VALUES (:flight_id, :timestamp, :lat, :lon, :alt, :heading)
            """, in_box_rows)
            conn.commit()

print("Done saving all tracks to SQLite.")
conn.close()
