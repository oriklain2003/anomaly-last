import folium
import csv
from datetime import datetime
import random

# -------------------------------------------
# LOAD A SINGLE CSV FILE
# -------------------------------------------
def load_csv_points(path: str):
    points = []  # (lat, lon, timestamp)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ts = int(row["timestamp"])
            lat = float(row["lat"])
            lon = float(row["lon"])
            alt = row["alt"]
            points.append((lat, lon, ts, alt))

    print(f"Loaded {len(points)} points from {path}")
    return points


# -------------------------------------------
# CONFIG: YOUR 2 FILES
# -------------------------------------------
CSV1 = "e.csv"
CSV2 = "e2.csv"

# Load datasets
points1 = load_csv_points(CSV1)
points2 = load_csv_points(CSV2)

# Assign random colors
color1 = "#{:06x}".format(random.randint(0, 0xFFFFFF))
color2 = "#{:06x}".format(random.randint(0, 0xFFFFFF))

# -------------------------------------------
# CREATE MAP
# -------------------------------------------
m = folium.Map(location=[31.5, 34.8], zoom_start=7)

# -------------------------------------------
# PLOT DATASET 1
# -------------------------------------------
for lat, lon, ts, alt in points1:
    pretty = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")

    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color=color1,
        fill=True,
        fill_color=color1,
        fill_opacity=0.9,
        tooltip=f"[CSV 1] {pretty} {alt}"
    ).add_to(m)

# -------------------------------------------
# PLOT DATASET 2
# -------------------------------------------
for lat, lon, ts, alt in points2:
    pretty = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")

    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color=color2,
        fill=True,
        fill_color=color2,
        fill_opacity=0.9,
        tooltip=f"[CSV 2] {pretty} {alt}"
    ).add_to(m)


# -------------------------------------------
# SAVE MAP
# -------------------------------------------
m.save("two_flight_tracks_map.html")
print("Map saved as two_flight_tracks_map.html")
