import sqlite3
import folium
import random
import os
import csv

# ======================================================
# LOAD DATA (CSV OR SQLITE)
# ======================================================

def load_from_csv(path: str):
    flights = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            fid = row["flight_id"]
            lat = float(row["lat"])
            lon = float(row["lon"])

            if fid not in flights:
                flights[fid] = []

            flights[fid].append((lat, lon))

    print(f"Loaded {len(flights)} flights from CSV")
    return flights


def load_from_sqlite(path: str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute("""
        SELECT flight_id, lat, lon 
        FROM flight_tracks
        ORDER BY timestamp ASC
    """)

    rows = cur.fetchall()
    conn.close()

    flights = {}
    for fid, lat, lon in rows:
        if fid not in flights:
            flights[fid] = []
        flights[fid].append((lat, lon))

    print(f"Loaded {len(flights)} flights from SQLite")
    return flights


# ======================================================
# AUTO-DETECT SOURCE
# ======================================================
FILE_PATH = "service/flight_3a763f49.csv"   # <-- change to your CSV if needed

if FILE_PATH.lower().endswith(".csv"):
    flights = load_from_csv(FILE_PATH)
else:
    flights = load_from_sqlite(FILE_PATH)


# ======================================================
# DRAW THE MAP
# ======================================================

# Center on Israel
m = folium.Map(location=[31.5, 34.8], zoom_start=7)

for fid, coords in flights.items():
    if len(coords) < 2:
        continue

    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    folium.PolyLine(
        coords,
        color=color,
        weight=2,
        opacity=0.8,
        tooltip=f"Flight {fid}"
    ).add_to(m)

m.save("flights_map.html")
print("Map saved as flights_map.html")
