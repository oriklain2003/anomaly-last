import sqlite3
import csv

def export_query_to_csv(db_path, query, output_csv):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(query)
    rows = cursor.fetchall()

    # Get column names
    column_names = [description[0] for description in cursor.description]

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)  # header
        writer.writerows(rows)         # data

    conn.close()
    print(f"CSV saved: {output_csv}")


# Example usage:
export_query_to_csv(
    db_path="last.db",
    query="""
        SELECT *
        FROM flight_tracks
        WHERE flight_id = '3aeb8b95'
        ORDER BY timestamp ASC
    """,
    output_csv="eee.csv"
)

import sqlite3
import folium
import csv
import os
from datetime import datetime

# -------------------------------------------
# LOAD FROM CSV
# -------------------------------------------
def load_from_csv(path: str):
    points = []  # (lat, lon, timestamp)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["timestamp"])
            points.append((float(row["lat"]), float(row["lon"]), ts, row["alt"]))

    print(f"Loaded {len(points)} points from CSV")
    return points


# -------------------------------------------
# LOAD FROM SQLITE
# -------------------------------------------
def load_from_sqlite(path: str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute("SELECT lat, lon, timestamp, alt FROM flight_tracks ORDER BY timestamp")
    rows = cur.fetchall()
    conn.close()

    points = [(lat, lon, ts, alt) for lat, lon, ts, alt in rows]
    print(f"Loaded {len(points)} points from SQLite")
    return points


# -------------------------------------------
# AUTO-DETECT INPUT FILE
# -------------------------------------------
FILE_PATH = "eee.csv"   # <-- change to your CSV if needed

if FILE_PATH.lower().endswith(".csv"):
    points = load_from_csv(FILE_PATH)
else:
    points = load_from_sqlite(FILE_PATH)


# -------------------------------------------
# DRAW DOTS ON MAP WITH TOOLTIP
# -------------------------------------------
m = folium.Map(location=[31.5, 34.8], zoom_start=7)

for lat, lon, ts, alt in points:

    # Convert timestamp → pretty format
    pretty_time = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")

    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.9,
        tooltip=f"📍 {pretty_time} {ts} alt={alt} lon={lon} lat={lat}"
    ).add_to(m)

m.save("flight_points_map.html")
print("Map saved as flight_points_map.html")
