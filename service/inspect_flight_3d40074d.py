import sqlite3
import json
import sys

flight_id = "3d40074d"
anomalies_db = "realtime/live_anomalies.db"
tracks_db = "realtime/live_tracks.db"

print(f"--- Inspecting {flight_id} ---")

# 1. Get Anomaly Report
try:
    conn = sqlite3.connect(anomalies_db)
    cursor = conn.cursor()
    cursor.execute("SELECT full_report FROM anomaly_reports WHERE flight_id = ? ORDER BY timestamp DESC LIMIT 1", (flight_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        report = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        print("\nAnomaly Report Found.")
        print(json.dumps(report, indent=2))
        layer1 = report.get("layer_1_rules", {})
        matched = layer1.get("matched_rules", [])
        for rule in matched:
            print(f"Rule {rule.get('id')}: {rule.get('summary')}")
            print(f"Details: {json.dumps(rule.get('details'), indent=2)}")
    else:
        print("\nNo anomaly report found.")
except Exception as e:
    print(f"Error reading anomalies: {e}")

# 2. Get Track Points
try:
    conn = sqlite3.connect(tracks_db)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, lat, lon FROM flight_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
    rows = cursor.fetchall()
    conn.close()

    if rows:
        print(f"\nTrack Points Found: {len(rows)}")
        track_timestamps = [r[0] for r in rows]
        print("First 5 timestamps:", track_timestamps[:5])
        print("Last 5 timestamps:", track_timestamps[-5:])
        
        anomaly_ts = [1764060719, 1764060771, 1764060823]
        print("\nChecking specific anomaly timestamps:", anomaly_ts)
        for ts in anomaly_ts:
            if ts in track_timestamps:
                print(f"  [OK] {ts} found in track.")
            else:
                print(f"  [MISSING] {ts} NOT found in track.")
                # find closest
                closest = min(track_timestamps, key=lambda x: abs(x-ts))
                print(f"       Closest is {closest} (diff {closest-ts})")

        start_ts = 1764060719
        end_ts = 1764060823
        points_in_range = [t for t in track_timestamps if start_ts <= t <= end_ts]
        print(f"\nPoints in range {start_ts}-{end_ts}: {len(points_in_range)}")
        print(points_in_range)

    else:
        print("\nNo track points found in live_tracks.db.")
except Exception as e:
    print(f"Error reading tracks: {e}")

