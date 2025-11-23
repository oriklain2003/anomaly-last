import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Set

import requests
from fr24sdk.client import Client

# Configuration ----------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as cfg:
        return json.load(cfg)


CONFIG = load_config(CONFIG_PATH)

API_TOKEN = CONFIG["api_token"]
BASE_URL = CONFIG["base_url"]
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Accept-Version": "v1",
    "User-Agent": "Mozilla/5.0",
}


BOUNDS = CONFIG["bounds"]
NORTH = BOUNDS["north"]
SOUTH = BOUNDS["south"]
WEST = BOUNDS["west"]
EAST = BOUNDS["east"]
BOUNDS_PARAM = f"{NORTH},{SOUTH},{WEST},{EAST}"

DB_CFG = CONFIG["database"]
DB_PATH = DB_CFG["path"]
TABLE_NAME = DB_CFG["table"]
INDEX_NAME = DB_CFG["index"]
SCAN_TABLE_NAME = DB_CFG.get("scan_table", "flight_scans")
SCAN_INDEX_NAME = DB_CFG.get("scan_index", "idx_scan_timestamp")

TIME_CFG = CONFIG["time"]
DAYS_BACK = TIME_CFG["days_back"]
WINDOW_SPAN = TIME_CFG["window_span_seconds"]
SCAN_INTERVAL = TIME_CFG["scan_interval_seconds"]

PACED_CFG = CONFIG["pacing"]
SCAN_SLEEP_SECONDS = PACED_CFG["scan_sleep_seconds"]
TRACK_SLEEP_SECONDS = PACED_CFG["track_sleep_seconds"]
RETRY_SLEEP_SECONDS = PACED_CFG["retry_sleep_seconds"]
BATCH_SIZE = PACED_CFG["batch_size"]

ADDITIONAL_COLUMNS = {
    "gspeed": "REAL",
    "vspeed": "REAL",
    "track": "REAL",
    "squawk": "TEXT",
    "callsign": "TEXT",
    "source": "TEXT",
}


# Helpers ----------------------------------------------------------------------
def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            flight_id TEXT,
            timestamp INTEGER,
            lat REAL,
            lon REAL,
            alt REAL,
            heading REAL,
            gspeed REAL,
            vspeed REAL,
            track REAL,
            squawk TEXT,
            callsign TEXT,
            source TEXT
        )
        """
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON {TABLE_NAME} (flight_id, timestamp)"
    )
    cur.execute(f"PRAGMA table_info({TABLE_NAME})")
    existing_columns = {row[1] for row in cur.fetchall()}
    for column, dtype in ADDITIONAL_COLUMNS.items():
        if column not in existing_columns:
            cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {column} {dtype}")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SCAN_TABLE_NAME} (
            scan_timestamp INTEGER,
            flight_id TEXT,
            raw_json TEXT
        )
        """
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS {SCAN_INDEX_NAME} ON {SCAN_TABLE_NAME} (scan_timestamp)"
    )
    conn.commit()


def store_scan_rows(conn: sqlite3.Connection, scan_rows: List[dict]) -> None:
    if not scan_rows:
        return
    conn.executemany(
        f"""
        INSERT INTO {SCAN_TABLE_NAME} (
            scan_timestamp,
            flight_id,
            raw_json
        )
        VALUES (
            :scan_timestamp,
            :flight_id,
            :raw_json
        )
        """,
        scan_rows,
    )
    conn.commit()


def collect_flight_ids(conn: sqlite3.Connection, window_start: int, window_end: int) -> Set[str]:
    """Scan the bounding box inside [window_start, window_end) and gather flight IDs."""
    current = window_start
    flights: Set[str] = set()
    print(
        f"  Scanning window {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(window_start))}"
        f" → {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(window_end))}"
    )

    while current < window_end:
        params = {"bounds": BOUNDS_PARAM, "timestamp": int(current)}
        resp = requests.get(f"{BASE_URL}/historic/flight-positions/light", headers=HEADERS, params=params)

        if resp.status_code != 200:
            print(f"    Warning {current}: HTTP {resp.status_code}, skipping")
            time.sleep(SCAN_SLEEP_SECONDS)
            current += SCAN_INTERVAL
            continue

        scan_rows = []
        for flight in resp.json().get("data", []):
            flight_id = flight.get("fr24_id") or flight.get("id")
            call_sign = flight.get("callsign")
            if flight_id and call_sign and not (call_sign.startswith("4XB") or call_sign.startswith("4XC") or call_sign.startswith("4XA") or call_sign.startswith("CHLE")):
                flights.add(flight_id)
            scan_rows.append(
                {
                    "scan_timestamp": int(current),
                    "flight_id": flight_id,
                    "raw_json": json.dumps(flight),
                }
            )

        if scan_rows:
            store_scan_rows(conn, scan_rows)

        time.sleep(SCAN_SLEEP_SECONDS)
        current += SCAN_INTERVAL

    print(f"    Found {len(flights)} flights in window")
    return flights


def fetch_tracks_with_retry(client: Client, batch: List[str]):
    attempt = 1
    while True:
        try:
            return client.flight_tracks.get(flight_id=batch)
        except Exception as exc:
            print(
                f"    Batch {attempt} failed for {len(batch)} flights: {exc}. "
                f"Retrying in {RETRY_SLEEP_SECONDS}s..."
            )
            attempt += 1
            time.sleep(RETRY_SLEEP_SECONDS)


def fetch_and_store_tracks(
    conn: sqlite3.Connection,
    client: Client,
    flight_ids: Set[str],
) -> None:
    if not flight_ids:
        return

    flight_list = list(flight_ids)
    total_rows: List[dict] = []

    for idx in range(0, len(flight_list), BATCH_SIZE):
        batch = flight_list[idx : idx + BATCH_SIZE]
        print(f"    Fetching tracks for batch {idx // BATCH_SIZE + 1} ({len(batch)} flights)")

        track_responses = fetch_tracks_with_retry(client, batch)

        for ft in getattr(track_responses, "data", []):
            fid = getattr(ft, "fr24_id", None) or getattr(ft, "id", None)
            if not fid:
                continue

            rows = []
            for point in getattr(ft, "tracks", []) or []:
                lat = getattr(point, "lat", None)
                lon = getattr(point, "lon", None)
                if lat is None or lon is None:
                    continue
                if not (SOUTH <= lat <= NORTH and WEST <= lon <= EAST):
                    continue
                try:
                    ts = int(datetime.strptime(point.timestamp, "%Y-%m-%dT%H:%M:%SZ").timestamp())
                except Exception:
                    continue

                track_val = getattr(point, "track", None)
                rows.append(
                    {
                        "flight_id": fid,
                        "timestamp": ts,
                        "lat": lat,
                        "lon": lon,
                        "alt": getattr(point, "alt", None),
                        "heading": track_val,
                        "gspeed": getattr(point, "gspeed", None),
                        "vspeed": getattr(point, "vspeed", None),
                        "track": track_val,
                        "squawk": getattr(point, "squawk", None),
                        "callsign": getattr(point, "callsign", None),
                        "source": getattr(point, "source", None),
                    }
                )

            total_rows.extend(rows)

        time.sleep(TRACK_SLEEP_SECONDS)

    if total_rows:
        print(f"    Writing {len(total_rows)} track points to DB")
        conn.executemany(
            f"""
            INSERT INTO {TABLE_NAME} (
                flight_id,
                timestamp,
                lat,
                lon,
                alt,
                heading,
                gspeed,
                vspeed,
                track,
                squawk,
                callsign,
                source
            )
            VALUES (
                :flight_id,
                :timestamp,
                :lat,
                :lon,
                :alt,
                :heading,
                :gspeed,
                :vspeed,
                :track,
                :squawk,
                :callsign,
                :source
            )
            """,
            total_rows,
        )
        conn.commit()


def main() -> None:
    end_time = int(time.time())
    start_time = end_time - DAYS_BACK * 24 * 3600

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    client = Client(api_token=API_TOKEN)

    window_start = start_time
    window_index = 1

    while window_start < end_time:
        window_end = min(window_start + WINDOW_SPAN, end_time)
        print(f"\nProcessing window #{window_index}")

        flights = collect_flight_ids(conn, window_start, window_end)
        fetch_and_store_tracks(conn, client, flights)

        window_start = window_end
        window_index += 1

    conn.close()
    print("Completed windowed flight track collection.")


if __name__ == "__main__":
    main()

