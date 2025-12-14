import sqlite3, json
path = "flight_cache.db"
conn = sqlite3.connect(path)
cur = conn.cursor()
print("tables", cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
print("schema flights", cur.execute("PRAGMA table_info(flights)").fetchall())
print("rows", cur.execute("SELECT COUNT(*) FROM flights").fetchone())
print("meta", cur.execute("SELECT flight_id, fetched_at FROM flights WHERE flight_id=?", ("3b5d8640",)).fetchone())
row = cur.execute("SELECT data FROM flights WHERE flight_id=?", ("3b5d8640",)).fetchone()
print("data len", len(row[0]) if row else None)
if row:
    data = json.loads(row[0])
    print("keys", list(data.keys()))
    pts = data.get("points", [])
    print("points", len(pts))
    if pts:
        print("first", pts[0])
        print("last", pts[-1])