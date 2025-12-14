import sqlite3
import json
import os

search_strings = ["Go-around detected", "go_around"]

databases = [
    {
        "path": "realtime/research.db",
        "queries": [
            ("SELECT flight_id, full_report FROM anomaly_reports WHERE full_report LIKE ?", ["%TERM%"]),
        ]
    },
    {
        "path": "service/llm_full_pipeline/llm_research.db",
        "queries": [
            ("SELECT flight_id, original_report FROM agreed_reports WHERE original_report LIKE ?", ["%TERM%"]),
            ("SELECT flight_id, original_report FROM disagreed_reports WHERE original_report LIKE ?", ["%TERM%"])
        ]
    },
    {
        "path": "flight_cache.db",
        "queries": [
            ("SELECT flight_id, data FROM flights WHERE data LIKE ?", ["%TERM%"]),
        ]
    },
    {
        "path": "rules/flight_tracks.db",
        "queries": [
            ("SELECT name FROM sqlite_master WHERE type='table'", []),
        ]
    }
]

print(f"Searching for go-around events in databases...\n")

for db_info in databases:
    db_path = db_info["path"]
    if not os.path.exists(db_path):
        print(f"Skipping {db_path} (not found)")
        continue
        
    print(f"--- Checking {db_path} ---")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for query_template, params_template in db_info["queries"]:
            if "sqlite_master" in query_template:
                 cursor.execute(query_template)
                 # Just check if tables exist, don't print unless useful
                 continue

            current_searches = []
            if "%TERM%" in params_template[0]:
                current_searches = [(s, query_template.replace("?", f"'%{s}%'")) for s in search_strings]
            
            for term, final_query in current_searches:
                try:
                    sql = query_template
                    param = f"%{term}%"
                    
                    cursor.execute(sql, (param,))
                    results = cursor.fetchall()
                    
                    if results:
                        print(f"  [MATCH] Found {len(results)} matches for '{term}' in query: {sql}")
                        seen_flights = set()
                        for row in results:
                            flight_id = row[0]
                            if flight_id in seen_flights: continue
                            seen_flights.add(flight_id)
                            
                            content = row[1] if len(row) > 1 else "N/A"
                            print(f"    Flight ID: {flight_id}")
                            # print snippet
                            print(f"      Snippet: {str(content)[:200]}...")
                except sqlite3.Error as e:
                     pass

        conn.close()
    except Exception as e:
        print(f"Error accessing {db_path}: {e}")



