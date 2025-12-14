import sqlite3
import json
import os

search_strings = ["Emergency code transmitted", "_rule_emergency_squawk"]
squawk_codes = ["7700", "7600", "7500"]

databases = [
    {
        "path": "realtime/research.db",
        "queries": [
            # Check full report for text matches
            ("SELECT flight_id, full_report FROM anomaly_reports WHERE full_report LIKE ?", ["%TERM%"]),
            # Check tracks for squawk codes
            ("SELECT flight_id, squawk FROM anomalies_tracks WHERE squawk LIKE ?", ["%SQUAWK%"]),
            ("SELECT flight_id, squawk FROM normal_tracks WHERE squawk LIKE ?", ["%SQUAWK%"])
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
             ("SELECT flight_id, data FROM flights WHERE data LIKE ?", ["%SQUAWK%"]) # Check for squawk codes in JSON
        ]
    },
    {
        "path": "rules/flight_tracks.db",
        "queries": [
             # Assuming similar structure, inspect if tables exist first
            ("SELECT name FROM sqlite_master WHERE type='table'", []), # Discovery query
        ]
    },
     {
        "path": "last.db",
        "queries": [
            ("SELECT flight_id, squawk FROM anomalous_tracks WHERE squawk LIKE ?", ["%SQUAWK%"]),
            ("SELECT flight_id, squawk FROM flight_tracks WHERE squawk LIKE ?", ["%SQUAWK%"])
        ]
    }
]

print(f"Searching for rules/squawks in databases...\n")

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
            # Discovery mode
            if "sqlite_master" in query_template:
                 cursor.execute(query_template)
                 tables = cursor.fetchall()
                 print(f"  Tables in {db_path}: {[t[0] for t in tables]}")
                 continue

            # Determine what we are searching for based on placeholders
            current_searches = []
            if "%TERM%" in params_template[0]:
                current_searches = [(s, query_template.replace("?", f"'%{s}%'")) for s in search_strings] # manual param injection for simplicity in loop
            elif "%SQUAWK%" in params_template[0]:
                current_searches = [(s, query_template.replace("?", f"'{s}'")) for s in squawk_codes] # Exact match often better for squawk columns, but LIKE '%7700%' safer if text
            
            for term, final_query in current_searches:
                try:
                    # Using parameterized query is safer but I constructed strings above for logic simplicity
                    # Let's revert to parameterized
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
                            if "squawk" in sql.lower() and len(str(content)) < 10:
                                 print(f"      Squawk: {content}")
                            else:
                                print(f"      Snippet: {str(content)[:100]}...")
                except sqlite3.Error as e:
                     # Ignore errors if table/column doesn't exist
                     # print(f"    Query failed: {e}")
                     pass

        conn.close()
    except Exception as e:
        print(f"Error accessing {db_path}: {e}")



