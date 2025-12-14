import sqlite3
import json
import os

search_term = "_rule_emergency_squawk"

databases = [
    {
        "path": "realtime/research.db",
        "queries": [
            "SELECT flight_id, full_report FROM anomaly_reports WHERE full_report LIKE ?"
        ],
        "params": [f"%{search_term}%"]
    },
    {
        "path": "service/llm_full_pipeline/llm_research.db",
        "queries": [
            "SELECT flight_id, original_report FROM agreed_reports WHERE original_report LIKE ?",
            "SELECT flight_id, original_report FROM disagreed_reports WHERE original_report LIKE ?"
        ],
        "params": [f"%{search_term}%"]
    },
    {
        "path": "flight_cache.db",
        "queries": [
            "SELECT flight_id, data FROM flights WHERE data LIKE ?"
        ],
        "params": [f"%{search_term}%"]
    }
]

print(f"Searching for '{search_term}' in databases...\n")

def recursive_search(obj, term, results_list):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if term in k:
                results_list.append(f"Key: {k}")
            if isinstance(v, str) and term in v:
                results_list.append(f"Value in {k}: {v}")
            recursive_search(v, term, results_list)
    elif isinstance(obj, list):
        for item in obj:
            recursive_search(item, term, results_list)

for db_info in databases:
    db_path = db_info["path"]
    if not os.path.exists(db_path):
        continue
        
    print(f"--- Checking {db_path} ---")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for query in db_info["queries"]:
            try:
                cursor.execute(query, db_info["params"])
                results = cursor.fetchall()
                
                if results:
                    print(f"Found {len(results)} matches in query: {query}")
                    for row in results:
                        flight_id = row[0]
                        content = row[1]
                        print(f"  Flight ID: {flight_id}")
                        
                        if isinstance(content, str) and (content.startswith('{') or content.startswith('[')):
                            try:
                                data = json.loads(content)
                                matches = []
                                recursive_search(data, search_term, matches)
                                if matches:
                                    for m in matches:
                                        print(f"    {m}")
                                else:
                                    print(f"    (Term found in raw text but not isolated in JSON structure)")
                            except json.JSONDecodeError:
                                print(f"    Snippet: {content[:100]}...")
                        else:
                            print(f"    Snippet: {str(content)[:100]}...")
            except sqlite3.Error as e:
                print(f"Error executing query '{query}': {e}")
                
        conn.close()
    except Exception as e:
        print(f"Error accessing {db_path}: {e}")
