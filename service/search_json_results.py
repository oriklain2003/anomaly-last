import json

try:
    with open('web2/public/test_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded JSON with {len(data)} top-level items")
    
    # The structure seems to be a list of flight reports or a dict with flight reports
    # Based on snippet: "report": {"flight_id": ... "matched_rules": [... "summary": "Go-around detected" ...]}
    
    def search_for_go_around(obj, flight_id=None):
        if isinstance(obj, dict):
            current_flight_id = obj.get('flight_id', flight_id)
            if obj.get('flight_id'):
                 # Update context if we enter a new flight object
                 pass 
            
            if obj.get('summary') == "Go-around detected":
                print(f"Match found! Flight ID: {current_flight_id}")
                
            for k, v in obj.items():
                search_for_go_around(v, current_flight_id)
        elif isinstance(obj, list):
            for item in obj:
                search_for_go_around(item, flight_id)

    search_for_go_around(data)

except Exception as e:
    print(f"Error: {e}")



