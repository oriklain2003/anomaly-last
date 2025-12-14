
import requests
import time

# URL for the research anomalies endpoint
url = "http://localhost:8000/api/research/anomalies"

# Timestamps for 16 Jun 2025 (using UTC for API)
# 2025-06-16 00:00:00 UTC -> 1750032000
# 2025-06-16 23:59:59 UTC -> 1750118399

start_ts = 1750032000
end_ts = 1750118399

params = {
    "start_ts": start_ts,
    "end_ts": end_ts
}

try:
    response = requests.get(url, params=params)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data)} anomalies")
        if len(data) > 0:
            print("First 2 anomalies:")
            print(data[:2])
    else:
        print("Error response:", response.text)
        
except Exception as e:
    print(f"Request failed: {e}")

