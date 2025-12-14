from typing import List, Dict

# Manually curated lists for evaluation
# These should be updated with real flight IDs that fit the criteria

NORMAL_FLIGHTS: List[str] = [
    # TODO: Add confirmed normal flights
]

ANOMALY_FLIGHTS: List[str] = [
    "3b637311", "3bd393f6", "3b9fab61", "3afbbb46", "3b0f0866" # From anomalous_flight_ids.json
]

SIGNAL_GLITCH_FLIGHTS: List[str] = [
    "3d2e4cdb", "3d359abb", "3d359f7d", "3d390cc0", "3d31eccd" # From filtered_flight_ids.json
]

BORDERLINE_FLIGHTS: List[str] = [
    # TODO: Add borderline flights
]

DATASET = {
    "normal": NORMAL_FLIGHTS,
    "anomaly": ANOMALY_FLIGHTS,
    "glitch": SIGNAL_GLITCH_FLIGHTS,
    "borderline": BORDERLINE_FLIGHTS
}



