"""
Intelligence gathering and pattern detection for Level 3 features.

Provides:
- GPS jamming detection and mapping
- Military aircraft tracking
- Anomaly DNA (pattern matching)
- Event correlation
"""
from __future__ import annotations

import json
import math
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime, timedelta


# Military callsign patterns and their associated metadata
MILITARY_CALLSIGN_PATTERNS = {
    # US Military
    'RCH': {'country': 'US', 'type': 'transport', 'name': 'REACH - USAF AMC'},
    'CNV': {'country': 'US', 'type': 'transport', 'name': 'Convoy - USAF'},
    'QUID': {'country': 'US', 'type': 'tanker', 'name': 'KC-135/KC-10 Tanker'},
    'NCHO': {'country': 'US', 'type': 'tanker', 'name': 'KC-135 Stratotanker'},
    'SHELL': {'country': 'US', 'type': 'tanker', 'name': 'Aerial Refueling'},
    'JAKE': {'country': 'US', 'type': 'ISR', 'name': 'RC-135 Rivet Joint'},
    'DOOM': {'country': 'US', 'type': 'ISR', 'name': 'E-8 JSTARS'},
    'HOMER': {'country': 'US', 'type': 'ISR', 'name': 'E-3 AWACS'},
    'VIPER': {'country': 'US', 'type': 'fighter', 'name': 'F-16 Fighting Falcon'},
    'RAGE': {'country': 'US', 'type': 'fighter', 'name': 'Fighter Aircraft'},
    
    # UK Military
    'RAF': {'country': 'GB', 'type': 'transport', 'name': 'Royal Air Force'},
    'RFR': {'country': 'GB', 'type': 'tanker', 'name': 'RAF Voyager Tanker'},
    'ASCOT': {'country': 'GB', 'type': 'transport', 'name': 'RAF Transport'},
    
    # Russian Military
    'RRR': {'country': 'RU', 'type': 'transport', 'name': 'Russian Air Force'},
    'RFF': {'country': 'RU', 'type': 'transport', 'name': 'Russian Federation'},
    
    # Israeli Military
    'IAF': {'country': 'IL', 'type': 'transport', 'name': 'Israeli Air Force'},
    'ISF': {'country': 'IL', 'type': 'fighter', 'name': 'Israeli Air Force'},
    
    # NATO/Allied
    'NATO': {'country': 'NATO', 'type': 'ISR', 'name': 'NATO Operations'},
    'MMF': {'country': 'NATO', 'type': 'transport', 'name': 'Multinational MRTT Fleet'},
}


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in nautical miles between two coordinates."""
    R = 3440.065  # Earth radius in nautical miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


class IntelligenceEngine:
    """Engine for intelligence gathering and pattern detection."""
    
    def __init__(self, db_paths: Dict[str, Path]):
        self.db_paths = db_paths
    
    def _get_connection(self, db_name: str) -> Optional[sqlite3.Connection]:
        """Get database connection."""
        path = self.db_paths.get(db_name)
        if not path or not path.exists():
            return None
        return sqlite3.connect(str(path))
    
    def _execute_query(self, db_name: str, query: str, params: tuple = ()) -> List[tuple]:
        """Execute query and return results."""
        conn = self._get_connection(db_name)
        if not conn:
            return []
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except sqlite3.OperationalError as e:
            # Handle missing tables gracefully
            if "no such table" in str(e):
                return []
            raise
        finally:
            conn.close()
    
    def get_gps_jamming_heatmap(self, start_ts: int, end_ts: int, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Detect GPS jamming/signal loss patterns by analyzing track gaps.
        
        Algorithm:
        1. Query track points ordered by flight_id and timestamp
        2. Detect gaps where next_timestamp - current_timestamp > 5 minutes (300s)
        3. Exclude gaps within 5nm of known airports (normal ADS-B coverage loss)
        4. Record gap location (lat/lon of last known position)
        5. Aggregate by geographic grid (1.0 degrees ~111km)
        6. Calculate intensity based on frequency and affected flights
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            limit: Max zones to return (default 30, prevents browser overload)
        
        Returns:
            [{lat, lon, intensity, first_seen, last_seen, event_count, affected_flights}]
        """
        # Grid-based aggregation (1.0 degree cells ~111km for broader zones)
        grid_size = 1.0
        gap_threshold_seconds = 300  # Signal loss if gap > 5 minutes (300 seconds)
        airport_exclusion_nm = 5  # Exclude signal loss within 5nm of airports
        
        # Major airports to exclude (signal loss near airports is normal - aircraft below coverage)
        airports = {
            'LLBG': {'lat': 32.0114, 'lon': 34.8867, 'name': 'Ben Gurion'},
            'LLER': {'lat': 29.9403, 'lon': 35.0004, 'name': 'Ramon'},
            'LLHA': {'lat': 32.8094, 'lon': 35.0431, 'name': 'Haifa'},
            'LLOV': {'lat': 31.2875, 'lon': 34.7228, 'name': 'Ovda'},
            'OJAI': {'lat': 31.7226, 'lon': 35.9932, 'name': 'Amman'},
            'OJAM': {'lat': 31.9726, 'lon': 35.9916, 'name': 'Marka'},
            'OLBA': {'lat': 33.8209, 'lon': 35.4884, 'name': 'Beirut'},
            'LCRA': {'lat': 34.5904, 'lon': 32.9879, 'name': 'Akrotiri'},
            'LCLK': {'lat': 34.8751, 'lon': 33.6249, 'name': 'Larnaca'},
            'LLSD': {'lat': 32.1147, 'lon': 34.7822, 'name': 'Sde Dov'},
            'HECA': {'lat': 30.1219, 'lon': 31.4056, 'name': 'Cairo'},
            'HEGN': {'lat': 27.1783, 'lon': 33.7994, 'name': 'Hurghada'},
            'HESH': {'lat': 27.9773, 'lon': 34.3950, 'name': 'Sharm El Sheikh'},
            'OMDB': {'lat': 25.2528, 'lon': 55.3644, 'name': 'Dubai'},
            'OERK': {'lat': 24.9576, 'lon': 46.6988, 'name': 'Riyadh'},
            'OTHH': {'lat': 25.2731, 'lon': 51.6081, 'name': 'Doha'},
            'LTFM': {'lat': 41.2753, 'lon': 28.7519, 'name': 'Istanbul'},
            'LGAV': {'lat': 37.9364, 'lon': 23.9445, 'name': 'Athens'},
            'KJFK': {'lat': 40.6413, 'lon': -73.7781, 'name': 'JFK'},
        }
        
        def is_near_airport(lat: float, lon: float) -> bool:
            """Check if position is within exclusion radius of any airport."""
            for icao, coords in airports.items():
                dist = _haversine_nm(lat, lon, coords['lat'], coords['lon'])
                if dist <= airport_exclusion_nm:
                    return True
            return False
        
        jamming_grid = defaultdict(lambda: {
            'events': [],
            'first_seen': None,
            'last_seen': None,
            'flights': set(),
            'total_gap_duration': 0,
            'lat_sum': 0.0,  # Track actual positions for centroid
            'lon_sum': 0.0
        })
        
        # Query track points from research.db tables to detect gaps
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT flight_id, timestamp, lat, lon
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                  AND lat IS NOT NULL 
                  AND lon IS NOT NULL
                ORDER BY flight_id, timestamp
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            
            # Process track points to find gaps
            prev_flight_id = None
            prev_timestamp = None
            prev_lat = None
            prev_lon = None
            
            for row in results:
                flight_id, timestamp, lat, lon = row
                
                if flight_id == prev_flight_id and prev_timestamp:
                    gap_seconds = timestamp - prev_timestamp
                    
                    # Signal loss detected if gap > threshold
                    if gap_seconds > gap_threshold_seconds:
                        # Skip if near an airport (normal coverage loss during takeoff/landing)
                        if is_near_airport(prev_lat, prev_lon):
                            prev_flight_id = flight_id
                            prev_timestamp = timestamp
                            prev_lat = lat
                            prev_lon = lon
                            continue
                        
                        # Use the last known position as the gap location
                        grid_lat = round(prev_lat / grid_size) * grid_size
                        grid_lon = round(prev_lon / grid_size) * grid_size
                        grid_key = (grid_lat, grid_lon)
                        
                        cell = jamming_grid[grid_key]
                        cell['events'].append(timestamp)
                        cell['flights'].add(flight_id)
                        cell['total_gap_duration'] += gap_seconds
                        # Track actual positions for centroid calculation
                        cell['lat_sum'] += prev_lat
                        cell['lon_sum'] += prev_lon
                        
                        if cell['first_seen'] is None or timestamp < cell['first_seen']:
                            cell['first_seen'] = timestamp
                        if cell['last_seen'] is None or timestamp > cell['last_seen']:
                            cell['last_seen'] = timestamp
                
                prev_flight_id = flight_id
                prev_timestamp = timestamp
                prev_lat = lat
                prev_lon = lon
        
        # Format output with intensity scores
        result = []
        for (grid_lat, grid_lon), data in jamming_grid.items():
            # Intensity based on: number of events + unique flights affected
            event_score = min(50, len(data['events']) * 5)
            flight_score = min(50, len(data['flights']) * 10)
            intensity = event_score + flight_score
            
            # Use centroid of actual event positions, not grid center
            num_events = len(data['events'])
            centroid_lat = data['lat_sum'] / num_events if num_events > 0 else grid_lat
            centroid_lon = data['lon_sum'] / num_events if num_events > 0 else grid_lon
            
            result.append({
                'lat': round(centroid_lat, 4),
                'lon': round(centroid_lon, 4),
                'intensity': intensity,
                'first_seen': data['first_seen'],
                'last_seen': data['last_seen'],
                'event_count': num_events,
                'affected_flights': len(data['flights']),
                'avg_gap_duration_s': int(data['total_gap_duration'] / max(num_events, 1))
            })
        
        # Sort by intensity and limit to prevent browser overload
        sorted_result = sorted(result, key=lambda x: x['intensity'], reverse=True)
        return sorted_result[:limit]
    
    def get_military_patterns(self, start_ts: int, end_ts: int,
                             country: Optional[str] = None,
                             aircraft_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Track military aircraft patterns.
        
        Identifies military aircraft by callsign patterns, detects flight patterns
        (racetrack, orbit, transit), and extracts track locations.
        
        Args:
            country: Filter by country (e.g., "US", "RU", "GB", "IL", "NATO")
            aircraft_type: Filter by type (e.g., "tanker", "ISR", "transport", "fighter")
        
        Returns:
            [{flight_id, callsign, country, type, pattern_type, locations: [...], frequency}]
        """
        patterns = []
        
        # Build callsign pattern query dynamically
        callsign_prefixes = list(MILITARY_CALLSIGN_PATTERNS.keys())
        like_clauses = ' OR '.join([f"callsign LIKE '{prefix}%'" for prefix in callsign_prefixes])
        
        # Query flights with military callsign patterns
        all_results = []
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT DISTINCT flight_id, callsign
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                AND ({like_clauses})
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            all_results.extend(results)
        
        # Remove duplicates
        seen_flights = set()
        unique_results = []
        for row in all_results:
            if row[0] not in seen_flights:
                seen_flights.add(row[0])
                unique_results.append(row)
        
        for row in unique_results:
            flight_id, callsign = row
            if not callsign:
                continue
            
            # Identify aircraft type and country from callsign
            mil_info = self._identify_military_aircraft(callsign)
            if not mil_info:
                continue
            
            # Apply filters
            if country and mil_info['country'] != country:
                continue
            if aircraft_type and mil_info['type'] != aircraft_type:
                continue
            
            # Get track data for pattern analysis and locations
            track_data = self._get_flight_track(flight_id, start_ts, end_ts)
            
            # Analyze flight pattern
            pattern_type = self._analyze_flight_pattern(track_data)
            
            # Extract key locations (sampled points)
            locations = self._extract_key_locations(track_data)
            
            patterns.append({
                'flight_id': flight_id,
                'callsign': callsign,
                'country': mil_info['country'],
                'type': mil_info['type'],
                'type_name': mil_info.get('name', ''),
                'pattern_type': pattern_type,
                'locations': locations,
                'frequency': 1,
                'track_points': len(track_data)
            })
        
        # Sort by country then type
        patterns.sort(key=lambda x: (x['country'], x['type']))
        
        return patterns[:50]  # Limit results
    
    def _identify_military_aircraft(self, callsign: str) -> Optional[Dict[str, str]]:
        """
        Identify military aircraft type and country from callsign.
        
        Returns:
            {'country': str, 'type': str, 'name': str} or None
        """
        if not callsign:
            return None
        
        callsign_upper = callsign.upper()
        
        # Check against known military callsign patterns
        for prefix, info in MILITARY_CALLSIGN_PATTERNS.items():
            if callsign_upper.startswith(prefix):
                return info.copy()
        
        # Additional heuristics for unrecognized patterns
        # Numeric-heavy callsigns with certain characteristics may be military
        if len(callsign) >= 5:
            # Check for all-caps with numbers pattern common in military
            alpha_count = sum(1 for c in callsign if c.isalpha())
            digit_count = sum(1 for c in callsign if c.isdigit())
            
            if digit_count >= 3 and alpha_count <= 3:
                # Likely military or government
                return {'country': 'UNKNOWN', 'type': 'unknown', 'name': 'Unidentified Military'}
        
        return None
    
    def _get_flight_track(self, flight_id: str, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get track points for a flight.
        
        Returns:
            List of {timestamp, lat, lon, alt, track, gspeed}
        """
        track_data = []
        
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT timestamp, lat, lon, alt, track, gspeed
                FROM {table_name}
                WHERE flight_id = ?
                  AND timestamp BETWEEN ? AND ?
                  AND lat IS NOT NULL AND lon IS NOT NULL
                ORDER BY timestamp
                LIMIT 500
            """
            results = self._execute_query('research', query, (flight_id, start_ts, end_ts))
            
            for row in results:
                ts, lat, lon, alt, heading, gspeed = row
                track_data.append({
                    'timestamp': ts,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt or 0,
                    'track': heading,
                    'gspeed': gspeed or 0
                })
            
            if track_data:
                break  # Found data, no need to check other table
        
        return track_data
    
    def _analyze_flight_pattern(self, track_data: List[Dict[str, Any]]) -> str:
        """
        Analyze track data to determine flight pattern type.
        
        Returns:
            'racetrack', 'orbit', 'transit', or 'unknown'
        """
        if len(track_data) < 10:
            return 'unknown'
        
        # Calculate total heading change
        total_heading_change = 0
        heading_changes = []
        
        for i in range(1, len(track_data)):
            prev_hdg = track_data[i-1].get('track')
            curr_hdg = track_data[i].get('track')
            
            if prev_hdg is not None and curr_hdg is not None:
                # Calculate signed heading change
                delta = ((curr_hdg - prev_hdg + 540) % 360) - 180
                heading_changes.append(delta)
                total_heading_change += delta
        
        if not heading_changes:
            return 'transit'
        
        # Analyze the pattern
        abs_total = abs(total_heading_change)
        
        # Count significant turns (> 45 degrees accumulated)
        significant_turns = 0
        accumulated = 0
        for delta in heading_changes:
            accumulated += delta
            if abs(accumulated) >= 45:
                significant_turns += 1
                accumulated = 0
        
        # Calculate distance traveled vs displacement
        if len(track_data) >= 2:
            start = track_data[0]
            end = track_data[-1]
            displacement = _haversine_nm(start['lat'], start['lon'], end['lat'], end['lon'])
            
            # Calculate total distance
            total_distance = 0
            for i in range(1, len(track_data)):
                p1, p2 = track_data[i-1], track_data[i]
                total_distance += _haversine_nm(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
            
            # Ratio of displacement to total distance
            if total_distance > 0:
                efficiency = displacement / total_distance
            else:
                efficiency = 1.0
        else:
            efficiency = 1.0
        
        # Pattern classification
        # Orbit: Multiple 360-degree turns, returns near start
        if abs_total >= 300 and efficiency < 0.3:
            return 'orbit'
        
        # Racetrack: 180-degree turns (back and forth pattern)
        if significant_turns >= 2 and 0.2 < efficiency < 0.6:
            return 'racetrack'
        
        # Transit: Relatively straight path
        if efficiency > 0.7 or abs_total < 90:
            return 'transit'
        
        # Mixed or complex pattern
        if significant_turns >= 3:
            return 'racetrack'
        
        return 'transit'
    
    def _extract_key_locations(self, track_data: List[Dict[str, Any]], max_points: int = 5) -> List[Dict[str, float]]:
        """
        Extract key locations from track data.
        
        Returns up to max_points evenly sampled locations.
        """
        if not track_data:
            return []
        
        locations = []
        
        # Sample evenly from the track
        if len(track_data) <= max_points:
            indices = range(len(track_data))
        else:
            step = len(track_data) / max_points
            indices = [int(i * step) for i in range(max_points)]
        
        for idx in indices:
            point = track_data[idx]
            locations.append({
                'lat': round(point['lat'], 4),
                'lon': round(point['lon'], 4),
                'alt': point.get('alt', 0)
            })
        
        return locations
    
    def get_anomaly_dna(self, flight_id: str, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Find similar historical flights (pattern fingerprinting).
        
        Enhanced algorithm with multi-factor similarity scoring:
        1. Geographic overlap (bounding box intersection)
        2. Rule-based similarity (same anomaly rules triggered)
        3. Temporal similarity (same time of day/week)
        4. Callsign pattern similarity (same airline/operator)
        5. Flight profile similarity (altitude, speed patterns)
        
        Returns:
            {
                flight_info: {...},
                similar_flights: [{flight_id, similarity_score, date, pattern, similarity_factors}],
                recurring_pattern: str,
                risk_assessment: str,
                insights: [...]
            }
        """
        now = int(datetime.now().timestamp())
        lookback_ts = now - (lookback_days * 86400)
        
        # Get the target flight's data
        flight_info = None
        flight_path = []
        flight_anomalies = []
        flight_timestamp = None
        
        # Query flight data from research.db
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT timestamp, lat, lon, alt, track, callsign
                FROM {table_name}
                WHERE flight_id = ?
                ORDER BY timestamp
                LIMIT 200
            """
            results = self._execute_query('research', query, (flight_id,))
            
            if results:
                for row in results:
                    ts, lat, lon, alt, track, callsign = row
                    flight_path.append({
                        'timestamp': ts,
                        'lat': lat, 
                        'lon': lon, 
                        'alt': alt or 0, 
                        'track': track
                    })
                    if not flight_info and callsign:
                        flight_info = {'callsign': callsign, 'flight_id': flight_id}
                    if not flight_timestamp:
                        flight_timestamp = ts
                break  # Found data
        
        if not flight_path:
            return {
                'flight_info': {'flight_id': flight_id},
                'similar_flights': [],
                'recurring_pattern': 'No data found for this flight',
                'risk_assessment': 'Unknown',
                'insights': []
            }
        
        # Calculate flight's "fingerprint"
        lats = [p['lat'] for p in flight_path if p['lat']]
        lons = [p['lon'] for p in flight_path if p['lon']]
        alts = [p['alt'] for p in flight_path if p['alt']]
        
        if lats and lons:
            bbox = {
                'min_lat': min(lats), 'max_lat': max(lats),
                'min_lon': min(lons), 'max_lon': max(lons)
            }
            centroid = {
                'lat': sum(lats) / len(lats),
                'lon': sum(lons) / len(lons)
            }
        else:
            bbox = None
            centroid = None
        
        # Calculate altitude profile
        avg_alt = sum(alts) / len(alts) if alts else 0
        max_alt = max(alts) if alts else 0
        
        # Get time of day (hour) for temporal matching
        flight_hour = datetime.fromtimestamp(flight_timestamp).hour if flight_timestamp else 12
        flight_weekday = datetime.fromtimestamp(flight_timestamp).weekday() if flight_timestamp else 0
        
        # Get anomaly reports for this flight
        target_rule_ids = set()
        for db_name in ['research', 'anomalies']:
            query = """
                SELECT timestamp, full_report
                FROM anomaly_reports
                WHERE flight_id = ?
            """
            results = self._execute_query(db_name, query, (flight_id,))
            
            for row in results:
                ts, report_json = row
                try:
                    report = json.loads(report_json) if isinstance(report_json, str) else report_json
                    matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                    for rule in matched_rules:
                        rule_id = rule.get('id')
                        target_rule_ids.add(rule_id)
                        flight_anomalies.append({
                            'rule_id': rule_id,
                            'rule_name': rule.get('name', 'Unknown'),
                            'timestamp': ts
                        })
                except:
                    pass
        
        # Find similar flights with enhanced scoring
        similar_flights = []
        candidate_flights = {}  # flight_id -> {callsign, first_ts, rule_ids}
        
        # Query geographically similar flights
        if bbox:
            for table_name in ['anomalies_tracks', 'normal_tracks']:
                query = f"""
                    SELECT DISTINCT flight_id, callsign, MIN(timestamp) as first_ts
                    FROM {table_name}
                    WHERE flight_id != ?
                      AND timestamp BETWEEN ? AND ?
                      AND lat BETWEEN ? AND ?
                      AND lon BETWEEN ? AND ?
                    GROUP BY flight_id
                    LIMIT 100
                """
                params = (
                    flight_id, lookback_ts, now,
                    bbox['min_lat'] - 1.0, bbox['max_lat'] + 1.0,
                    bbox['min_lon'] - 1.0, bbox['max_lon'] + 1.0
                )
                results = self._execute_query('research', query, params)
                
                for row in results:
                    other_id, other_callsign, first_ts = row
                    if other_id not in candidate_flights:
                        candidate_flights[other_id] = {
                            'callsign': other_callsign,
                            'first_ts': first_ts,
                            'rule_ids': set()
                        }
        
        # Get rule matches for candidate flights
        if candidate_flights:
            for db_name in ['research', 'anomalies']:
                flight_ids_str = ','.join([f"'{fid}'" for fid in candidate_flights.keys()])
                query = f"""
                    SELECT flight_id, full_report
                    FROM anomaly_reports
                    WHERE flight_id IN ({flight_ids_str})
                """
                results = self._execute_query(db_name, query, ())
                
                for row in results:
                    other_id, report_json = row
                    try:
                        report = json.loads(report_json) if isinstance(report_json, str) else report_json
                        matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                        for rule in matched_rules:
                            if other_id in candidate_flights:
                                candidate_flights[other_id]['rule_ids'].add(rule.get('id'))
                    except:
                        pass
        
        # Calculate similarity scores
        for other_id, data in candidate_flights.items():
            similarity_factors = {}
            total_score = 0
            
            # Factor 1: Geographic overlap (0-25 points)
            geo_score = 25  # Base score for being in search area
            similarity_factors['geographic'] = geo_score
            total_score += geo_score
            
            # Factor 2: Callsign similarity (0-30 points)
            callsign_score = 0
            if flight_info and data['callsign'] and flight_info.get('callsign'):
                target_cs = flight_info['callsign'].upper()
                other_cs = data['callsign'].upper()
                
                if other_cs == target_cs:
                    callsign_score = 30  # Exact match
                elif len(target_cs) >= 3 and len(other_cs) >= 3 and other_cs[:3] == target_cs[:3]:
                    callsign_score = 20  # Same airline
                elif len(target_cs) >= 2 and len(other_cs) >= 2 and other_cs[:2] == target_cs[:2]:
                    callsign_score = 10  # Similar prefix
            
            similarity_factors['callsign'] = callsign_score
            total_score += callsign_score
            
            # Factor 3: Rule-based similarity (0-25 points)
            rule_score = 0
            if target_rule_ids and data['rule_ids']:
                common_rules = target_rule_ids.intersection(data['rule_ids'])
                if common_rules:
                    # More common rules = higher score
                    rule_score = min(25, len(common_rules) * 10)
            
            similarity_factors['rules'] = rule_score
            total_score += rule_score
            
            # Factor 4: Temporal similarity (0-20 points)
            temporal_score = 0
            if data['first_ts']:
                other_hour = datetime.fromtimestamp(data['first_ts']).hour
                other_weekday = datetime.fromtimestamp(data['first_ts']).weekday()
                
                # Same hour of day (+10)
                hour_diff = abs(other_hour - flight_hour)
                if hour_diff <= 1 or hour_diff >= 23:
                    temporal_score += 10
                elif hour_diff <= 3 or hour_diff >= 21:
                    temporal_score += 5
                
                # Same day of week (+10)
                if other_weekday == flight_weekday:
                    temporal_score += 10
                elif abs(other_weekday - flight_weekday) <= 1:
                    temporal_score += 5
            
            similarity_factors['temporal'] = temporal_score
            total_score += temporal_score
            
            # Determine pattern description
            pattern_desc = []
            if callsign_score >= 20:
                pattern_desc.append('same_operator')
            if rule_score >= 15:
                pattern_desc.append('same_anomalies')
            if temporal_score >= 15:
                pattern_desc.append('same_schedule')
            if not pattern_desc:
                pattern_desc.append('geographic_overlap')
            
            similar_flights.append({
                'flight_id': other_id,
                'callsign': data['callsign'] or 'Unknown',
                'similarity_score': total_score,
                'date': datetime.fromtimestamp(data['first_ts']).isoformat() if data['first_ts'] else None,
                'pattern': '+'.join(pattern_desc),
                'similarity_factors': similarity_factors,
                'common_rules': list(target_rule_ids.intersection(data['rule_ids'])) if target_rule_ids and data['rule_ids'] else []
            })
        
        # Sort by similarity
        similar_flights.sort(key=lambda x: x['similarity_score'], reverse=True)
        similar_flights = similar_flights[:15]  # Top 15
        
        # Determine recurring pattern with enhanced analysis
        high_similarity = [f for f in similar_flights if f['similarity_score'] >= 60]
        same_operator = [f for f in similar_flights if 'same_operator' in f['pattern']]
        same_anomalies = [f for f in similar_flights if 'same_anomalies' in f['pattern']]
        
        if len(same_operator) >= 3 and len(same_anomalies) >= 2:
            recurring_pattern = f"Strong recurring pattern: {len(same_operator)} flights by same operator with {len(same_anomalies)} showing same anomaly types"
            risk_assessment = 'High - Systematic pattern requiring investigation'
        elif len(high_similarity) >= 3:
            recurring_pattern = f"Recurring pattern detected: {len(high_similarity)} highly similar flights"
            risk_assessment = 'High - Possible reconnaissance or surveillance pattern'
        elif len(same_operator) >= 2:
            recurring_pattern = f"Operator pattern: {len(same_operator)} flights by same operator in this area"
            risk_assessment = 'Medium - Repeated operator activity'
        elif len(similar_flights) >= 3:
            recurring_pattern = f"Partial pattern: {len(similar_flights)} flights in similar geographic area"
            risk_assessment = 'Medium - Geographic overlap but varied operators'
        else:
            recurring_pattern = 'No significant recurring pattern detected'
            risk_assessment = 'Low - Unique or infrequent flight path'
        
        # Generate enhanced insights
        insights = []
        if flight_anomalies:
            insights.append(f"This flight triggered {len(flight_anomalies)} anomaly rules")
            rule_names = list(set(a['rule_name'] for a in flight_anomalies))[:5]
            insights.append(f"Anomaly types: {', '.join(rule_names)}")
        
        if similar_flights:
            insights.append(f"Found {len(similar_flights)} historically similar flights")
            avg_score = sum(f['similarity_score'] for f in similar_flights) / len(similar_flights)
            insights.append(f"Average similarity score: {avg_score:.0f}/100")
        
        if same_operator:
            insights.append(f"Same operator flew this route {len(same_operator)} times in the past {lookback_days} days")
        
        if same_anomalies:
            insights.append(f"{len(same_anomalies)} other flights triggered the same anomaly rules")
        
        if bbox:
            insights.append(f"Flight operated in area: {bbox['min_lat']:.2f}°N to {bbox['max_lat']:.2f}°N, {bbox['min_lon']:.2f}°E to {bbox['max_lon']:.2f}°E")
        
        if avg_alt > 0:
            insights.append(f"Average altitude: {avg_alt:.0f} ft, Max altitude: {max_alt:.0f} ft")
        
        return {
            'flight_info': flight_info or {'flight_id': flight_id},
            'similar_flights': similar_flights,
            'recurring_pattern': recurring_pattern,
            'risk_assessment': risk_assessment,
            'insights': insights,
            'anomalies_detected': flight_anomalies,
            'fingerprint': {
                'bbox': bbox,
                'centroid': centroid,
                'avg_altitude': avg_alt,
                'max_altitude': max_alt,
                'rule_ids': list(target_rule_ids),
                'flight_hour': flight_hour,
                'flight_weekday': flight_weekday
            }
        }
    
    def detect_pattern_clusters(self, start_ts: int, end_ts: int,
                               min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Detect recurring suspicious patterns across multiple flights.
        
        Returns:
            [{pattern_id, description, flights: [...], first_seen, last_seen, risk_level}]
        """
        # Group flights by geographic hotspots and anomaly types
        patterns = []
        
        # Query anomaly reports and group by location
        location_clusters = defaultdict(list)
        
        for db_name in ['research', 'anomalies']:
            query = """
                SELECT flight_id, timestamp, full_report
                FROM anomaly_reports
                WHERE timestamp BETWEEN ? AND ?
            """
            results = self._execute_query(db_name, query, (start_ts, end_ts))
            
            for row in results:
                flight_id, timestamp, report_json = row
                try:
                    report = json.loads(report_json) if isinstance(report_json, str) else report_json
                    
                    # Get flight path centroid
                    flight_path = report.get('summary', {}).get('flight_path', [])
                    if flight_path:
                        avg_lon = sum(p[0] for p in flight_path) / len(flight_path)
                        avg_lat = sum(p[1] for p in flight_path) / len(flight_path)
                        
                        # Grid to 0.5 degree cells
                        grid_key = (round(avg_lat * 2) / 2, round(avg_lon * 2) / 2)
                        
                        location_clusters[grid_key].append({
                            'flight_id': flight_id,
                            'timestamp': timestamp,
                            'rules': [r.get('id') for r in report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])]
                        })
                except:
                    pass
        
        # Find clusters with min_occurrences
        pattern_id = 1
        for (lat, lon), flights in location_clusters.items():
            if len(flights) >= min_occurrences:
                timestamps = [f['timestamp'] for f in flights]
                patterns.append({
                    'pattern_id': f'CLUSTER_{pattern_id}',
                    'description': f'Anomaly cluster at {lat}°N, {lon}°E',
                    'location': {'lat': lat, 'lon': lon},
                    'flights': [f['flight_id'] for f in flights],
                    'first_seen': min(timestamps),
                    'last_seen': max(timestamps),
                    'occurrence_count': len(flights),
                    'risk_level': 'High' if len(flights) >= 5 else 'Medium'
                })
                pattern_id += 1
        
        return sorted(patterns, key=lambda x: x['occurrence_count'], reverse=True)

