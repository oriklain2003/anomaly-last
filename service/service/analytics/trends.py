"""
Trend analysis and comparative analytics for Level 2 insights.

Provides:
- Airline efficiency comparisons
- Holding pattern analysis
- Alternate airport behavior
- Seasonal trends
"""
from __future__ import annotations

import json
import math
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path

# Load airport data from config
def _load_airports() -> List[Dict[str, Any]]:
    """Load airport data from rule_config.json."""
    config_path = Path(__file__).parent.parent.parent / 'rules' / 'rule_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('airports', [])
    return []

AIRPORTS = _load_airports()


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


def _find_nearest_airport(lat: float, lon: float) -> Tuple[Optional[str], float]:
    """Find nearest airport to given coordinates.
    
    Returns:
        (airport_code, distance_nm) or (None, inf) if no airports loaded
    """
    best_code = None
    best_distance = float('inf')
    
    for airport in AIRPORTS:
        distance = _haversine_nm(lat, lon, airport['lat'], airport['lon'])
        if distance < best_distance:
            best_distance = distance
            best_code = airport['code']
    
    return best_code, best_distance


class TrendsAnalyzer:
    """Analyzer for operational trends and comparative insights."""
    
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
    
    def get_airline_efficiency(self, start_ts: int = 0, end_ts: int = 0, route: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Compare airline efficiency by analyzing holding patterns (360-degree turns) per airline.
        
        Holding time is extracted from Rule 3 events in anomaly reports.
        Flight duration is calculated from track timestamps.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            route: Optional route filter (e.g., "LLBG-EGLL")
        
        Returns:
            [{airline, avg_flight_time_min, avg_holding_time_min, sample_count}]
        """
        # If no timestamps provided, use last 30 days
        if not start_ts or not end_ts:
            import time
            end_ts = int(time.time())
            start_ts = end_ts - (30 * 24 * 60 * 60)
        
        # First, get flight durations by airline from track data
        airline_flight_durations = self._get_flight_durations_by_airline(start_ts, end_ts)
        
        # Query anomaly reports to get holding patterns per airline
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Aggregate holding time by airline (extracted from callsign prefix)
        airline_stats = defaultdict(lambda: {'total_holding_s': 0, 'flights': set(), 'holding_events': 0})
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                
                # Get callsign from summary
                callsign = report.get('summary', {}).get('callsign', '')
                if not callsign or len(callsign) < 2:
                    continue
                    
                # Extract airline code (first 2-3 letters)
                airline = ''.join(c for c in callsign[:3] if c.isalpha()).upper()
                if not airline:
                    continue
                
                flight_id = report.get('summary', {}).get('flight_id', '')
                
                # Look for Rule 3 (holding patterns / 360 turns)
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                for rule in matched_rules:
                    if rule.get('id') == 3:
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            duration_s = event.get('duration_s', 0)
                            if duration_s > 0:
                                airline_stats[airline]['total_holding_s'] += duration_s
                                airline_stats[airline]['flights'].add(flight_id)
                                airline_stats[airline]['holding_events'] += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Convert to output format
        result = []
        for airline, data in airline_stats.items():
            if data['holding_events'] > 0:
                avg_holding_min = (data['total_holding_s'] / data['holding_events']) / 60
                # Get avg flight time from pre-calculated durations
                avg_flight_time = airline_flight_durations.get(airline, {}).get('avg_duration_min', 0)
                result.append({
                    'airline': airline,
                    'avg_flight_time_min': round(avg_flight_time, 1),
                    'avg_holding_time_min': round(avg_holding_min, 1),
                    'sample_count': len(data['flights']),
                    'total_holding_events': data['holding_events']
                })
        
        # Sort by sample count descending
        result.sort(key=lambda x: x['sample_count'], reverse=True)
        
        # Return top 10 airlines
        return result[:10] if result else []
    
    def _get_flight_durations_by_airline(self, start_ts: int, end_ts: int) -> Dict[str, Dict[str, float]]:
        """
        Calculate average flight duration by airline from track data.
        
        Returns:
            {airline_code: {'avg_duration_min': float, 'flight_count': int}}
        """
        airline_durations = defaultdict(lambda: {'total_duration_s': 0, 'flight_count': 0})
        
        # Query flight durations from track tables
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT callsign, 
                       MIN(timestamp) as start_time, 
                       MAX(timestamp) as end_time
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                  AND callsign IS NOT NULL
                  AND callsign != ''
                GROUP BY flight_id
                HAVING (MAX(timestamp) - MIN(timestamp)) > 300  -- At least 5 min flight
                   AND (MAX(timestamp) - MIN(timestamp)) < 86400  -- Less than 24 hours
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            
            for row in results:
                callsign, start_time, end_time = row
                if not callsign or len(callsign) < 2:
                    continue
                
                # Extract airline code (first 2-3 letters)
                airline = ''.join(c for c in callsign[:3] if c.isalpha()).upper()
                if not airline:
                    continue
                
                duration_s = end_time - start_time
                airline_durations[airline]['total_duration_s'] += duration_s
                airline_durations[airline]['flight_count'] += 1
        
        # Calculate averages
        result = {}
        for airline, data in airline_durations.items():
            if data['flight_count'] > 0:
                avg_duration_min = (data['total_duration_s'] / data['flight_count']) / 60
                result[airline] = {
                    'avg_duration_min': avg_duration_min,
                    'flight_count': data['flight_count']
                }
        
        return result
    
    def get_holding_pattern_analysis(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Analyze holding patterns and estimate costs.
        
        Holding patterns are detected from Rule 3 (360-degree turns) in anomaly reports.
        Each holding event has a duration_s field that we aggregate.
        Events are attributed to nearest airport based on event coordinates.
        
        Returns:
            {
                total_time_hours: float,
                estimated_fuel_cost_usd: float,
                peak_hours: [int],
                events_by_airport: {airport_code: count}
            }
        """
        from datetime import datetime
        
        # Query for holding pattern detections (Rule 3 - abrupt turns/360s)
        # Query research.db which has the anomaly reports
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        total_holding_minutes = 0
        hour_distribution = defaultdict(int)
        airport_events = defaultdict(int)
        
        # Track flight IDs to get their track data for airport lookup
        flight_holding_events = []  # [(flight_id, timestamp, duration_s)]
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                flight_id = report.get('summary', {}).get('flight_id', '')
                
                for rule in matched_rules:
                    # Look for holding patterns (360-degree turns)
                    if rule.get('id') == 3:
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            # Estimate holding time (simplified)
                            duration_s = event.get('duration_s', 300)  # Default 5 min
                            total_holding_minutes += duration_s / 60
                            
                            # Track by hour
                            hour = datetime.fromtimestamp(timestamp).hour
                            hour_distribution[hour] += 1
                            
                            # Get event timestamp for airport lookup
                            event_ts = event.get('timestamp', timestamp)
                            flight_holding_events.append((flight_id, event_ts, duration_s))
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Get airport attribution for holding events by looking up track positions
        if flight_holding_events:
            airport_events = self._attribute_events_to_airports(flight_holding_events, start_ts, end_ts)
        
        # Estimate fuel cost: ~$3/min of holding (rough aviation estimate)
        fuel_cost = total_holding_minutes * 3
        
        # Find peak hours
        peak_hours = sorted(hour_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'total_time_hours': round(total_holding_minutes / 60, 2),
            'estimated_fuel_cost_usd': int(fuel_cost),
            'peak_hours': [h for h, _ in peak_hours],
            'events_by_airport': dict(airport_events)
        }
    
    def _attribute_events_to_airports(self, events: List[Tuple[str, int, int]], 
                                       start_ts: int, end_ts: int) -> Dict[str, int]:
        """
        Attribute holding events to nearest airports based on track positions.
        
        Args:
            events: List of (flight_id, event_timestamp, duration_s)
            start_ts, end_ts: Time range for track queries
            
        Returns:
            {airport_code: event_count}
        """
        airport_counts = defaultdict(int)
        
        # Get unique flight IDs
        flight_ids = list(set(e[0] for e in events if e[0]))
        if not flight_ids:
            return airport_counts
        
        # Build a mapping of flight_id -> [(timestamp, lat, lon)]
        flight_positions = defaultdict(list)
        
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            # Query in batches to avoid too many parameters
            for i in range(0, len(flight_ids), 50):
                batch = flight_ids[i:i+50]
                placeholders = ','.join(['?' for _ in batch])
                query = f"""
                    SELECT flight_id, timestamp, lat, lon
                    FROM {table_name}
                    WHERE flight_id IN ({placeholders})
                      AND timestamp BETWEEN ? AND ?
                      AND lat IS NOT NULL AND lon IS NOT NULL
                    ORDER BY flight_id, timestamp
                """
                params = tuple(batch) + (start_ts, end_ts)
                results = self._execute_query('research', query, params)
                
                for row in results:
                    fid, ts, lat, lon = row
                    flight_positions[fid].append((ts, lat, lon))
        
        # For each event, find the nearest airport based on position at event time
        for flight_id, event_ts, _ in events:
            if not flight_id or flight_id not in flight_positions:
                continue
            
            positions = flight_positions[flight_id]
            if not positions:
                continue
            
            # Find position closest to event timestamp
            closest_pos = min(positions, key=lambda p: abs(p[0] - event_ts))
            _, lat, lon = closest_pos
            
            # Find nearest airport
            airport_code, distance = _find_nearest_airport(lat, lon)
            if airport_code and distance < 100:  # Within 100nm of an airport
                airport_counts[airport_code] += 1
        
        return airport_counts
    
    def get_alternate_airports(self, airport: str, event_date: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Analyze where flights divert when primary airport is unavailable.
        
        Queries Rule 8 (diversion) matches from anomaly reports to find:
        - Which airports flights diverted TO when they couldn't land at the primary
        - What aircraft types were involved
        
        Args:
            airport: Primary airport code (e.g., "LLBG") - flights planned for this airport
            event_date: Optional timestamp to filter by date
            
        Returns:
            [{alternate_airport, count, aircraft_types: [...]}]
        """
        import time
        
        # Default time range: last 90 days if no date specified
        if event_date:
            start_ts = event_date - (24 * 60 * 60)  # One day before
            end_ts = event_date + (24 * 60 * 60)    # One day after
        else:
            end_ts = int(time.time())
            start_ts = end_ts - (90 * 24 * 60 * 60)  # Last 90 days
        
        # Query anomaly reports for Rule 8 (diversion) matches
        query = """
            SELECT full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Aggregate by alternate airport
        alternate_stats = defaultdict(lambda: {'count': 0, 'aircraft_types': set()})
        
        for row in results:
            report_json = row[0]
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                callsign = report.get('summary', {}).get('callsign', '')
                
                for rule in matched_rules:
                    if rule.get('id') == 8:  # Diversion rule
                        details = rule.get('details', {})
                        planned = details.get('planned', '')
                        actual = details.get('actual', '')
                        
                        # Filter by primary airport if specified
                        if airport and planned.upper() != airport.upper():
                            continue
                        
                        # Skip if landed at planned destination (no diversion)
                        if not actual or actual == planned:
                            continue
                        
                        alternate_stats[actual]['count'] += 1
                        
                        # Try to extract aircraft type from callsign
                        # Common patterns: first 2-3 chars are airline, rest is flight number
                        if callsign:
                            # Look for aircraft type in the report metadata
                            aircraft_type = self._extract_aircraft_type(report)
                            if aircraft_type:
                                alternate_stats[actual]['aircraft_types'].add(aircraft_type)
                                
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Convert to output format
        result = []
        for alt_airport, data in alternate_stats.items():
            result.append({
                'alternate_airport': alt_airport,
                'count': data['count'],
                'aircraft_types': list(data['aircraft_types'])[:5]  # Limit to 5 types
            })
        
        # Sort by count descending
        result.sort(key=lambda x: x['count'], reverse=True)
        
        return result[:10] if result else []
    
    def _extract_aircraft_type(self, report: Dict[str, Any]) -> Optional[str]:
        """
        Extract aircraft type from anomaly report.
        
        Looks for aircraft type in various report fields.
        """
        # Try to get from metadata if available
        metadata = report.get('metadata', {})
        if metadata:
            aircraft = metadata.get('aircraft_type') or metadata.get('aircraft')
            if aircraft:
                return aircraft
        
        # Try to infer from callsign pattern (less reliable)
        callsign = report.get('summary', {}).get('callsign', '')
        if callsign:
            # Some callsigns include aircraft type hints
            # This is a simplified heuristic
            airline_prefixes = {
                'ELY': 'B738',   # El Al typically uses 737/787
                'THY': 'A321',   # Turkish Airlines
                'MEA': 'A320',   # Middle East Airlines
                'UAE': 'B77W',   # Emirates
                'RJA': 'E190',   # Royal Jordanian
            }
            prefix = callsign[:3].upper()
            if prefix in airline_prefixes:
                return airline_prefixes[prefix]
        
        return None
    
    def get_monthly_trends(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get monthly aggregated trends for flights and anomalies.
        
        Returns:
            [{month, total_flights, anomalies, safety_events, busiest_hour}]
        """
        from datetime import datetime
        
        monthly_data = {}
        
        # Aggregate flights by month from research.db tables
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT 
                    strftime('%Y-%m', datetime(timestamp, 'unixepoch')) as month,
                    COUNT(DISTINCT flight_id) as flight_count,
                    strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                    COUNT(*) as hour_count
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY month, hour
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            
            for row in results:
                month, flight_count, hour, hour_count = row
                if month not in monthly_data:
                    monthly_data[month] = {
                        'month': month,
                        'total_flights': 0,
                        'anomalies': 0,
                        'safety_events': 0,
                        'hour_counts': {}
                    }
                monthly_data[month]['total_flights'] += flight_count or 0
                monthly_data[month]['hour_counts'][hour] = monthly_data[month]['hour_counts'].get(hour, 0) + (hour_count or 0)
        
        # Convert to list and find busiest hour per month
        result = []
        for month, data in sorted(monthly_data.items()):
            busiest_hour = max(data['hour_counts'].items(), key=lambda x: x[1])[0] if data['hour_counts'] else 0
            result.append({
                'month': month,
                'total_flights': data['total_flights'],
                'anomalies': data['anomalies'],
                'safety_events': data['safety_events'],
                'busiest_hour': int(busiest_hour)
            })
        
        return result
    
    def get_peak_hours_analysis(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Analyze peak traffic hours and their correlation with safety events.
        
        Returns:
            {
                'peak_hours': [hours],
                'traffic_by_hour': {hour: count},
                'safety_by_hour': {hour: count},
                'correlation_score': float
            }
        """
        from collections import defaultdict
        
        traffic_by_hour = defaultdict(int)
        safety_by_hour = defaultdict(int)
        
        # Count traffic by hour from research.db tables
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT strftime('%H', datetime(timestamp, 'unixepoch')) as hour, 
                       COUNT(*) as count
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY hour
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            for row in results:
                hour, count = row
                if hour:
                    traffic_by_hour[int(hour)] += count or 0
        
        # Count safety events by hour from research.db only (distinct flights per hour)
        query = """
            SELECT strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                   COUNT(DISTINCT json_extract(full_report, '$.summary.flight_id')) as count
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY hour
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        for row in results:
            hour, count = row
            if hour:
                safety_by_hour[int(hour)] += count or 0
        
        # Find peak hours (top 3)
        peak_hours = sorted(traffic_by_hour.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_hours = [h[0] for h in peak_hours]
        
        # Calculate Pearson correlation between traffic and safety events by hour
        correlation_score = self._calculate_pearson_correlation(traffic_by_hour, safety_by_hour)
        
        return {
            'peak_hours': peak_hours,
            'traffic_by_hour': dict(traffic_by_hour),
            'safety_by_hour': dict(safety_by_hour),
            'correlation_score': correlation_score
        }
    
    def _calculate_pearson_correlation(self, data1: Dict[int, int], data2: Dict[int, int]) -> float:
        """
        Calculate Pearson correlation coefficient between two hourly distributions.
        
        Returns:
            Correlation coefficient between -1 and 1, or 0 if insufficient data.
        """
        import math
        
        # Create aligned vectors for all 24 hours
        hours = range(24)
        x = [data1.get(h, 0) for h in hours]
        y = [data2.get(h, 0) for h in hours]
        
        n = len(x)
        if n == 0:
            return 0.0
        
        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Calculate covariance and standard deviations
        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        
        # Avoid division by zero
        if std_x == 0 or std_y == 0:
            return 0.0
        
        correlation = covariance / (std_x * std_y)
        return round(correlation, 3)
    
    def get_alternate_airports_by_time(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get all alternate airports used during diversions in a time period.
        
        Unlike get_alternate_airports() which filters by primary airport, this returns
        ALL diversions in the time range.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            
        Returns:
            [{airport, count, aircraft_types: [...], last_used: timestamp}]
        """
        # Query anomaly reports for Rule 8 (diversion) matches
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Aggregate by alternate airport
        alternate_stats = defaultdict(lambda: {'count': 0, 'aircraft_types': set(), 'last_used': 0})
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                callsign = report.get('summary', {}).get('callsign', '')
                
                for rule in matched_rules:
                    if rule.get('id') == 8:  # Diversion rule
                        details = rule.get('details', {})
                        actual = details.get('actual', '')
                        planned = details.get('planned', '')
                        
                        # Skip if no actual landing airport or same as planned
                        if not actual or actual == planned:
                            continue
                        
                        alternate_stats[actual]['count'] += 1
                        alternate_stats[actual]['last_used'] = max(
                            alternate_stats[actual]['last_used'], 
                            timestamp
                        )
                        
                        # Try to extract aircraft type
                        aircraft_type = self._extract_aircraft_type(report)
                        if aircraft_type:
                            alternate_stats[actual]['aircraft_types'].add(aircraft_type)
                            
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Convert to output format
        result = []
        for airport, data in alternate_stats.items():
            result.append({
                'airport': airport,
                'count': data['count'],
                'aircraft_types': list(data['aircraft_types'])[:5],
                'last_used': data['last_used']
            })
        
        # Sort by count descending
        result.sort(key=lambda x: x['count'], reverse=True)
        
        return result[:15] if result else []

