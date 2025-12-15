"""
Statistics aggregation engine for Level 1 analytics.

Provides methods to compute statistics from flight data including:
- Safety events (emergency codes, near-miss, go-arounds)
- Traffic statistics (flights/day, busiest routes, military tracking)
- Signal loss analysis

Includes caching layer for expensive queries with 1-hour expiry.
"""
from __future__ import annotations

import json
import sqlite3
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter

from .queries import QueryBuilder


# In-memory cache for expensive queries
_stats_cache: Dict[str, Dict[str, Any]] = {}
CACHE_EXPIRY_SECONDS = 3600  # 1 hour


def _get_cache_key(method_name: str, *args) -> str:
    """Generate a unique cache key for a method call."""
    key_data = f"{method_name}:{':'.join(str(a) for a in args)}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _get_cached(cache_key: str) -> Optional[Any]:
    """Get cached result if not expired."""
    if cache_key in _stats_cache:
        entry = _stats_cache[cache_key]
        if time.time() - entry['timestamp'] < CACHE_EXPIRY_SECONDS:
            return entry['data']
        else:
            # Expired, remove it
            del _stats_cache[cache_key]
    return None


def _set_cached(cache_key: str, data: Any) -> None:
    """Store result in cache."""
    _stats_cache[cache_key] = {
        'data': data,
        'timestamp': time.time()
    }


def clear_stats_cache() -> int:
    """Clear all cached statistics. Returns number of entries cleared."""
    count = len(_stats_cache)
    _stats_cache.clear()
    return count


def get_cache_info() -> Dict[str, Any]:
    """Get cache statistics."""
    now = time.time()
    valid_entries = sum(1 for e in _stats_cache.values() if now - e['timestamp'] < CACHE_EXPIRY_SECONDS)
    return {
        'total_entries': len(_stats_cache),
        'valid_entries': valid_entries,
        'expiry_seconds': CACHE_EXPIRY_SECONDS
    }


class StatisticsEngine:
    """Engine for computing flight statistics with caching."""
    
    def __init__(self, db_paths: Dict[str, Path]):
        """
        Initialize statistics engine.
        
        Args:
            db_paths: Dictionary mapping db names to paths
                     e.g., {'live': Path('realtime/live_tracks.db'),
                            'research': Path('research.db'),
                            'anomalies': Path('realtime/live_anomalies.db')}
        """
        self.db_paths = db_paths
        self.query_builder = QueryBuilder()
    
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
    
    def _is_military_callsign(self, callsign: str) -> bool:
        """Detect if callsign indicates military aircraft."""
        if not callsign:
            return False
        
        # Common military callsign patterns
        military_prefixes = [
            'RCH', 'REACH',  # US Air Force
            'RRR',  # Russian Air Force
            'RFF', 'RAF',  # Royal Air Force
            'IAF',  # Israeli Air Force
            'CNV', 'CONVOY',
            'EVAC',
            'DUKE', 'KING', 'VIPER', 'HAWK', 'EAGLE',
            'N00',  # US Navy
        ]
        
        callsign_upper = callsign.upper()
        for prefix in military_prefixes:
            if callsign_upper.startswith(prefix):
                return True
        
        # Callsigns without typical 3-letter airline prefix
        if len(callsign) <= 4 and not any(c.isdigit() for c in callsign[:3]):
            return False  # Likely civilian
        
        # Numeric-heavy callsigns often military
        if len(callsign) > 5 and sum(c.isdigit() for c in callsign) > 3:
            return True
            
        return False
    
    def get_overview_stats(self, start_ts: int, end_ts: int, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get overview statistics for the dashboard.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            use_cache: Whether to use cached results (default True)
        
        Returns:
            {
                'total_flights': int,
                'total_anomalies': int,
                'safety_events': int,
                'go_arounds': int,
                'emergency_codes': int,
                'near_miss': int
            }
        """
        # Check cache first
        cache_key = _get_cache_key('get_overview_stats', start_ts, end_ts)
        if use_cache:
            cached = _get_cached(cache_key)
            if cached is not None:
                return cached
        
        stats = {
            'total_flights': 0,
            'total_anomalies': 0,
            'safety_events': 0,
            'go_arounds': 0,
            'emergency_codes': 0,
            'near_miss': 0
        }
        
        # Count total flights - optimized query using timestamp index
        # Simply count distinct flight_ids that have ANY point in the time range
        # This is much faster than computing min/max per flight
        
        # Query research.db tables (anomalies_tracks and normal_tracks)
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT COUNT(DISTINCT flight_id)
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            if results:
                stats['total_flights'] += results[0][0] or 0
        
        # Also check live_tracks.db for real-time data
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM flight_tracks
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('live', query, (start_ts, end_ts))
        if results:
            stats['total_flights'] += results[0][0] or 0
        
        # Count unique flights with anomalies from anomalies_tracks table
        # This ensures we only count flights that have actual track data
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM anomalies_tracks
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        stats['total_anomalies'] = results[0][0] if results else 0
        
        # Count specific event types by parsing anomaly reports from research db
        # Use sets to track distinct flights per event type
        emergency_flights = set()
        near_miss_flights = set()
        go_around_flights = set()
        safety_event_flights = set()
        
        query = """
            SELECT full_report FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            try:
                report = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                flight_id = report.get('summary', {}).get('flight_id', '')
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    # Rule 1: Emergency squawk
                    if rule_id == 1 and flight_id:
                        emergency_flights.add(flight_id)
                        safety_event_flights.add(flight_id)
                    # Rule 4: Dangerous proximity (near-miss)
                    elif rule_id == 4 and flight_id:
                        near_miss_flights.add(flight_id)
                        safety_event_flights.add(flight_id)
                    # Rule 6: Go-around
                    elif rule_id == 6 and flight_id:
                        go_around_flights.add(flight_id)
                        safety_event_flights.add(flight_id)
            except (json.JSONDecodeError, KeyError):
                continue
        
        stats['emergency_codes'] = len(emergency_flights)
        stats['near_miss'] = len(near_miss_flights)
        stats['go_arounds'] = len(go_around_flights)
        stats['safety_events'] = len(safety_event_flights)
        
        # Cache the results
        _set_cached(cache_key, stats)
        return stats
    
    def get_emergency_codes_stats(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get emergency code statistics broken down by code and airline.
        
        Returns:
            [{code, count, airline, flights: [...]}]
        """
        query = """
            SELECT flight_id, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Group by code, track distinct flights per code
        code_stats = defaultdict(lambda: {'flights': set(), 'airlines': Counter()})
        
        for row in results:
            flight_id, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                # Get callsign from the JSON report
                callsign = report.get('summary', {}).get('callsign', '')
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 1:  # Emergency squawk rule
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            code = event.get('squawk', 'UNKNOWN')
                            code_stats[code]['flights'].add(flight_id)
                            
                            # Extract airline from callsign (first 3 chars typically)
                            airline = callsign[:3] if callsign else 'UNKNOWN'
                            code_stats[code]['airlines'][airline] += 1
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Format output with distinct flight counts
        result = []
        for code, data in code_stats.items():
            result.append({
                'code': code,
                'count': len(data['flights']),  # Distinct flights
                'airlines': dict(data['airlines']),
                'flights': list(data['flights'])[:10]  # Limit to 10 examples
            })
        
        return sorted(result, key=lambda x: x['count'], reverse=True)
    
    def get_near_miss_events(self, start_ts: int, end_ts: int, 
                            severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get near-miss (proximity) events.
        
        Args:
            severity: 'high' for < 1nm/1000ft, None for all
        
        Returns:
            [{timestamp, flight_id, other_flight_id, distance_nm, altitude_diff_ft, severity}]
        """
        query = """
            SELECT timestamp, flight_id, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        events = []
        for row in results:
            timestamp, flight_id, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 4:  # Proximity rule
                        rule_events = rule.get('details', {}).get('events', [])
                        for event in rule_events:
                            distance_nm = event.get('distance_nm', 999)
                            alt_diff = event.get('altitude_diff_ft', 9999)
                            
                            # Determine severity
                            event_severity = 'high' if (distance_nm < 1 and alt_diff < 1000) else 'medium'
                            
                            if severity and event_severity != severity:
                                continue
                            
                            events.append({
                                'timestamp': timestamp,
                                'flight_id': flight_id,
                                'other_flight_id': event.get('other_flight', 'UNKNOWN'),
                                'distance_nm': round(distance_nm, 2),
                                'altitude_diff_ft': int(alt_diff),
                                'severity': event_severity
                            })
            except (json.JSONDecodeError, KeyError):
                continue
        
        return sorted(events, key=lambda x: x['timestamp'], reverse=True)
    
    def get_go_around_stats(self, start_ts: int, end_ts: int, 
                           airport: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get go-around statistics.
        
        Returns:
            [{airport, count, avg_per_day, by_hour: {...}}]
        """
        # Query both anomalies db and research db
        all_results = []
        for db_name in ['anomalies', 'research']:
            query = """
                SELECT timestamp, full_report
                FROM anomaly_reports
                WHERE timestamp BETWEEN ? AND ?
            """
            results = self._execute_query(db_name, query, (start_ts, end_ts))
            all_results.extend(results)
        
        # Group by airport
        airport_stats = defaultdict(lambda: {'count': 0, 'by_hour': Counter()})
        results = all_results  # Use combined results
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 6:  # Go-around rule
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            apt = event.get('airport', 'UNKNOWN')
                            if airport and apt != airport:
                                continue
                            
                            airport_stats[apt]['count'] += 1
                            
                            # Extract hour from timestamp
                            hour = datetime.fromtimestamp(timestamp).hour
                            airport_stats[apt]['by_hour'][hour] += 1
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Calculate avg per day
        days = (end_ts - start_ts) / 86400
        result = []
        for apt, data in airport_stats.items():
            result.append({
                'airport': apt,
                'count': data['count'],
                'avg_per_day': round(data['count'] / max(days, 1), 2),
                'by_hour': dict(data['by_hour'])
            })
        
        return sorted(result, key=lambda x: x['count'], reverse=True)
    
    def get_flights_per_day(self, start_ts: int, end_ts: int, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get flight counts per day.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            use_cache: Whether to use cached results (default True)
        
        Returns:
            [{date, count, military_count, civilian_count}]
        """
        # Check cache first
        cache_key = _get_cache_key('get_flights_per_day', start_ts, end_ts)
        if use_cache:
            cached = _get_cached(cache_key)
            if cached is not None:
                return cached
        
        # Aggregate from all databases
        daily_counts = defaultdict(lambda: {'total': 0, 'military': 0, 'civilian': 0})
        
        # Query research.db tables (anomalies_tracks and normal_tracks)
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT 
                    DATE(timestamp, 'unixepoch') as date,
                    callsign,
                    COUNT(DISTINCT flight_id) as count
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY date, callsign
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            
            for row in results:
                date, callsign, count = row
                if not date:
                    continue
                    
                daily_counts[date]['total'] += count
                
                # Simple heuristic: military callsigns often lack typical airline codes
                is_military = self._is_military_callsign(callsign) if callsign else False
                if is_military:
                    daily_counts[date]['military'] += count
                else:
                    daily_counts[date]['civilian'] += count
        
        # Also query live_tracks.db
        query = """
            SELECT 
                DATE(timestamp, 'unixepoch') as date,
                callsign,
                COUNT(DISTINCT flight_id) as count
            FROM flight_tracks
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY date, callsign
        """
        results = self._execute_query('live', query, (start_ts, end_ts))
        
        for row in results:
            date, callsign, count = row
            if not date:
                continue
                
            daily_counts[date]['total'] += count
            
            # Simple heuristic: military callsigns
            is_military = self._is_military_callsign(callsign) if callsign else False
            if is_military:
                daily_counts[date]['military'] += count
            else:
                daily_counts[date]['civilian'] += count
        
        result = [
            {
                'date': date,
                'count': data['total'],
                'military_count': data['military'],
                'civilian_count': data['civilian']
            }
            for date, data in sorted(daily_counts.items())
        ]
        
        # Cache the results
        _set_cached(cache_key, result)
        return result
    
    def get_busiest_airports(self, start_ts: int, end_ts: int, 
                            limit: int = 10, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get busiest airports by detecting arrivals/departures from track data.
        
        Returns:
            [{airport, arrivals, departures, total}]
        """
        # Check cache first
        cache_key = _get_cache_key('get_busiest_airports', start_ts, end_ts, limit)
        if use_cache:
            cached = _get_cached(cache_key)
            if cached is not None:
                return cached
        
        # Airport coordinates and proximity detection
        airports = {
            'LLBG': {'lat': 32.0114, 'lon': 34.8867, 'name': 'Ben Gurion'},
            'LLER': {'lat': 29.9403, 'lon': 35.0004, 'name': 'Ramon'},
            'LLSD': {'lat': 32.1147, 'lon': 34.7822, 'name': 'Sde Dov'},
            'LLHA': {'lat': 32.8094, 'lon': 35.0431, 'name': 'Haifa'},
            'LCLK': {'lat': 34.8750, 'lon': 33.6249, 'name': 'Larnaca'},
            'OLBA': {'lat': 33.8209, 'lon': 35.4884, 'name': 'Beirut'},
            'OJAI': {'lat': 31.7228, 'lon': 35.9932, 'name': 'Amman'}
        }
        
        airport_stats = {icao: {'arrivals': 0, 'departures': 0} for icao in airports}
        proximity_nm = 5  # Within 5nm of airport
        
        # Query low altitude flights near airports - detect arrivals AND departures
        # Optimized: use a single query per airport that gets first/last altitude in one go
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            for icao, coords in airports.items():
                # Convert nm to degrees (roughly)
                lat_range = proximity_nm / 60
                lon_range = proximity_nm / (60 * abs(coords['lat'] / 57.3) if coords['lat'] != 0 else 60)
                
                # Get first and last altitude near airport in a single optimized query
                query = f"""
                    WITH airport_points AS (
                        SELECT flight_id, timestamp, alt,
                               ROW_NUMBER() OVER (PARTITION BY flight_id ORDER BY timestamp ASC) as rn_first,
                               ROW_NUMBER() OVER (PARTITION BY flight_id ORDER BY timestamp DESC) as rn_last
                        FROM {table_name}
                        WHERE timestamp BETWEEN ? AND ?
                          AND lat BETWEEN ? AND ?
                          AND lon BETWEEN ? AND ?
                          AND alt < 5000
                    )
                    SELECT 
                        flight_id,
                        MAX(CASE WHEN rn_first = 1 THEN alt END) as first_alt,
                        MAX(CASE WHEN rn_last = 1 THEN alt END) as last_alt
                    FROM airport_points
                    WHERE rn_first = 1 OR rn_last = 1
                    GROUP BY flight_id
                    HAVING MIN(CASE WHEN rn_first = 1 THEN alt END) < 2000 
                        OR MIN(CASE WHEN rn_last = 1 THEN alt END) < 2000
                """
                params = (
                    start_ts, end_ts,
                    coords['lat'] - lat_range, coords['lat'] + lat_range,
                    coords['lon'] - lon_range, coords['lon'] + lon_range
                )
                results = self._execute_query('research', query, params)
                
                for row in results:
                    flight_id, first_alt, last_alt = row
                    if first_alt is not None and last_alt is not None:
                        # Descending = arrival, Ascending = departure
                        if first_alt > last_alt + 500:  # Descending by 500ft+
                            airport_stats[icao]['arrivals'] += 1
                        elif last_alt > first_alt + 500:  # Climbing by 500ft+
                            airport_stats[icao]['departures'] += 1
                        elif first_alt < 1000:
                            # Started low near airport = departure
                            airport_stats[icao]['departures'] += 1
                        elif last_alt < 1000:
                            # Ended low near airport = arrival
                            airport_stats[icao]['arrivals'] += 1
        
        # Format output
        result = []
        for icao, stats in airport_stats.items():
            total = stats['arrivals'] + stats['departures']
            if total > 0:
                result.append({
                    'airport': icao,
                    'name': airports[icao]['name'],
                    'arrivals': stats['arrivals'],
                    'departures': stats['departures'],
                    'total': total
                })
        
        sorted_result = sorted(result, key=lambda x: x['total'], reverse=True)[:limit]
        
        # Cache the results
        _set_cached(cache_key, sorted_result)
        return sorted_result
    
    def get_runway_usage(self, airport: str, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get runway usage statistics by inferring runway from approach/departure heading.
        
        Detects both landings (descending to low altitude) and takeoffs (climbing from low altitude).
        
        Returns:
            [{runway, landings, takeoffs, total}]
        """
        # Runway definitions for Israeli airports
        # Format: runway_name: (min_heading, max_heading)
        runway_configs = {
            'LLBG': {  # Ben Gurion International
                '03': (20, 40),   # Heading 030 +/- 10
                '21': (200, 220),
                '08': (70, 90),   # Heading 080 +/- 10
                '26': (250, 270),
                '12': (110, 130), # Heading 120 +/- 10
                '30': (290, 310)
            },
            'LLER': {  # Ramon International (Eilat)
                '03': (20, 40),
                '21': (200, 220)
            },
            'LLHA': {  # Haifa Airport
                '16': (150, 170),
                '34': (330, 350)
            },
            'LLOV': {  # Ovda (military/civilian)
                '02': (10, 30),
                '20': (190, 210)
            },
            'LLSD': {  # Sde Dov (now closed, but historical data)
                '08': (70, 90),
                '26': (250, 270)
            },
            'LLET': {  # Eilat (old airport)
                '01': (0, 20),
                '19': (180, 200)
            },
            'LLRD': {  # Rosh Pina
                '15': (140, 160),
                '33': (320, 340)
            },
            'LLMZ': {  # Mitzpe Ramon (Airstrip)
                '03': (20, 40),
                '21': (200, 220)
            }
        }
        
        if airport not in runway_configs:
            return []
        
        runways = runway_configs[airport]
        runway_stats = {rwy: {'landings': 0, 'takeoffs': 0} for rwy in runways}
        
        # Airport coordinates for all Israeli airports
        airport_coords = {
            'LLBG': {'lat': 32.0114, 'lon': 34.8867},  # Ben Gurion
            'LLER': {'lat': 29.7255, 'lon': 35.0119},  # Ramon
            'LLHA': {'lat': 32.8094, 'lon': 35.0431},  # Haifa
            'LLOV': {'lat': 29.9403, 'lon': 34.9358},  # Ovda
            'LLSD': {'lat': 32.1147, 'lon': 34.7822},  # Sde Dov
            'LLET': {'lat': 29.5613, 'lon': 34.9601},  # Eilat old
            'LLRD': {'lat': 32.9810, 'lon': 35.5718},  # Rosh Pina
            'LLMZ': {'lat': 30.7761, 'lon': 34.8067}   # Mitzpe Ramon
        }
        
        if airport not in airport_coords:
            return []
        
        coords = airport_coords[airport]
        proximity_nm = 10
        lat_range = proximity_nm / 60
        lon_range = proximity_nm / 60
        
        # Query low altitude flights with heading and altitude
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT flight_id, track, alt, timestamp, gspeed
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                  AND lat BETWEEN ? AND ?
                  AND lon BETWEEN ? AND ?
                  AND alt BETWEEN 0 AND 5000
                ORDER BY flight_id, timestamp
            """
            params = (
                start_ts, end_ts,
                coords['lat'] - lat_range, coords['lat'] + lat_range,
                coords['lon'] - lon_range, coords['lon'] + lon_range
            )
            results = self._execute_query('research', query, params)
            
            # Group by flight
            flight_data = defaultdict(list)
            for row in results:
                flight_id, heading, alt, ts, gspeed = row
                if heading is not None and alt is not None:
                    flight_data[flight_id].append({
                        'ts': ts, 
                        'heading': heading, 
                        'alt': alt,
                        'gspeed': gspeed or 0
                    })
            
            for flight_id, points in flight_data.items():
                if len(points) < 3:
                    continue
                
                # Sort by timestamp
                points.sort(key=lambda x: x['ts'])
                
                # Determine if landing or takeoff by altitude trend
                first_alt = points[0]['alt']
                last_alt = points[-1]['alt']
                min_alt = min(p['alt'] for p in points)
                
                # Find the point at minimum altitude
                min_alt_point = min(points, key=lambda x: x['alt'])
                
                is_landing = False
                is_takeoff = False
                
                # Landing: descending trend, ends at low altitude
                if first_alt > last_alt and min_alt < 1500:
                    is_landing = True
                    heading_at_runway = min_alt_point['heading']
                
                # Takeoff: ascending trend, starts at low altitude  
                elif first_alt < last_alt and min_alt < 1500:
                    is_takeoff = True
                    heading_at_runway = min_alt_point['heading']
                
                # Match to runway
                if is_landing or is_takeoff:
                    for runway, (min_hdg, max_hdg) in runways.items():
                        if min_hdg <= heading_at_runway <= max_hdg:
                            if is_landing:
                                runway_stats[runway]['landings'] += 1
                            else:
                                runway_stats[runway]['takeoffs'] += 1
                            break
        
        # Format output
        result = []
        for runway, stats in runway_stats.items():
            total = stats['landings'] + stats['takeoffs']
            result.append({
                'runway': runway,
                'airport': airport,
                'landings': stats['landings'],
                'takeoffs': stats['takeoffs'],
                'total': total
            })
        
        return sorted(result, key=lambda x: x['total'], reverse=True)
    
    def get_signal_loss_locations(self, start_ts: int, end_ts: int, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get geographic distribution of signal loss events by detecting gaps in track data.
        
        Algorithm:
        1. Query track points ordered by flight_id and timestamp
        2. Detect gaps where next_timestamp - current_timestamp > 5 minutes
        3. Record gap location (lat/lon of last known position)
        4. Aggregate by geographic grid (1.0 degrees ~111km for broader zones)
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            limit: Maximum number of zones to return (default 50, prevents browser overload)
        
        Returns:
            [{lat, lon, count, avgDuration, intensity, affected_flights}]
        """
        # Grid-based aggregation (1.0 degree cells ~111km for broader zones)
        grid_size = 1.0
        gap_threshold_seconds = 300  # Signal loss if gap > 5 minutes (300 seconds)
        
        location_stats = defaultdict(lambda: {
            'count': 0, 
            'total_duration': 0,
            'flights': set()
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
                        # Use the last known position as the gap location
                        grid_lat = round(prev_lat / grid_size) * grid_size
                        grid_lon = round(prev_lon / grid_size) * grid_size
                        key = (grid_lat, grid_lon)
                        
                        location_stats[key]['count'] += 1
                        location_stats[key]['total_duration'] += gap_seconds
                        location_stats[key]['flights'].add(flight_id)
                
                prev_flight_id = flight_id
                prev_timestamp = timestamp
                prev_lat = lat
                prev_lon = lon
        
        # Format output
        result = []
        max_count = max((d['count'] for d in location_stats.values()), default=1)
        
        for (lat, lon), data in location_stats.items():
            avg_duration = data['total_duration'] / max(data['count'], 1)
            intensity = min(100, int((data['count'] / max_count) * 100))
            
            result.append({
                'lat': lat,
                'lon': lon,
                'count': data['count'],
                'avgDuration': int(avg_duration),
                'intensity': intensity,
                'affected_flights': len(data['flights'])
            })
        
        # Limit to top 50 zones to prevent browser overload
        sorted_result = sorted(result, key=lambda x: x['count'], reverse=True)
        return sorted_result[:50]
    
    def get_diversion_stats(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get diversion statistics - flights that changed destination or deviated from route.
        
        Analyzes anomaly reports for:
        - Rule 8: Diversions (planned vs actual destination mismatch)
        - Rule 3: Holding patterns (360-degree turns)
        - Rule 11: Off-course deviations
        
        Returns:
            {
                'total_diversions': int,
                'total_large_deviations': int,
                'total_holding_360s': int,
                'by_airport': {airport_code: count},
                'by_airline': {airline_code: count}
            }
        """
        total_diversions = 0
        total_large_deviations = 0
        total_holding_360s = 0
        by_airport = defaultdict(int)
        by_airline = defaultdict(int)
        
        # Query anomaly reports
        query = """
            SELECT full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            report_json = row[0]
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                callsign = report.get('summary', {}).get('callsign', '')
                
                # Extract airline code from callsign
                airline = ''.join(c for c in (callsign or '')[:3] if c.isalpha()).upper() if callsign else None
                
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    details = rule.get('details', {})
                    
                    # Rule 8: Diversion
                    if rule_id == 8:
                        total_diversions += 1
                        actual_airport = details.get('actual')
                        if actual_airport:
                            by_airport[actual_airport] += 1
                        if airline:
                            by_airline[airline] += 1
                    
                    # Rule 3: Holding patterns (360-degree turns)
                    elif rule_id == 3:
                        events = details.get('events', [])
                        for event in events:
                            # Count 360-degree turns
                            turn_deg = abs(event.get('accumulated_deg', 0))
                            if turn_deg >= 300:  # Near full circle
                                total_holding_360s += 1
                    
                    # Rule 11: Off-course
                    elif rule_id == 11:
                        # Count significant deviations
                        off_path_events = details.get('off_path', [])
                        if len(off_path_events) >= 3:  # Multiple off-path points
                            total_large_deviations += 1
                            
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        return {
            'total_diversions': total_diversions,
            'total_large_deviations': total_large_deviations,
            'total_holding_360s': total_holding_360s,
            'by_airport': dict(by_airport),
            'by_airline': dict(by_airline)
        }
    
    def get_rtb_events(self, start_ts: int, end_ts: int, max_duration_min: int = 30) -> List[Dict[str, Any]]:
        """
        Get Return-To-Base events - flights that took off and landed at same airport within time limit.
        
        Returns:
            [{flight_id, callsign, departure_time, landing_time, duration_min, airport}]
        """
        rtb_events = []
        
        # Query flight tracks looking for same departure/arrival within time limit
        for db_name in ['live', 'research']:
            query = """
                SELECT DISTINCT flight_id, callsign, MIN(timestamp) as start_time, 
                       MAX(timestamp) as end_time, origin, destination
                FROM flight_tracks
                WHERE timestamp BETWEEN ? AND ?
                  AND origin IS NOT NULL 
                  AND destination IS NOT NULL
                GROUP BY flight_id
                HAVING origin = destination 
                   AND (MAX(timestamp) - MIN(timestamp)) <= ?
            """
            max_duration_s = max_duration_min * 60
            results = self._execute_query(db_name, query, (start_ts, end_ts, max_duration_s))
            
            for row in results:
                flight_id, callsign, start_time, end_time, airport, _ = row
                duration_min = (end_time - start_time) / 60
                rtb_events.append({
                    'flight_id': flight_id,
                    'callsign': callsign or 'UNKNOWN',
                    'departure_time': start_time,
                    'landing_time': end_time,
                    'duration_min': round(duration_min, 1),
                    'airport': airport or 'UNKNOWN'
                })
        
        return sorted(rtb_events, key=lambda x: x['departure_time'], reverse=True)

