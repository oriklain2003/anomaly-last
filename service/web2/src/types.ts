export interface TrackPoint {
    lat: number;
    lon: number;
    alt: number;
    timestamp: number;
    gspeed?: number;
    track?: number;
    flight_id?: string;
    callsign?: string;
}

export interface FlightTrack {
    flight_id: string;
    points: TrackPoint[];
}

// ML Model anomaly point location
export interface AnomalyPoint {
    lat: number;
    lon: number;
    timestamp: number;
    point_score: number;
}

// Layer result with optional anomaly points
export interface LayerResult {
    is_anomaly?: boolean;
    status?: string;
    score?: number;
    threshold?: number;
    severity?: number;
    error?: string;
    anomaly_points?: AnomalyPoint[];
    triggers?: string[];
    report?: any;
}

export interface AnomalyReport {
    flight_id: string;
    callsign?: string;
    timestamp: number;
    is_anomaly: boolean;
    severity_cnn: number;
    severity_dense: number;
    full_report: any;
    feedback_id?: number;  // For history mode
    feedback_comments?: string;  // For history mode
    feedback_rule_id?: number | null;  // For history mode
    feedback_other_details?: string;  // For history mode
    user_label?: number; // 0: Normal, 1: Anomaly
}

export interface AnalysisResult {
    is_anomaly: boolean;
    severity_cnn?: number;
    severity_dense?: number;
    full_report?: any;
}

export interface DataFlight {
    flight_id: string;
    callsign?: string;
    start_time: number;
    end_time: number;
    point_count: number;
    source: string;
}

export interface AIReasoningResponse {
    type: 'message' | 'flights';
    response: string;
    flights?: AnomalyReport[];
}

// Intelligence Dashboard Types

export interface OverviewStats {
  total_flights: number;
  total_anomalies: number;
  safety_events: number;
  go_arounds: number;
  emergency_codes: number;
  near_miss: number;
}

export interface EmergencyCodeStat {
  code: string;
  count: number;
  airlines: Record<string, number>;
  flights: string[];
}

export interface NearMissEvent {
  timestamp: number;
  flight_id: string;
  other_flight_id: string;
  distance_nm: number;
  altitude_diff_ft: number;
  severity: 'high' | 'medium';
}

export interface GoAroundStat {
  airport: string;
  count: number;
  avg_per_day: number;
  by_hour: Record<number, number>;
}

export interface FlightPerDay {
  date: string;
  count: number;
  military_count: number;
  civilian_count: number;
}

export interface SignalLossLocation {
  lat: number;
  lon: number;
  count: number;
  avgDuration: number;  // Average gap duration in seconds
  intensity?: number;
  affected_flights?: number;
}

export interface AirlineEfficiency {
  airline: string;
  avg_flight_time_min: number;
  avg_holding_time_min: number;
  sample_count: number;
}

export interface HoldingPatternAnalysis {
  total_time_hours: number;
  estimated_fuel_cost_usd: number;
  peak_hours: number[];
  events_by_airport: Record<string, number>;
}

export interface GPSJammingPoint {
  lat: number;
  lon: number;
  intensity: number;
  first_seen: number;
  last_seen: number;
  event_count: number;
  affected_flights: number;
}

export interface MilitaryPattern {
  flight_id: string;
  callsign: string;
  country: string;
  type: string;
  pattern_type: string;
  locations: any[];
  frequency: number;
}

export interface AirspaceRisk {
  risk_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  factors: RiskFactor[];
  recommendation: string;
  timestamp: number;
}

export interface RiskFactor {
  name: string;
  weight: number;
  value: number | string;
  impact: number;
  description: string;
}

// Additional analytics types for missing features
export interface DiversionStats {
  total_diversions: number;
  total_large_deviations: number;  // >20nm from route
  total_holding_360s: number;  // 360° holds before landing
  by_airport: Record<string, number>;
  by_airline: Record<string, number>;
}

export interface RTBEvent {
  flight_id: string;
  callsign: string;
  departure_time: number;
  landing_time: number;
  duration_min: number;
  airport: string;
}

export interface RunwayStats {
  runway: string;
  airport: string;
  landings: number;
  takeoffs: number;
  total: number;
}

export interface MonthlyTrend {
  month: string;  // YYYY-MM
  total_flights: number;
  anomalies: number;
  safety_events: number;
  busiest_hour: number;
}

export interface SafetyForecast {
  forecast_period_hours: number;
  expected_events: number;
  confidence_interval: [number, number];
  peak_risk_hours: number[];
}

export interface SimilarFlight {
  flight_id: string;
  callsign: string;
  similarity_score: number;
  date: string | null;
  pattern: string;
}

export interface AnomalyDNA {
  flight_info: {
    flight_id: string;
    callsign?: string;
  };
  similar_flights: SimilarFlight[];
  recurring_pattern: string;
  risk_assessment: string;
  insights: string[];
  anomalies_detected?: Array<{
    rule_id: number;
    rule_name: string;
    timestamp: number;
  }>;
}

export interface PatternCluster {
  pattern_id: string;
  description: string;
  location: { lat: number; lon: number };
  flights: string[];
  first_seen: number;
  last_seen: number;
  occurrence_count: number;
  risk_level: string;
}
