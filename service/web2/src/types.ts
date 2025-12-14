export interface TrackPoint {
    lat: number;
    lon: number;
    alt: number;
    timestamp: number;
    gspeed?: number;
    track?: number;
    flight_id?: string;
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
