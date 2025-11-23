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

export interface AnomalyReport {
    flight_id: string;
    callsign?: string;
    timestamp: number;
    is_anomaly: boolean;
    severity_cnn: number;
    severity_dense: number;
    full_report: any;
}

export interface AnalysisResult {
    is_anomaly: boolean;
    severity_cnn?: number;
    severity_dense?: number;
    full_report?: any;
}
