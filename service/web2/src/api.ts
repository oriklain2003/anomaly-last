import { 
    AnomalyReport, FlightTrack, TrackPoint, DataFlight, AIReasoningResponse,
    AnomalyDNA, PatternCluster, SafetyForecast,
    OverviewStats, EmergencyCodeStat, NearMissEvent, GoAroundStat,
    FlightPerDay, SignalLossLocation, AirlineEfficiency, HoldingPatternAnalysis,
    GPSJammingPoint, MilitaryPattern, AirspaceRisk
} from './types';
import type { AIAction } from './utils/aiActions';
import type { ChatMessage } from './chatTypes';

const API_BASE = (import.meta.env.VITE_API_URL || '') + '/api';

// ============================================================
// AI Analyze Types
// ============================================================

export interface AIAnalyzeRequest {
    screenshot: string;  // base64 PNG (with or without data URL prefix)
    question: string;
    flight_id: string;
    flight_data: TrackPoint[];
    anomaly_report: any;
    selected_point?: { lat: number; lon: number; timestamp?: number };
    history?: { role: string; content: string }[];  // Conversation history
}

export interface AIAnalyzeResponse {
    response: string;
    actions: AIAction[];
}

export const fetchLiveAnomalies = async (startTs: number, endTs: number): Promise<AnomalyReport[]> => {
    const response = await fetch(`${API_BASE}/live/anomalies?start_ts=${startTs}&end_ts=${endTs}`);
    if (!response.ok) {
        throw new Error('Failed to fetch anomalies');
    }
    return response.json();
};

export const fetchLiveTrack = async (flightId: string): Promise<FlightTrack> => {
    const response = await fetch(`${API_BASE}/live/track/${flightId}`);
    if (!response.ok) {
        throw new Error('Failed to fetch track');
    }
    return response.json();
};

export const fetchResearchAnomalies = async (startTs: number, endTs: number): Promise<AnomalyReport[]> => {
    const response = await fetch(`${API_BASE}/research/anomalies?start_ts=${startTs}&end_ts=${endTs}`);
    if (!response.ok) {
        throw new Error('Failed to fetch research anomalies');
    }
    return response.json();
};

export const fetchAnalyzeFlight = async (flightId: string): Promise<FlightTrack> => {
    const response = await fetch(`${API_BASE}/analyze/${flightId}`);
    if (!response.ok) {
        throw new Error('Failed to analyze flight');
    }
    const result = await response.json();
    if (result.track) {
        return result.track;
    }
    throw new Error('Track data missing in analysis result');
};

export const fetchUnifiedTrack = async (flightId: string): Promise<FlightTrack> => {
    const response = await fetch(`${API_BASE}/track/unified/${flightId}`);
    if (!response.ok) {
        throw new Error('Failed to fetch flight track');
    }
    return response.json();
};

export const fetchResearchTrack = async (flightId: string): Promise<FlightTrack> => {
    const response = await fetch(`${API_BASE}/research/track/${flightId}`);
    if (!response.ok) {
        throw new Error('Failed to fetch research track');
    }
    return response.json();
};

export const fetchFeedbackTrack = async (flightId: string): Promise<FlightTrack> => {
    const response = await fetch(`${API_BASE}/feedback/track/${flightId}`);
    if (!response.ok) {
        throw new Error('Failed to fetch feedback track');
    }
    return response.json();
};

export const fetchLearnedPaths = async (): Promise<any> => {
    const response = await fetch(`${API_BASE}/paths`);
    if (!response.ok) {
        throw new Error('Failed to fetch learned paths');
    }
    return response.json();
};

// ============================================================
// Learned Layers Types (SID, STAR, Turns, Paths)
// ============================================================

export interface LearnedPathPoint {
    lat: number;
    lon: number;
    alt: number;
}

export interface LearnedPath {
    id: string;
    origin: string | null;
    destination: string | null;
    centerline: LearnedPathPoint[];
    width_nm?: number;
    member_count: number;
}

export interface LearnedTurnZone {
    id: number;
    lat: number;
    lon: number;
    radius_nm: number;
    avg_alt_ft: number;
    angle_range_deg: [number, number];
    avg_speed_kts: number;
    member_count: number;
    directions: { left: number; right: number };
}

export interface LearnedProcedure {
    id: string;
    airport: string;
    type: 'SID' | 'STAR';
    centerline: LearnedPathPoint[];
    width_nm?: number;
    member_count: number;
}

export interface LearnedLayers {
    paths: LearnedPath[];
    turns: LearnedTurnZone[];
    sids: LearnedProcedure[];
    stars: LearnedProcedure[];
}

export const fetchLearnedLayers = async (): Promise<LearnedLayers> => {
    const response = await fetch(`${API_BASE}/learned-layers`);
    if (!response.ok) {
        throw new Error('Failed to fetch learned layers');
    }
    return response.json();
};

export const fetchRules = async (): Promise<{ id: number; name: string; description: string }[]> => {
    const response = await fetch(`${API_BASE}/rules`);
    if (!response.ok) {
        throw new Error('Failed to fetch rules');
    }
    return response.json();
};

export const fetchFlightsByRule = async (ruleId: number, signal?: AbortSignal): Promise<AnomalyReport[]> => {
    const response = await fetch(`${API_BASE}/rules/${ruleId}/flights`, { signal });
    if (!response.ok) {
        throw new Error('Failed to fetch flights by rule');
    }
    return response.json();
};

export const fetchCallsignFromResearch = async (flightId: string): Promise<string | null> => {
    try {
        const response = await fetch(`${API_BASE}/research/callsign/${flightId}`);
        if (!response.ok) return null;
        const data = await response.json();
        return data?.callsign || null;
    } catch (error) {
        console.warn('Failed to fetch callsign', error);
        return null;
    }
};

export interface FeedbackParams {
    flightId: string;
    isAnomaly: boolean;
    comments?: string;
    ruleId?: number | null;  // null means "Other" option
    otherDetails?: string;   // Used when ruleId is null
}

export interface FeedbackHistoryItem {
    feedback_id: number;
    flight_id: string;
    timestamp: number;
    callsign?: string;
    comments: string;
    rule_id?: number | null;
    other_details?: string;
    model_version?: string;
    full_report?: any;
    is_anomaly: boolean;
}

export interface UpdateFeedbackParams {
    ruleId?: number | null;  // null means "Other" option
    comments?: string;
    otherDetails?: string;   // Used when ruleId is null
}

export const submitFeedback = async (params: FeedbackParams): Promise<void> => {
    const { flightId, isAnomaly, comments = "", ruleId, otherDetails = "" } = params;
    
    const response = await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            flight_id: flightId,
            is_anomaly: isAnomaly,
            comments: comments,
            rule_id: ruleId,
            other_details: otherDetails
        }),
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to submit feedback');
    }
};

export const fetchFeedbackHistory = async (startTs: number = 0, endTs?: number, limit: number = 100): Promise<AnomalyReport[]> => {
    const params = new URLSearchParams({
        start_ts: startTs.toString(),
        limit: limit.toString()
    });
    
    if (endTs !== undefined) {
        params.append('end_ts', endTs.toString());
    }
    
    const response = await fetch(`${API_BASE}/feedback/history?${params}`);
    if (!response.ok) {
        throw new Error('Failed to fetch feedback history');
    }
    return response.json();
};

export const updateFeedback = async (feedbackId: number, params: UpdateFeedbackParams): Promise<void> => {
    const { ruleId, comments = "", otherDetails = "" } = params;
    
    const response = await fetch(`${API_BASE}/feedback/${feedbackId}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            rule_id: ruleId,
            comments: comments,
            other_details: otherDetails
        }),
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to update feedback');
    }
};

export const reanalyzeFeedbackFlight = async (flightId: string): Promise<AnomalyReport> => {
    const response = await fetch(`${API_BASE}/feedback/reanalyze/${flightId}`, {
        method: 'POST',
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to re-analyze flight');
    }
    
    return response.json();
};

// ============================================================
// AI Analyze Endpoint
// ============================================================

/**
 * Send a screenshot and question to the AI co-pilot for analysis.
 * Returns the AI's response text and optional map actions.
 */
export const analyzeWithAI = async (request: AIAnalyzeRequest): Promise<AIAnalyzeResponse> => {
    const response = await fetch(`${API_BASE}/ai/analyze`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            screenshot: request.screenshot,
            question: request.question,
            flight_id: request.flight_id,
            flight_data: request.flight_data,
            anomaly_report: request.anomaly_report,
            selected_point: request.selected_point,
            history: request.history || []
        }),
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'AI analysis failed');
    }
    
    return response.json();
};

export const fetchDataFlights = async (startTs: number, endTs: number): Promise<DataFlight[]> => {
    const response = await fetch(`${API_BASE}/data/flights?start_ts=${startTs}&end_ts=${endTs}`);
    if (!response.ok) {
        throw new Error('Failed to fetch data flights');
    }
    return response.json();
};

// ============================================================
// AI Reasoning Endpoint
// ============================================================

export interface ReasoningFlightContext {
    flightId: string;
    points: TrackPoint[];
    anomalyReport?: any;
}

/**
 * Send a message to the AI reasoning agent.
 * The agent can query the flight database and return either:
 * - A text response (type: 'message')
 * - A list of flights to display (type: 'flights')
 * 
 * Optionally pass flight context for visual analysis with map image.
 */
export const sendReasoningQuery = async (
    message: string,
    history: ChatMessage[],
    flightContext?: ReasoningFlightContext
): Promise<AIReasoningResponse> => {
    const body: any = {
        message,
        history: history.map(m => ({ role: m.role, content: m.content }))
    };
    
    // Add flight context if provided
    if (flightContext) {
        body.flight_id = flightContext.flightId;
        body.points = flightContext.points;
        body.anomaly_report = flightContext.anomalyReport;
    }
    
    const response = await fetch(`${API_BASE}/ai/reasoning`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'AI reasoning failed');
    }
    
    return response.json();
};

// ============================================================
// Intelligence Dashboard API Functions
// ============================================================

// Cache Management
export const clearCache = async (): Promise<{ status: string; cleared_entries: number }> => {
    const response = await fetch(`${API_BASE}/cache/clear`, { method: 'POST' });
    if (!response.ok) throw new Error('Failed to clear cache');
    return response.json();
};

export const getCacheInfo = async (): Promise<{ total_entries: number; valid_entries: number; expiry_seconds: number }> => {
    const response = await fetch(`${API_BASE}/cache/info`);
    if (!response.ok) throw new Error('Failed to get cache info');
    return response.json();
};

// Level 1: Statistics
export const fetchStatsOverview = async (startTs: number, endTs: number, forceRefresh = false): Promise<OverviewStats> => {
    const url = `${API_BASE}/stats/overview?start_ts=${startTs}&end_ts=${endTs}${forceRefresh ? '&force_refresh=true' : ''}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch overview stats');
    return response.json();
};

export const fetchEmergencyCodes = async (startTs: number, endTs: number): Promise<EmergencyCodeStat[]> => {
    const response = await fetch(`${API_BASE}/stats/safety/emergency-codes?start_ts=${startTs}&end_ts=${endTs}`);
    if (!response.ok) throw new Error('Failed to fetch emergency codes');
    return response.json();
};

export const fetchNearMissEvents = async (startTs: number, endTs: number, severity?: string): Promise<NearMissEvent[]> => {
    const url = severity 
        ? `${API_BASE}/stats/safety/near-miss?start_ts=${startTs}&end_ts=${endTs}&severity=${severity}`
        : `${API_BASE}/stats/safety/near-miss?start_ts=${startTs}&end_ts=${endTs}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch near-miss events');
    return response.json();
};

export const fetchGoArounds = async (startTs: number, endTs: number, airport?: string): Promise<GoAroundStat[]> => {
    const url = airport 
        ? `${API_BASE}/stats/safety/go-arounds?start_ts=${startTs}&end_ts=${endTs}&airport=${airport}`
        : `${API_BASE}/stats/safety/go-arounds?start_ts=${startTs}&end_ts=${endTs}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch go-arounds');
    return response.json();
};

export const fetchFlightsPerDay = async (startTs: number, endTs: number, forceRefresh = false): Promise<FlightPerDay[]> => {
    const url = `${API_BASE}/stats/traffic/flights-per-day?start_ts=${startTs}&end_ts=${endTs}${forceRefresh ? '&force_refresh=true' : ''}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch flights per day');
    return response.json();
};

export const fetchBusiestAirports = async (startTs: number, endTs: number, limit = 10, forceRefresh = false): Promise<any[]> => {
    const url = `${API_BASE}/stats/traffic/busiest-airports?start_ts=${startTs}&end_ts=${endTs}&limit=${limit}${forceRefresh ? '&force_refresh=true' : ''}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch busiest airports');
    return response.json();
};

export const fetchSignalLoss = async (startTs: number, endTs: number): Promise<SignalLossLocation[]> => {
    const response = await fetch(`${API_BASE}/stats/traffic/signal-loss?start_ts=${startTs}&end_ts=${endTs}`);
    if (!response.ok) throw new Error('Failed to fetch signal loss');
    return response.json();
};

// Level 2: Insights
export const fetchAirlineEfficiency = async (startTs?: number, endTs?: number, route?: string): Promise<AirlineEfficiency[]> => {
    const params = new URLSearchParams();
    if (startTs) params.append('start_ts', startTs.toString());
    if (endTs) params.append('end_ts', endTs.toString());
    if (route) params.append('route', route);
    
    const url = params.toString() 
        ? `${API_BASE}/insights/airline-efficiency?${params.toString()}`
        : `${API_BASE}/insights/airline-efficiency`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch airline efficiency');
    return response.json();
};

export const fetchHoldingPatterns = async (startTs: number, endTs: number): Promise<HoldingPatternAnalysis> => {
    const response = await fetch(`${API_BASE}/insights/holding-patterns?start_ts=${startTs}&end_ts=${endTs}`);
    if (!response.ok) throw new Error('Failed to fetch holding patterns');
    return response.json();
};

export const fetchAlternateAirports = async (airport: string, eventDate?: number): Promise<any[]> => {
    const url = eventDate 
        ? `${API_BASE}/insights/alternate-airports?airport=${airport}&event_date=${eventDate}`
        : `${API_BASE}/insights/alternate-airports?airport=${airport}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch alternate airports');
    return response.json();
};

// Level 3: Intelligence
export const fetchGPSJamming = async (startTs: number, endTs: number): Promise<GPSJammingPoint[]> => {
    const response = await fetch(`${API_BASE}/intel/gps-jamming?start_ts=${startTs}&end_ts=${endTs}`);
    if (!response.ok) throw new Error('Failed to fetch GPS jamming');
    return response.json();
};

export const fetchMilitaryPatterns = async (startTs: number, endTs: number, country?: string, aircraftType?: string): Promise<MilitaryPattern[]> => {
    let url = `${API_BASE}/intel/military-patterns?start_ts=${startTs}&end_ts=${endTs}`;
    if (country) url += `&country=${country}`;
    if (aircraftType) url += `&aircraft_type=${aircraftType}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch military patterns');
    return response.json();
};

// Level 4: Predictive
export const fetchAirspaceRisk = async (): Promise<AirspaceRisk> => {
    const response = await fetch(`${API_BASE}/predict/airspace-risk`);
    if (!response.ok) throw new Error('Failed to fetch airspace risk');
    return response.json();
};

export const predictTrajectory = async (flightId: string, currentPosition: any): Promise<any> => {
    const response = await fetch(`${API_BASE}/predict/trajectory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flight_id: flightId, current_position: currentPosition })
    });
    if (!response.ok) throw new Error('Failed to predict trajectory');
    return response.json();
};

export const fetchSafetyForecast = async (hoursAhead = 24): Promise<SafetyForecast> => {
    const response = await fetch(`${API_BASE}/predict/safety-forecast?hours_ahead=${hoursAhead}`);
    if (!response.ok) throw new Error('Failed to fetch safety forecast');
    return response.json();
};

// Anomaly DNA endpoints
export const fetchAnomalyDNA = async (flightId: string, lookbackDays = 30): Promise<AnomalyDNA> => {
    const response = await fetch(`${API_BASE}/intelligence/anomaly-dna/${flightId}?lookback_days=${lookbackDays}`);
    if (!response.ok) throw new Error('Failed to fetch anomaly DNA');
    return response.json();
};

export const fetchPatternClusters = async (startTs: number, endTs: number, minOccurrences = 3): Promise<PatternCluster[]> => {
    const params = new URLSearchParams({
        start_ts: startTs.toString(),
        end_ts: endTs.toString(),
        min_occurrences: minOccurrences.toString()
    });
    const response = await fetch(`${API_BASE}/intelligence/pattern-clusters?${params}`);
    if (!response.ok) throw new Error('Failed to fetch pattern clusters');
    return response.json();
};

// ============================================================
// Additional Analytics Endpoints (New UI Features)
// ============================================================

// Peak Hours Analysis with correlation
export interface PeakHoursAnalysis {
    peak_traffic_hours: number[];
    peak_safety_hours: number[];
    correlation_score: number;
    hourly_data: { hour: number; traffic: number; safety_events: number }[];
}

export const fetchPeakHoursAnalysis = async (startTs: number, endTs: number): Promise<PeakHoursAnalysis> => {
    const response = await fetch(`${API_BASE}/trends/peak-hours?start_ts=${startTs}&end_ts=${endTs}`);
    if (!response.ok) throw new Error('Failed to fetch peak hours analysis');
    return response.json();
};

// Diversion Statistics
export interface DiversionStats {
    total_diversions: number;
    total_large_deviations: number;
    total_holding_360s: number;
    by_airport: Record<string, number>;
    by_airline: Record<string, number>;
}

export const fetchDiversionStats = async (startTs: number, endTs: number): Promise<DiversionStats> => {
    const response = await fetch(`${API_BASE}/stats/diversions?start_ts=${startTs}&end_ts=${endTs}`);
    if (!response.ok) throw new Error('Failed to fetch diversion stats');
    return response.json();
};

// Alternate Airports
export interface AlternateAirport {
    airport: string;
    count: number;
    aircraft_types: string[];
    last_used: number;
}

export const fetchAlternateAirportsData = async (startTs: number, endTs: number): Promise<AlternateAirport[]> => {
    const response = await fetch(`${API_BASE}/trends/alternate-airports?start_ts=${startTs}&end_ts=${endTs}`);
    if (!response.ok) throw new Error('Failed to fetch alternate airports');
    return response.json();
};

// Runway Usage
export interface RunwayUsage {
    runway: string;
    airport: string;
    landings: number;
    takeoffs: number;
    total: number;
}

export const fetchRunwayUsage = async (airport: string, startTs: number, endTs: number): Promise<RunwayUsage[]> => {
    const response = await fetch(`${API_BASE}/stats/runway-usage?airport=${airport}&start_ts=${startTs}&end_ts=${endTs}`);
    if (!response.ok) throw new Error('Failed to fetch runway usage');
    return response.json();
};

// Trajectory Prediction with Restricted Zones
export interface TrajectoryPrediction {
    flight_id: string;
    predicted_path: { lat: number; lon: number; time_offset_s: number }[];
    breach_warning: boolean;
    breach_zone?: string;
    breach_severity?: string;
    closest_zone?: { name: string; distance_nm: number };
    prediction_confidence: number;
}

export const fetchTrajectoryPrediction = async (flightId: string): Promise<TrajectoryPrediction> => {
    const response = await fetch(`${API_BASE}/predict/trajectory/${flightId}`, {
        method: 'POST'
    });
    if (!response.ok) throw new Error('Failed to fetch trajectory prediction');
    return response.json();
};

// Anomaly DNA (enhanced)
export const fetchAnomalyDNAEnhanced = async (flightId: string, lookbackDays = 30): Promise<AnomalyDNA> => {
    const response = await fetch(`${API_BASE}/intel/anomaly-dna/${flightId}?lookback_days=${lookbackDays}`);
    if (!response.ok) throw new Error('Failed to fetch anomaly DNA');
    return response.json();
};
