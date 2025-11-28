import { AnomalyReport, FlightTrack } from './types';

const API_BASE = 'http://localhost:8000/api'; // Assuming dev setup

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

export const fetchLearnedPaths = async (): Promise<any> => {
    const response = await fetch(`${API_BASE}/paths`);
    if (!response.ok) {
        throw new Error('Failed to fetch learned paths');
    }
    return response.json();
};

export const submitFeedback = async (flightId: string, isAnomaly: boolean, comments: string = ""): Promise<void> => {
    const response = await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            flight_id: flightId,
            is_anomaly: isAnomaly,
            comments: comments
        }),
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to submit feedback');
    }
};

