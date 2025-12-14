import type { TrackPoint } from '../types';

/**
 * Types for AI-generated map actions
 */
export type AIActionType = 
    | 'highlight_point' 
    | 'highlight_segment' 
    | 'focus_time' 
    | 'zoom_to_bounds'
    | 'clear_highlights';

export interface HighlightPointAction {
    action: 'highlight_point';
    lat: number;
    lon: number;
    label?: string;
}

export interface HighlightSegmentAction {
    action: 'highlight_segment';
    startIndex: number;
    endIndex: number;
}

export interface FocusTimeAction {
    action: 'focus_time';
    timestamp: string | number; // ISO string or Unix timestamp
}

export interface ZoomToBoundsAction {
    action: 'zoom_to_bounds';
    north: number;
    south: number;
    east: number;
    west: number;
}

export interface ClearHighlightsAction {
    action: 'clear_highlights';
}

export type AIAction = 
    | HighlightPointAction 
    | HighlightSegmentAction 
    | FocusTimeAction 
    | ZoomToBoundsAction
    | ClearHighlightsAction;

/**
 * Parse AI response text to extract JSON action blocks.
 * Actions are expected to be wrapped in ```json ... ``` blocks.
 */
export function parseActionsFromResponse(responseText: string): AIAction[] {
    const actions: AIAction[] = [];
    
    // Match JSON blocks: ```json { ... } ``` or just { "action": ... }
    const jsonBlockRegex = /```json\s*([\s\S]*?)```/gi;
    const inlineJsonRegex = /\{[^{}]*"action"\s*:\s*"[^"]+[^{}]*\}/gi;
    
    // First try to find fenced code blocks
    let match;
    while ((match = jsonBlockRegex.exec(responseText)) !== null) {
        try {
            const jsonStr = match[1].trim();
            const parsed = JSON.parse(jsonStr);
            
            if (isValidAction(parsed)) {
                actions.push(parsed as AIAction);
            }
        } catch (e) {
            console.warn('Failed to parse JSON action block:', e);
        }
    }
    
    // If no fenced blocks found, try inline JSON
    if (actions.length === 0) {
        while ((match = inlineJsonRegex.exec(responseText)) !== null) {
            try {
                const parsed = JSON.parse(match[0]);
                
                if (isValidAction(parsed)) {
                    actions.push(parsed as AIAction);
                }
            } catch (e) {
                console.warn('Failed to parse inline JSON action:', e);
            }
        }
    }
    
    return actions;
}

/**
 * Validate that an object is a valid AI action
 */
function isValidAction(obj: any): boolean {
    if (!obj || typeof obj !== 'object' || !obj.action) {
        return false;
    }
    
    switch (obj.action) {
        case 'highlight_point':
            return typeof obj.lat === 'number' && typeof obj.lon === 'number';
        case 'highlight_segment':
            return typeof obj.startIndex === 'number' && typeof obj.endIndex === 'number';
        case 'focus_time':
            return obj.timestamp !== undefined;
        case 'zoom_to_bounds':
            return (
                typeof obj.north === 'number' &&
                typeof obj.south === 'number' &&
                typeof obj.east === 'number' &&
                typeof obj.west === 'number'
            );
        case 'clear_highlights':
            return true;
        default:
            return false;
    }
}

/**
 * Find the closest point in a track to a given timestamp
 */
export function findPointByTimestamp(
    points: TrackPoint[], 
    timestamp: string | number
): { point: TrackPoint; index: number } | null {
    if (!points || points.length === 0) return null;
    
    // Convert ISO string to Unix timestamp if needed
    let targetTs: number;
    if (typeof timestamp === 'string') {
        targetTs = new Date(timestamp).getTime() / 1000;
    } else {
        targetTs = timestamp;
    }
    
    // Find closest point
    let closestIndex = 0;
    let closestDiff = Math.abs(points[0].timestamp - targetTs);
    
    for (let i = 1; i < points.length; i++) {
        const diff = Math.abs(points[i].timestamp - targetTs);
        if (diff < closestDiff) {
            closestDiff = diff;
            closestIndex = i;
        }
    }
    
    return {
        point: points[closestIndex],
        index: closestIndex
    };
}

/**
 * Result of processing AI actions
 */
export interface ProcessedActions {
    highlightedPoint: { lat: number; lon: number; label?: string } | null;
    highlightedSegment: { startIndex: number; endIndex: number } | null;
    zoomBounds: { north: number; south: number; east: number; west: number } | null;
}

/**
 * Process a list of AI actions and convert them to map state.
 * Later actions override earlier ones of the same type.
 */
export function processActions(
    actions: AIAction[],
    flightPoints: TrackPoint[]
): ProcessedActions {
    const result: ProcessedActions = {
        highlightedPoint: null,
        highlightedSegment: null,
        zoomBounds: null
    };
    
    for (const action of actions) {
        switch (action.action) {
            case 'highlight_point':
                result.highlightedPoint = {
                    lat: action.lat,
                    lon: action.lon,
                    label: action.label
                };
                break;
                
            case 'highlight_segment':
                result.highlightedSegment = {
                    startIndex: action.startIndex,
                    endIndex: action.endIndex
                };
                break;
                
            case 'focus_time': {
                const found = findPointByTimestamp(flightPoints, action.timestamp);
                if (found) {
                    result.highlightedPoint = {
                        lat: found.point.lat,
                        lon: found.point.lon
                    };
                }
                break;
            }
                
            case 'zoom_to_bounds':
                result.zoomBounds = {
                    north: action.north,
                    south: action.south,
                    east: action.east,
                    west: action.west
                };
                break;
                
            case 'clear_highlights':
                result.highlightedPoint = null;
                result.highlightedSegment = null;
                break;
        }
    }
    
    return result;
}

/**
 * Strip action JSON blocks from the response text for clean display.
 */
export function stripActionsFromText(responseText: string): string {
    // Remove ```json ... ``` blocks
    let cleaned = responseText.replace(/```json\s*[\s\S]*?```/gi, '');
    
    // Clean up extra whitespace
    cleaned = cleaned.replace(/\n{3,}/g, '\n\n').trim();
    
    return cleaned;
}

