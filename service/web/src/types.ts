export interface FlightPath {
    [index: number]: [number, number]; // Array of [lon, lat]
}

export interface RuleDefinition {
    id: number;
    name: string;
    definition: string;
    operational_significance: string;
    matched: boolean;
    summary: string;
    details: any;
}

export interface RuleReport {
    flight_id: string;
    total_rules: number;
    matched_rules: RuleDefinition[];
    evaluations: RuleDefinition[];
}

export interface RuleResult {
    status: string;
    triggers: string[];
    report?: RuleReport; // Full detailed report
}

export interface ModelResult {
    status: string;
    score?: number;
    severity?: number;
    is_anomaly: boolean;
}

export interface AnalysisSummary {
    flight_id: string;
    is_anomaly: boolean;
    confidence_score: number;
    triggers: string[];
    num_points: number;
    flight_path: [number, number][];
}

export interface AnalysisResult {
    summary: AnalysisSummary;
    layer_1_rules: RuleResult;
    layer_2_xgboost: ModelResult;
    layer_3_deep_dense: ModelResult;
    layer_4_deep_cnn: ModelResult;
    layer_5_transformer: ModelResult;
    layer_6_hybrid: ModelResult;
}

export interface Airport {
    code: string;
    name: string;
    lat: number;
    lon: number;
}
