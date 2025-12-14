from __future__ import annotations

import json
import time
import dataclasses
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence

from anomaly_pipeline import AnomalyPipeline
from core.db import DbConfig, FlightRepository
from core.models import FlightTrack, TrackPoint
from ..detection import DetectionWrapper
from ..filtering import FilteringLayer
from ..models import FilteredTrack, SummaryPayload, DetectionOutput
from ..reasoning import LLMReasoner
from ..summarizer import SummaryBuilder
from ..flight_fetcher import ensure_local_db, TABLE_NAME

REPORT_DIR = Path(__file__).resolve().parent / "reports"

@dataclass
class FlightEvaluationResult:
    flight_id: str
    category: str
    
    # Layer outputs (from Full Pipeline)
    filter_actions: Dict[str, Any]
    rule_findings: Dict[str, Any]
    llm_reasoning: Dict[str, Any]
    
    # Final decision (Full Pipeline)
    final_decision: bool # Is anomaly?
    
    # Metrics for combinations
    baseline_is_anomaly: bool
    filtering_only_is_anomaly: bool
    llm_only_is_anomaly: bool
    full_pipeline_is_anomaly: bool
    
    # Performance
    time_taken: float
    
    # Data for UI
    path: List[List[float]] = field(default_factory=list)

@dataclass
class AggregateMetrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    
    def calculate(self):
        total = self.tp + self.fp + self.tn + self.fn
        if total == 0: return
        self.accuracy = (self.tp + self.tn) / total
        self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

@dataclass
class EvaluationSummary:
    generated_at: str
    total_flights: int
    metrics_by_layer: Dict[str, AggregateMetrics]
    records: List[FlightEvaluationResult]

class ComprehensiveEvaluator:
    def __init__(
        self,
        repository: Optional[FlightRepository] = None,
        db_path: Optional[Path] = None,
        table: str = TABLE_NAME,
    ) -> None:
        self.db_path = ensure_local_db(db_path)
        self.table = table
        self.repository = repository or FlightRepository(
            DbConfig(path=self.db_path, table=self.table)
        )
        
        # Initialize layers
        self.old_pipeline = AnomalyPipeline()
        self.filtering = FilteringLayer()
        self.detection = DetectionWrapper()
        self.summarizer = SummaryBuilder()
        self.reasoner = LLMReasoner()

    def evaluate_dataset(self, dataset: Dict[str, List[str]]) -> EvaluationSummary:
        records = []
        
        all_ids = []
        category_map = {}
        
        for category, ids in dataset.items():
            for fid in ids:
                all_ids.append(fid)
                category_map[fid] = category
                
        # Pre-fetch all if possible, or fetch one by one
        # fetch_flight handles caching if properly used
        
        for flight_id in all_ids:
            try:
                flight = self.repository.fetch_flight(flight_id)
            except Exception as e:
                print(f"Error fetching flight {flight_id}: {e}")
                continue
                
            if not flight or not flight.points:
                # Try fetching from source if repository fails or returns empty
                # Assuming repository is connected to cached DB, manual fetch might be needed
                # But repository.fetch_flight should handle it if it's the main repo
                # Here we assume the repo is set up correctly.
                print(f"Skipping {flight_id} - no data found")
                continue
            
            category = category_map[flight_id]
            result = self._evaluate_flight(flight, category)
            records.append(result)
            
        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(records)
        
        summary = EvaluationSummary(
            generated_at=datetime.utcnow().isoformat(),
            total_flights=len(records),
            metrics_by_layer=metrics,
            records=records
        )
        
        self._write_report(summary)
        return summary

    def _evaluate_flight(self, flight: FlightTrack, category: str) -> FlightEvaluationResult:
        start_time = time.time()
        
        # 1. Baseline
        try:
            baseline_report = self.old_pipeline.analyze(flight)
            baseline_is_anomaly = bool(baseline_report.get("summary", {}).get("is_anomaly", False))
        except Exception as e:
            print(f"Baseline error for {flight.flight_id}: {e}")
            baseline_is_anomaly = False # Fail safe

        # 2. Filtering Only (Filter -> Detection -> Heuristic Score)
        try:
            filtered_track = self.filtering.process(flight)
            flight_filtered = FlightTrack(flight_id=flight.flight_id, points=filtered_track.clean_points)
            detection_filtered = self.detection.run(flight_filtered)
            filtering_only_is_anomaly = self._heuristic_decision(detection_filtered)
        except Exception as e:
            print(f"FilteringOnly error for {flight.flight_id}: {e}")
            # If filtering fails, maybe fallback to raw?
            filtering_only_is_anomaly = False
            filtered_track = self.filtering._minimal_track(flight) # Fallback for next steps
            detection_filtered = DetectionOutput(flight.flight_id, [], None, None, {}) # Dummy

        # 3. LLM Only (Raw -> Detection -> LLM)
        try:
            raw_filtered_track = self.filtering._minimal_track(flight) # Just wrapper, no filter
            detection_raw = self.detection.run(flight)
            summary_llm_only = self.summarizer.build(raw_filtered_track, detection_raw)
            decision_llm_only = self.reasoner.evaluate(summary_llm_only)
            llm_only_is_anomaly = decision_llm_only.is_anomaly
        except Exception as e:
            print(f"LLMOnly error for {flight.flight_id}: {e}")
            llm_only_is_anomaly = False

        # 4. Full Pipeline (Filter -> Detection -> LLM)
        try:
            # We already have filtered_track and detection_filtered from step 2
            summary_full = self.summarizer.build(filtered_track, detection_filtered)
            decision_full = self.reasoner.evaluate(summary_full)
            full_pipeline_is_anomaly = decision_full.is_anomaly
            
            # Capture details
            filter_actions = {
                "original_points": len(flight.points),
                "filtered_points": len(filtered_track.clean_points),
                "flags": asdict(filtered_track.signal_quality_flags)
            }
            rule_findings = {
                "triggers": [r.name for r in detection_filtered.rules_triggered],
                "scores": asdict(detection_filtered.model_scores)
            }
            llm_reasoning = {
                "explanation": decision_full.reasoning,
                "corrections": decision_full.rule_corrections,
                "score": decision_full.logical_anomaly_score
            }
        except Exception as e:
            print(f"FullPipeline error for {flight.flight_id}: {e}")
            full_pipeline_is_anomaly = False
            filter_actions = {"error": str(e)}
            rule_findings = {}
            llm_reasoning = {}

        elapsed = time.time() - start_time
        
        # Prepare path for UI (lat, lon, alt)
        ui_path = [[p.lat, p.lon, p.alt] for p in flight.points] if flight.points else []

        return FlightEvaluationResult(
            flight_id=flight.flight_id,
            category=category,
            filter_actions=filter_actions,
            rule_findings=rule_findings,
            llm_reasoning=llm_reasoning,
            final_decision=full_pipeline_is_anomaly,
            baseline_is_anomaly=baseline_is_anomaly,
            filtering_only_is_anomaly=filtering_only_is_anomaly,
            llm_only_is_anomaly=llm_only_is_anomaly,
            full_pipeline_is_anomaly=full_pipeline_is_anomaly,
            path=ui_path,
            time_taken=elapsed
        )

    def _heuristic_decision(self, detection: DetectionOutput) -> bool:
        # Mimic AnomalyPipeline._calculate_confidence logic
        score = 0.0
        
        # Rules
        if detection.rules_triggered:
            score += 6.0
            
        # Models
        scores = detection.model_scores
        # Thresholds are implicitly handled in pipeline, but here we just have scores.
        # DetectionWrapper returns scores, not "is_anomaly" boolean for models.
        # We need to know if models triggered.
        # The DetectionWrapper wraps output in 'raw_layers'.
        
        raw = detection.raw_layers
        
        if raw.get("layer_2_xgboost", {}).get("is_anomaly"): score += 2.5
        if raw.get("layer_4_deep_cnn", {}).get("is_anomaly"): score += 1.0
        if raw.get("layer_3_deep_dense", {}).get("is_anomaly"): score += 1.0
        # Transformer/Hybrid not in DetectionWrapper explicitly yet? 
        # DetectionWrapper (current version) only checks Rules, XGB, Dense, CNN.
        
        confidence = min(1.0, score / 6.0) * 100
        return confidence >= 40.0

    def _compute_aggregate_metrics(self, records: List[FlightEvaluationResult]) -> Dict[str, AggregateMetrics]:
        layers = ["baseline", "filtering_only", "llm_only", "full_pipeline"]
        metrics = {layer: AggregateMetrics() for layer in layers}
        
        for record in records:
            is_anomaly_truth = (record.category == "anomaly")
            # Glitch/Borderline: ambiguous. 
            # "signal_glitch_flights" -> Should be NORMAL (not anomaly).
            # "borderline" -> Depends? Let's assume NORMAL for strict anomaly detection, or ignore?
            # User says: "signal_glitch_flights – flights with bad FR24 data...". These usually cause False Positives. 
            # So Truth should be False (Normal).
            
            if record.category == "borderline":
                # Skip borderline for strict metrics? Or assume Normal?
                # Let's skip for now to avoid noise, or treat as Normal.
                # Usually borderline implies "not clear anomaly", so Normal.
                is_anomaly_truth = False
            elif record.category == "glitch":
                is_anomaly_truth = False
            elif record.category == "normal":
                is_anomaly_truth = False
            elif record.category == "anomaly":
                is_anomaly_truth = True
            
            for layer in layers:
                pred = getattr(record, f"{layer}_is_anomaly")
                
                if is_anomaly_truth and pred:
                    metrics[layer].tp += 1
                elif not is_anomaly_truth and not pred:
                    metrics[layer].tn += 1
                elif not is_anomaly_truth and pred:
                    metrics[layer].fp += 1
                elif is_anomaly_truth and not pred:
                    metrics[layer].fn += 1
                    
        for m in metrics.values():
            m.calculate()
            
        return metrics

    def _write_report(self, summary: EvaluationSummary) -> Path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"comprehensive_eval_{timestamp}.json"
        
        # Serialize
        data = dataclasses.asdict(summary)
        path.write_text(json.dumps(data, indent=2))
        print(f"Report written to {path}")
        return path

