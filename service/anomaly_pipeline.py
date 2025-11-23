from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

# Import all our detectors
# 1. Rules
from core.db import FlightRepository, DbConfig
from rules.rule_engine import AnomalyRuleEngine

# 2. XGBoost
from mlboost.detector import XGBoostDetector
# 3. Deep Dense
from ml_deep.detector import DeepAnomalyDetector
# 4. Deep CNN
from ml_deep_cnn.detector import DeepCNNDetector
# 5. Transformer
from ml_transformer.detector import TransformerAnomalyDetector

class AnomalyPipeline:
    def __init__(self):
        print("Initializing Anomaly Pipeline...")
        
        # --- 1. Rule Engine ---
        self.rules_path = Path("anomaly_rule.json")
        
        # Try to find the database in likely locations
        potential_dbs = [
            Path("last.db")
        ]
        self.db_path = next((p for p in potential_dbs if p.exists()), Path("rules/flight_tracks.db"))
        
        try:
            # We might not have a DB handy in this context, so pass None for repo if strictly analyzing passed tracks
            # However, some rules (Proximity) NEED a repo to find other flights.
            # For this demo, we'll try to use the flight_tracks.db if it exists, or None.
            repo = None
            if self.db_path.exists():
                    repo = FlightRepository(DbConfig(path=self.db_path))
            
            if self.rules_path.exists():
                self.rule_engine = AnomalyRuleEngine(repository=repo, rules_path=self.rules_path)
                print("  [+] Rule Engine Loaded")
            else:
                print("  [-] Rules Config NOT Found")
                self.rule_engine = None
        except Exception as e:
            print(f"  [-] Rule Engine Error: {e}")
            self.rule_engine = None
            
        # --- 2. XGBoost ---
        self.xgb_model_path = Path("mlboost/output/xgb_model.joblib")
        try:
            self.xgb_detector = XGBoostDetector(self.xgb_model_path)
            print("  [+] XGBoost Detector Loaded")
        except Exception as e:
            print(f"  [-] XGBoost Detector Error: {e}")
            self.xgb_detector = None

        # --- 3. Deep Dense ---
        self.deep_dir = Path("ml_deep/output")
        if (self.deep_dir / "clusters.joblib").exists():
            self.deep_detector = DeepAnomalyDetector(self.deep_dir)
            print("  [+] Deep Dense Detector Loaded")
        else:
            print("  [-] Deep Dense Detector NOT Found")
            self.deep_detector = None

        # --- 4. Deep CNN ---
        self.cnn_dir = Path("ml_deep_cnn/output")
        if (self.cnn_dir / "clusters.joblib").exists():
            self.cnn_detector = DeepCNNDetector(self.cnn_dir)
            print("  [+] Deep CNN Detector Loaded")
        else:
            print("  [-] Deep CNN Detector NOT Found")
            self.cnn_detector = None
            
        # --- 5. Transformer ---
        self.trans_dir = Path("ml_transformer/output")
        if (self.trans_dir / "clusters.joblib").exists():
            try:
                self.trans_detector = TransformerAnomalyDetector(self.trans_dir)
                print("  [+] Transformer Detector Loaded")
            except Exception as e:
                    print(f"  [-] Transformer Detector Error: {e}")
                    self.trans_detector = None
        else:
            print("  [-] Transformer Detector NOT Found")
            self.trans_detector = None

    def _calculate_confidence(self, results):
        weights = {
            "rules": 5.0,    # Rule hit = 100%
            "xgboost": 2.5,  # XGB hit = 50%
            "trans": 2.5,    # Trans hit = 50%
            "cnn": 1.0,      # CNN hit = 20%
            "dense": 1.0     # Dense hit = 20%
        }
        normalization_factor = 5.0 
        
        score = 0.0
        
        # Rules
        if results.get("layer_1_rules", {}).get("status") == "ANOMALY":
            score += weights["rules"]
            
        # XGBoost
        if results.get("layer_2_xgboost", {}).get("is_anomaly"):
            score += weights["xgboost"]
            
        # CNN
        if results.get("layer_4_deep_cnn", {}).get("is_anomaly"):
            score += weights["cnn"]
            
        # Dense
        if results.get("layer_3_deep_dense", {}).get("is_anomaly"):
            score += weights["dense"]
            
        # Transformer
        if results.get("layer_5_transformer", {}).get("is_anomaly"):
            score += weights["trans"]
            
        # Calculate percentage, capped at 1.0
        probability = min(1.0, score / normalization_factor)
        return round(probability * 100, 2) # Return percent

    def analyze(self, flight, active_flights_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run all layers on the flight and return a unified report.
        Filters points to strictly match the training bounding box.
        
        Args:
            flight: The FlightTrack object to analyze.
            active_flights_context: Optional dictionary of other active FlightTrack objects for proximity checks.
        """
        # --- 0. Filter Data to Training Region ---
        # Train Box: 28.53 - 34.59 N, 32.29 - 37.39 E
        TRAIN_NORTH = 34.597042
        TRAIN_SOUTH = 28.536275
        TRAIN_WEST  = 32.299805
        TRAIN_EAST  = 37.397461
        
        filtered_points = []
        for p in flight.sorted_points():
            if (TRAIN_SOUTH <= p.lat <= TRAIN_NORTH and 
                TRAIN_WEST <= p.lon <= TRAIN_EAST):
                filtered_points.append(p)
                
        # Create a virtual flight track with only relevant points
        # We clone the flight object structure but swap points
        from core.models import FlightTrack
        flight_active = FlightTrack(flight_id=flight.flight_id, points=filtered_points)
        
        results = {}
        
        # Check minimum points after filtering
        if len(filtered_points) < 50:
            return {
                "summary": {
                    "is_anomaly": False,
                    "triggers": [],
                    "flight_id": flight.flight_id,
                    "num_points": len(filtered_points),
                    "status": "SKIPPED_TOO_SHORT",
                    "info": f"Only {len(filtered_points)} points in monitored region (Need 50+)"
                }
            }

        is_anomaly_any = False
        summary_triggers = []

        # --- Layer 1: Rule Engine ---
        if self.rule_engine:
            try:
                # If we have context, we should use it to mock a repository for proximity checks
                # This is a bit of a hack: we swap the repository in the engine temporarily
                # or we pass the context to a specialized method.
                # The current RuleLogic uses ctx.repository.fetch_points_between.
                # We need to implement an "InMemoryRepository" adapter if we want real proximity checks without DB.
                
                # For now, we just run the track against the rules. Proximity will only work if
                # self.rule_engine was initialized with a real DB that has the other flights.
                # Since RealtimeMonitor does NOT write to the DB until AFTER analysis, 
                # pure DB-based proximity check will fail to see "current" neighbors.
                
                # FUTURE TODO: Implement InMemoryRepository(active_flights_context)
                
                rule_report = self.rule_engine.evaluate_track(flight_active)
                
                # Parse results for summary
                matched_rules = rule_report.get("matched_rules", [])

                if matched_rules:
                    status = "ANOMALY"
                    is_anomaly_any = True
                    summary_triggers.append("Rules")
                    triggers_text = [r["name"] for r in matched_rules]
                else:
                    status = "NORMAL"
                    triggers_text = []

                results["layer_1_rules"] = {
                    "status": status,
                    "triggers": triggers_text,
                    "report": rule_report # Full detailed report
                }
                
            except Exception as e:
                results["layer_1_rules"] = {"error": str(e), "status": "ERROR"}
        else:
                results["layer_1_rules"] = {"status": "SKIPPED", "info": "Engine not loaded"}

        # --- Layer 2: XGBoost ---
        if self.xgb_detector:
            try:
                res = self.xgb_detector.predict(flight_active)
                if "error" not in res:
                    results["layer_2_xgboost"] = res
                    if res["is_anomaly"]:
                        is_anomaly_any = True
                        summary_triggers.append("XGBoost")
                else:
                    results["layer_2_xgboost"] = {"error": res["error"]}
            except Exception as e:
                    results["layer_2_xgboost"] = {"error": str(e)}

        # --- Layer 3: Deep Dense ---
        if self.deep_detector:
            try:
                res = self.deep_detector.predict(flight_active)
                if "error" not in res:
                    results["layer_3_deep_dense"] = res
                    if res["is_anomaly"]:
                        is_anomaly_any = True
                        summary_triggers.append("DeepDense")
                else:
                    results["layer_3_deep_dense"] = {"error": res["error"]}
            except Exception as e:
                    results["layer_3_deep_dense"] = {"error": str(e)}

        # --- Layer 4: Deep CNN ---
        if self.cnn_detector:
            try:
                res = self.cnn_detector.predict(flight_active)
                if "error" not in res:
                    results["layer_4_deep_cnn"] = res
                    if res["is_anomaly"]:
                        is_anomaly_any = True
                        summary_triggers.append("DeepCNN")
                else:
                    results["layer_4_deep_cnn"] = {"error": res["error"]}
            except Exception as e:
                results["layer_4_deep_cnn"] = {"error": str(e)}

        # --- Layer 5: Transformer ---
        if self.trans_detector:
            try:
                res = self.trans_detector.predict(flight_active)
                # Transformer detector returns (score, is_anom) tuple, need to normalize to dict or update detector
                # The current detector.py for transformer returns (score, is_anom)
                # Let's wrap it
                if isinstance(res, tuple):
                        score, is_anom = res
                        results["layer_5_transformer"] = {
                            "score": float(score),
                            "is_anomaly": bool(is_anom),
                            "status": "ANOMALY" if is_anom else "NORMAL"
                        }
                        if is_anom:
                            is_anomaly_any = True
                            summary_triggers.append("Transformer")
                else:
                        results["layer_5_transformer"] = {"error": "Invalid return type"}
            except Exception as e:
                    results["layer_5_transformer"] = {"error": str(e)}

        # --- Summary ---
        # Extract simplified path for UI (lon, lat)
        # We return the FULL path so the UI shows the complete flight
        flight_path = [[p.lon, p.lat] for p in flight.sorted_points()]
        
        # Attempt to find a callsign from the points
        callsign = None
        for p in flight.points:
            if p.callsign and p.callsign.strip():
                callsign = p.callsign
                break

        # Calculate Confidence Score
        confidence_score = self._calculate_confidence(results)
        
        results["summary"] = {
            "is_anomaly": is_anomaly_any,
            "confidence_score": confidence_score,
            "triggers": summary_triggers,
            "flight_id": flight.flight_id,
            "callsign": callsign,
            "num_points": len(flight_active.points), # Analyzed points
            "flight_path": flight_path
        }
        
        return results

if __name__ == "__main__":
    # Self-test
    from flight_fetcher import get
    flight = get()
    
    pipeline = AnomalyPipeline()
    report = pipeline.analyze(flight)
    
    print("\n" + "="*30)
    print("FINAL REPORT")
    print("="*30)
    print(json.dumps(report, indent=2))
