import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import sqlite3
from rich.console import Console
from rich.table import Table

# Fix for DLL load error on Windows by importing torch first
try:
    import torch
except ImportError:
    pass

# Set paths correctly
sys.path.append(str(Path(__file__).resolve().parent))

from core.db import DbConfig, FlightRepository
from core.models import FlightTrack, FlightMetadata
from mlboost.detector import XGBoostDetector
from ml_deep.detector import DeepAnomalyDetector
from ml_deep_cnn.detector import DeepCNNDetector
from ml_transformer.detector import TransformerAnomalyDetector
try:
    from rules.rule_engine import AnomalyRuleEngine
except ImportError:
    from rule_engine import AnomalyRuleEngine # If running from inside rules dir

# Configure Logging
logging.basicConfig(level=logging.ERROR) # Silence verbose logs

def main():
    console = Console()
    console.print("[bold blue]=== Comprehensive Anomaly Detection Test ===[/bold blue]")
    
    # Paths
    db_path = Path("last.db")
    xgb_path = Path("mlboost/output/xgb_model.joblib")
    deep_dir = Path("ml_deep/output")
    cnn_dir = Path("ml_deep_cnn/output")
    trans_dir = Path("ml_transformer/output")
    rules_path = Path("anomaly_rule.json")
    
    # 1. Initialize Detectors
    console.print("\n[bold]1. Initializing Models...[/bold]")
    try:
        xgb = XGBoostDetector(xgb_path)
        console.print("[green][OK][/green] XGBoost Loaded")
    except Exception as e:
        console.print(f"[red][FAIL][/red] XGBoost Failed: {e}")
        xgb = None
        
    try:
        deep = DeepAnomalyDetector(deep_dir)
        console.print("[green][OK][/green] Deep Autoencoder Loaded")
    except Exception as e:
        console.print(f"[red][FAIL][/red] Deep Autoencoder Failed: {e}")
        deep = None
        
    try:
        cnn = DeepCNNDetector(cnn_dir)
        console.print("[green][OK][/green] CNN Autoencoder Loaded")
    except Exception as e:
        console.print(f"[red][FAIL][/red] CNN Autoencoder Failed: {e}")
        cnn = None
        
    try:
        trans = TransformerAnomalyDetector(model_dir=trans_dir)
        console.print("[green][OK][/green] Transformer Model Loaded")
    except Exception as e:
        console.print(f"[red][FAIL][/red] Transformer Model Failed: {e}")
        trans = None
        
    try:
        repo = FlightRepository(DbConfig(path=db_path))
        rule_engine = AnomalyRuleEngine(repo, rules_path)
        console.print("[green][OK][/green] Rule Engine Loaded")
    except Exception as e:
        console.print(f"[red][FAIL][/red] Rule Engine Failed: {e}")
        rule_engine = None
        
    # 2. Load Test Data
    console.print("\n[bold]2. Loading Test Flights...[/bold]")
    
    # Load 5 Normal Flights
    repo_normal = FlightRepository(DbConfig(path=db_path, table="flight_tracks"))
    normal_flights = list(repo_normal.iter_flights(limit=5, min_points=50))
    
    # Load 5 Anomalous Flights
    repo_anom = FlightRepository(DbConfig(path=db_path, table="anomalous_tracks"))
    anom_flights = list(repo_anom.iter_flights(limit=5, min_points=50))
    
    all_flights = normal_flights + anom_flights
    labels = ["Normal"] * len(normal_flights) + ["Anomaly"] * len(anom_flights)
    
    # 3. Evaluate
    console.print("\n[bold]3. Evaluation Results[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Flight ID", width=10)
    table.add_column("Type", width=8)
    table.add_column("Rules", width=8)
    table.add_column("XGBoost", width=8)
    table.add_column("Deep AE", width=8)
    table.add_column("CNN", width=8)
    table.add_column("Transf.", width=8)
    table.add_column("Decision", style="bold")
    
    for flight, true_label in zip(all_flights, labels):
        # Rules
        rule_res = rule_engine.evaluate_flight(flight.flight_id)
        rule_hit = len(rule_res.get("matched_rules", [])) > 0
        
        # XGBoost
        xgb_res = xgb.predict(flight) if xgb else {}
        xgb_score = xgb_res.get("score", 0.0)
        xgb_anom = xgb_res.get("is_anomaly", False)
        
        # Deep AE
        deep_res = deep.predict(flight) if deep else {}
        deep_score = deep_res.get("score", 0.0)
        deep_anom = deep_res.get("is_anomaly", False)
        
        # CNN
        cnn_res = cnn.predict(flight) if cnn else {}
        cnn_score = cnn_res.get("score", 0.0)
        cnn_anom = cnn_res.get("is_anomaly", False)
        
        # Transformer
        trans_score, trans_anom = (0.0, False)
        if trans:
            try:
                trans_score, trans_anom = trans.predict(flight)
            except Exception as e:
                print(f"Transformer Error on {flight.flight_id}: {e}")
        
        # Formatting
        rule_str = "[red]YES[/red]" if rule_hit else "[green]NO[/green]"
        xgb_str = f"[red]{xgb_score:.2f}[/red]" if xgb_anom else f"[green]{xgb_score:.2f}[/green]"
        deep_str = f"[red]{deep_score:.3f}[/red]" if deep_anom else f"[green]{deep_score:.3f}[/green]"
        cnn_str = f"[red]{cnn_score:.3f}[/red]" if cnn_anom else f"[green]{cnn_score:.3f}[/green]"
        trans_str = f"[red]{trans_score:.3f}[/red]" if trans_anom else f"[green]{trans_score:.3f}[/green]"
        
        # Consensus
        votes = sum([rule_hit, xgb_anom, deep_anom, cnn_anom, trans_anom])
        consensus = "[bold red]ANOMALY[/bold red]" if votes >= 2 else "[green]NORMAL[/green]"
        
        table.add_row(
            flight.flight_id,
            true_label,
            rule_str,
            xgb_str,
            deep_str,
            cnn_str,
            trans_str,
            consensus
        )
        
    console.print(table)

if __name__ == "__main__":
    main()

