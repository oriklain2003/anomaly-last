# Anomaly Detection Pipeline Evaluation System

This module provides a comprehensive framework for evaluating the anomaly detection pipeline, comparing different layer combinations, and auditing the results via a dedicated UI.

## Components

1. **Dataset Config**: `evaluator/dataset_config.py` defines the lists of flight IDs for Normal, Anomaly, Glitch, and Borderline categories.
2. **Evaluator**: `evaluator/comprehensive_evaluator.py` runs the pipeline variations and collects metrics.
3. **Runner**: `runners/run_comprehensive_eval.py` executes the evaluation and generates a JSON report.
4. **Dashboard**: `ui/audit_dashboard.html` allows visual inspection of the results.

## How to Run Evaluation

1. **Configure Dataset**:
   Edit `service/llm_full_pipeline/evaluator/dataset_config.py` to add your manually curated flight IDs.

2. **Run Evaluation**:
   Execute the runner script from the root `service` directory:
   ```bash
   python service/llm_full_pipeline/runners/run_comprehensive_eval.py
   ```
   This will:
   - Fetch flight data (from DB or API).
   - Run 4 pipeline variations: Baseline, Filtering Only, LLM Only, Full Pipeline.
   - Generate a JSON report in `service/llm_full_pipeline/evaluator/reports/`.
   - Print a summary of metrics (Accuracy, Precision, Recall) to the console.

## How to View Results (Audit Dashboard)

1. **Start the UI Server**:
   ```bash
   python service/llm_full_pipeline/ui/serve_ui.py
   ```
   This starts a local web server on port 8085.

2. **Open Dashboard**:
   Navigate to [http://localhost:8085](http://localhost:8085) in your browser.

3. **Using the Dashboard**:
   - **Top Bar**: Select which evaluation report to view.
   - **Sidebar**: Filter flights by category (All, Anom, Norm, etc.) and search by ID.
   - **Main View**: Click a flight to see:
     - **Verdict Comparison**: See how each layer classified the flight.
     - **Map**: Visual flight track.
     - **Trace**: Detailed logic trace (points removed, flags raised, rules triggered).
     - **LLM Reasoning**: Full text explanation and suggested rule corrections.
     - **Tabs**: Inspect raw JSON output for each layer.

## Metrics Explained

- **Baseline**: The legacy `anomaly_pipeline.py` logic.
- **Filtering Only**: New `FilteringLayer` + Standard Detection + Heuristic Threshold.
- **LLM Only**: Raw Data + Standard Detection + LLM Reasoning.
- **Full Pipeline**: Filtering + Detection + LLM (The target architecture).

## File Structure

```
service/llm_full_pipeline/
├── evaluator/
│   ├── comprehensive_evaluator.py  # Core logic
│   ├── dataset_config.py           # Test data definitions
│   └── reports/                    # Output JSON files
├── runners/
│   └── run_comprehensive_eval.py   # CLI entry point
└── ui/
    ├── audit_dashboard.html        # Frontend
    └── serve_ui.py                 # Simple HTTP server
```



