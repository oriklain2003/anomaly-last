# Flight Anomaly Detection

Hybrid detection stack (rules + ML) for FR24 flight data over Israel:

- `service/rules`: deterministic checks (squawk changes, steep turns, go-arounds…)
- `service/ml`: feature extraction + IsolationForest baseline
- `windowed_flight_tracker.py`: data collector that builds/updates `flight_tracks.db`

## 1. Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

Environment variables used by `windowed_flight_tracker.py` live in `config.json` (API token, bounds, DB path).

## 2. Collect flight tracks

```bash
python windowed_flight_tracker.py
```

This fills the configured SQLite database (`flight_tracks.db` by default) with ADS-B points for each flight intersecting the bounding box.

## 3. Run the rule engine

Evaluate the 10 anomaly rules for a given flight:

```bash
python -m service.rules.run_rules \
  --db flight_tracks.db \
  --rules service/rules/rule_config.json \
  --flight-id <FR24_FLIGHT_ID>
```

Optional: `--metadata planned.json` if you have a planned destination/route for diversion detection.

## 4. Train & evaluate the IsolationForest baseline

```bash
python -m service.ml.run \
  --db flight_tracks.db \
  --limit 500 \
  --model-path models/iforest.joblib \
  --normalizer-path models/normalizers.json
```

- Splits flights 60/40 (time-ordered) for train/test.
- Extracts engineered kinematic features (turn rate, deltas, phase one-hot).
- Prints precision/recall under the assumption the dataset is all “normal”.
- Saves the trained model + normalization stats to the supplied paths.

## 5. Repository layout

```
service/
  core/        # shared models, DB access, geo helpers, config
  rules/       # rule logic, CLI, JSON thresholds
  ml/          # feature extraction, normalization, IsolationForest CLI
windowed_flight_tracker.py  # FR24 ingestion
anomaly_rule.json           # rule descriptions (hebrew text)
```

## 6. Next steps

- Wire `service.rules.rule_engine.AnomalyRuleEngine` and the saved IF model into a FastAPI scoring service.
- Extend `service/ml` with the LSTM sequence model described in `Flight anomaly detection model.txt`.


