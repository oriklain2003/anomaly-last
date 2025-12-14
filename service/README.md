# FiveAir Anomaly Detection System
# מערכת זיהוי חריגות טיסה FiveAir

This project implements a multi-layer anomaly detection system for commercial flights, utilizing rule-based checks, supervised machine learning (XGBoost), and deep unsupervised learning (Autoencoders & CNNs).

---

## 🏗 Project Structure
## מבנה הפרויקט

- **`anomaly_pipeline.py`**: The main entry point. Orchestrates all detection layers and produces a unified JSON report.
- **`flight_fetcher.py`**: Utility to fetch live flight data from FR24 API or Playground.
- **`core/`**: Core database and model definitions (`FlightTrack`, `TrackPoint`).
- **`mlboost/`**: **Layer 2 (Supervised)**. XGBoost model trained on known anomalies.
  - `features.py`: Aggregates flight points into statistical features (mean, std, min, max).
  - `model.py`: XGBoost wrapper.
  - `detector.py`: Runtime detector class.
  - `run.py`: Training script.
- **`ml_deep/`**: **Layer 3 (Unsupervised Dense)**. Dense Autoencoder for route deviation detection.
  - `clustering.py`: Clusters flights into "Flows".
  - `model.py`: Simple Autoencoder.
  - `detector.py`: Runtime detector.
  - `train.py`: Training script.
- **`ml_deep_cnn/`**: **Layer 4 (Unsupervised CNN)**. 1D-CNN Autoencoder for fine-grained maneuver detection.
  - `model.py`: Convolutional Autoencoder.
  - `detector.py`: Runtime detector.
  - `train.py`: Training script.
- **`rules/`**: **Layer 1 (Rule Based)**. Basic physics and safety rules.

---

## 🧠 Detection Logic
## לוגיקת זיהוי

The system analyzes a flight through 4 layers of defense:
המערכת מנתחת כל טיסה דרך 4 שכבות הגנה:

### 1. Rule Engine (חוקים פיזיקליים)
Checks for impossible or immediately dangerous conditions.
- **Logic**: Negative altitude, supersonic speed, invalid squawk codes.
- **Status**: Manual checks implemented in pipeline.

### 2. XGBoost Classifier (זיהוי מבוסס דוגמאות - Supervised)
Detects "Known Anomalies" based on aggregate statistics.
- **Logic**: Calculates stats like "Standard Deviation of Turn Rate over 5 mins".
- **Training**: Trained on labeled dataset of Normal vs. Anomalous flights.
- **Strength**: Extremely accurate (99%) for known patterns like Holding Patterns or Erratic Turns.

### 3. Deep Dense Autoencoder (זיהוי סטייה מנתיב - Unsupervised)
Detects "Shape Deviations".
- **Logic**: Resamples flight to 50 points. Clusters it into a known "Flow" (e.g., TLV->EUR). Tries to reconstruct the shape.
- **Anomaly**: High reconstruction error means the flight shape does not match the cluster norm.
- **Strength**: Good at finding flights going to the wrong place.

### 4. Deep CNN Autoencoder (זיהוי תמרונים חריגים - Unsupervised)
Detects "Fine-Grained Anomalies" and "Texture".
- **Logic**: Uses 1D Convolutions to learn local patterns (jitters, sharp turns).
- **Anomaly**: Very high sensitivity to unusual maneuvers that aggregate stats might miss.
- **Strength**: Highest sensitivity (24x signal-to-noise ratio).

---

## ⚙️ Configuration & Usage
## הגדרות ושימוש

### Installation
```bash
pip install -r requirements.txt
```

### Running the Pipeline
```bash
python anomaly_pipeline.py
```
This will fetch a live test flight and run all detectors.

### Training Models (אימון מודלים)
To retrain the models on new data:

1. **XGBoost**:
   ```bash
   python -m mlboost.run --db last.db --anomalies-table anomalous_tracks
   ```
2. **Deep Models**:
   ```bash
   python -m ml_deep.train --db last.db
   python -m ml_deep_cnn.train --db last.db
   ```

### Testing Individual Layers
- `python mlboost/test_inference.py`
- `python ml_deep/test_inference.py`
- `python ml_deep_cnn/test_inference.py`

---

## 📊 API Output Format
## פורמט פלט

```json
{
  "layer_1_rules": { "status": "NORMAL" },
  "layer_2_xgboost": { 
      "status": "ANOMALY", 
      "score": 0.99, 
      "is_anomaly": true 
  },
  "layer_3_deep_dense": { 
      "status": "ANOMALY", 
      "severity": 13.0, 
      "cluster": 0 
  },
  "layer_4_deep_cnn": { 
      "status": "ANOMALY", 
      "severity": 24.8, 
      "cluster": 0 
  },
  "summary": {
      "is_anomaly": true,
      "triggers": ["XGBoost", "DeepDense", "DeepCNN"],
      "flight_id": "3bc6854c"
  }
}
```

