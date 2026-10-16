# Antellay Space — AEGIS Spacecraft Telemetry Anomaly Detection

AEGIS is a spacecraft telemetry anomaly-detection and triage dashboard validated against the public **NASA Telemanom (SMAP Satellite & MSL Curiosity Rover)** benchmark dataset.

## 1. Setup & Reproducible Execution
```bash
pip install -r requirements.txt
# Run unit tests (missing values, invalid inputs, duplicate timestamps, empty datasets)
pytest tests/test_adapter.py
# Launch the Streamlit dashboard (automatically loads channel P-1 by default)
streamlit run app.py
## 2. Public Dataset Access & Provenance
* **Source**: Loaded dynamically via the Hugging Face Parquet mirror (`appleparan/telemanom`)[cite: 4] and the official ground-truth interval file (`khundman/telemanom/labeled_anomalies.csv`)[cite: 4].
* **Missions**: NASA SMAP Earth-observation satellite (55 channels) and MSL Curiosity rover (27 channels)[cite: 4]. Note: MSL is a rover benchmark used alongside SMAP for spacecraft telemetry validation[cite: 4].
* **Units & Scaling Caveat**: All telemetry values in Telemanom are anonymized and scaled (dimensionless)[cite: 4] and timestamps are anonymized relative step indices[cite: 4]. They do not represent live physical °C, V, or W engineering units[cite: 4].

## 3. Model Training & Leak-Free Thresholding
* **Rolling Robust Baseline**: Computes rolling median and Median Absolute Deviation (MAD) residuals[cite: 5] calibrated strictly on the held-out training split[cite: 5].
* **Isolation Forest (Unsupervised ML)**: Fits a multivariate Isolation Forest (`n_estimators=150`) across the primary telemetry channel (`value`) and spacecraft command states (`cmd_*`)[cite: 5]. Both `StandardScaler` parameters and percentile decision thresholds are fit exclusively on train data to prevent test data leakage[cite: 5].

## 4. Evaluation Methodology
Predictions on held-out test data are compared directly against the benchmark's `anomaly_sequences` intervals[cite: 5], reporting:
* Pointwise Precision, Recall, F1 Score, and False Positive Rate (FPR)[cite: 5].
* Sequence-level detection rate and Mean Detection Delay (timesteps between interval onset and first alert)[cite: 5].

## 5. Operational Boundaries & Limitations
* This system is validated on a historical research benchmark[cite: 5], not a live operating spacecraft feed[cite: 5].
* Root-cause attributions represent statistical hypotheses derived from correlated signal Z-scores[cite: 5] and require human flight-controller review before any operational action is taken[cite: 5].
## Step 6: Commit and Push to Update Your Live Streamlit App

Once `pytest` passes and `streamlit run app.py` loads locally in your browser, push your changes to GitHub:

```bash
git add requirements.txt telemanom_engine.py app.py tests/test_adapter.py README.md
git commit -m "Validate AEGIS against NASA Telemanom benchmark with evaluation metrics and unit tests"
git push origin main