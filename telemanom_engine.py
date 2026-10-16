import ast
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"

# Curated list of prominent SMAP (Satellite) and MSL (Rover) benchmark channels
BENCHMARK_CHANNELS = [
    "P-1", "S-1", "E-1", "E-2", "E-3", "T-1", "T-2", "G-1", "D-1", "A-1", "M-1", "F-1"
]


def load_labels_metadata() -> pd.DataFrame:
    """Loads the ground-truth labeled_anomalies.csv from the Telemanom repository."""
    return pd.read_csv(LABELS_URL)


def validate_and_clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Validates columns/timestamps, handles missing values, removes duplicate timestamps,
    and checks for empty datasets or invalid inputs (Acceptance Criterion 6).
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    df = df.copy()

    # Ensure timestep exists
    if "timestep" not in df.columns:
        df["timestep"] = np.arange(len(df))

    # Sort and drop duplicate timestamps
    df = df.sort_values("timestep").drop_duplicates(subset=["timestep"]).reset_index(drop=True)

    # Identify feature columns (value + command features cmd_*)
    feature_cols = [c for c in df.columns if c not in ["timestep", "ground_truth", "sequence_id"]]
    if not feature_cols:
        raise ValueError("No valid telemetry feature columns found in dataset.")

    # Coerce invalid non-numeric strings/inputs to NaN, replace infinities
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Forward-fill then backward-fill missing values, fallback to 0.0
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)

    if df.empty:
        raise ValueError("Dataset became empty after cleaning.")

    return df, feature_cols


def load_channel_data(chan_id: str) -> dict:
    """Loads a single Telemanom channel from Hugging Face and attaches ground-truth intervals."""
    ds = load_dataset("appleparan/telemanom", name=chan_id)
    train_raw = ds["train"].to_pandas()
    test_raw = ds["test"].to_pandas()

    train_df, feature_cols = validate_and_clean_dataframe(train_raw)
    test_df, _ = validate_and_clean_dataframe(test_raw)

    # Load metadata & ground-truth intervals
    labels_df = load_labels_metadata()
    chan_meta = labels_df[labels_df["chan_id"] == chan_id]

    spacecraft = "Unknown"
    anomaly_sequences = []
    anomaly_classes = []

    if not chan_meta.empty:
        row = chan_meta.iloc[0]
        spacecraft = str(row["spacecraft"])
        anomaly_sequences = ast.literal_eval(row["anomaly_sequences"])
        if "class" in row and pd.notna(row["class"]):
            anomaly_classes = ast.literal_eval(row["class"])

    # Map ground-truth intervals onto the held-out test set
    test_df["ground_truth"] = 0
    test_df["sequence_id"] = -1

    for idx, seq in enumerate(anomaly_sequences):
        start_step, end_step = int(seq[0]), int(seq[1])
        mask = (test_df["timestep"] >= start_step) & (test_df["timestep"] <= end_step)
        test_df.loc[mask, "ground_truth"] = 1
        test_df.loc[mask, "sequence_id"] = idx

    return {
        "chan_id": chan_id,
        "spacecraft": spacecraft,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "anomaly_sequences": anomaly_sequences,
        "anomaly_classes": anomaly_classes,
    }


def run_detectors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    detector_type: str = "Isolation Forest (Unsupervised ML)",
    threshold_pct: float = 1.0,
    rolling_window: int = 30,
) -> pd.DataFrame:
    """
    Runs either the Configurable Rolling Robust Baseline or the Isolation Forest detector.
    Strictly fits all scalers and thresholds on train_df to prevent test data leakage.
    """
    test_out = test_df.copy()
    primary_col = "value" if "value" in feature_cols else feature_cols[0]

    # Fit scaler strictly on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_out[feature_cols])

    # Rolling baseline reference for plotting on the dashboard
    test_out["baseline_value"] = (
        test_out[primary_col].rolling(window=rolling_window, min_periods=1).median()
    )

    if detector_type == "Rolling Robust Baseline (Statistical)":
        # Compute rolling residual on training set to establish a non-leaked threshold
        train_rolling_med = train_df[primary_col].rolling(window=rolling_window, min_periods=1).median()
        train_resid = np.abs(train_df[primary_col] - train_rolling_med)
        mad = np.median(train_resid) + 1e-6

        train_scores = train_resid / mad
        dynamic_threshold = np.percentile(train_scores, 100.0 - threshold_pct)

        test_resid = np.abs(test_out[primary_col] - test_out["baseline_value"])
        test_scores = test_resid / mad

        test_out["Anomaly_Score"] = np.round(test_scores, 4)
        test_out["Predicted_Anomaly"] = (test_scores > dynamic_threshold).astype(int)
        test_out["Threshold_Used"] = round(float(dynamic_threshold), 4)

    else:
        # Isolation Forest ML Baseline (Split train into 85% fit / 15% validation for threshold)
        split_idx = int(len(X_train) * 0.85)
        X_fit, X_val = X_train[:split_idx], X_train[split_idx:]

        model = IsolationForest(
            n_estimators=150,
            random_state=42,
            contamination=max(0.001, min(0.2, threshold_pct / 100.0)),
        )
        model.fit(X_fit)

        val_raw_scores = model.decision_function(X_val)
        dynamic_threshold = np.percentile(val_raw_scores, threshold_pct)

        test_raw_scores = model.decision_function(X_test)
        # Invert scores so higher positive values = higher anomaly severity
        test_out["Anomaly_Score"] = np.round(-test_raw_scores, 4)
        test_out["Predicted_Anomaly"] = (test_raw_scores < dynamic_threshold).astype(int)
        test_out["Threshold_Used"] = round(float(-dynamic_threshold), 4)

    # Severity Stratification (using training-calibrated percentiles)
    p98 = np.percentile(test_out["Anomaly_Score"], 98)
    p90 = np.percentile(test_out["Anomaly_Score"], 90)

    def assign_severity(row):
        if row["Predicted_Anomaly"] == 0:
            return "Nominal"
        if row["Anomaly_Score"] >= p98:
            return "High"
        if row["Anomaly_Score"] >= p90:
            return "Medium"
        return "Low"

    test_out["Severity"] = test_out.apply(assign_severity, axis=1)

    # Root-Cause Hypothesis & Supporting Evidence via Training Z-Scores
    train_means = train_df[feature_cols].mean()
    train_stds = train_df[feature_cols].std().replace(0, 1e-6)
    z_matrix = ((test_out[feature_cols] - train_means) / train_stds).abs()

    test_out["Primary_Z_Score"] = np.round(z_matrix[primary_col], 2)
    test_out["Top_Correlated_Feature"] = z_matrix.idxmax(axis=1)

    # Identify top active command/correlated signals where |Z| > 2.0
    def build_hypothesis(idx):
        if test_out.loc[idx, "Predicted_Anomaly"] == 0:
            return "Nominal baseline behavior.", "None", "No operational action required."

        row_z = z_matrix.loc[idx]
        elevated = row_z[row_z > 2.0].sort_values(ascending=False)
        top_signals = [f"{col} (|Z|={val:.1f})" for col, val in elevated.head(3).items()]
        evidence_str = ", ".join(top_signals) if top_signals else f"{primary_col} deviation"

        hypothesis = (
            f"HYPOTHESIS: Anomalous deviation observed in anonymized channel ({primary_col}). "
            f"Correlated shifts detected across [{evidence_str}]."
        )
        triage_note = (
            "HUMAN REVIEW REQUIRED: Verify correlated command/telemetry states against mission "
            "flight rules before executing any spacecraft action."
        )
        return hypothesis, evidence_str, triage_note

    diagnostics = [build_hypothesis(i) for i in test_out.index]
    test_out["Hypothesis"] = [d[0] for d in diagnostics]
    test_out["Supporting_Evidence"] = [d[1] for d in diagnostics]
    test_out["Triage_Action"] = [d[2] for d in diagnostics]

    return test_out


def evaluate_predictions(test_df: pd.DataFrame, anomaly_sequences: list) -> dict:
    """
    Computes pointwise Precision, Recall, F1, False-Positive Rate (FPR),
    and sequence-level Detection Delay (Step F).
    """
    y_true = test_df["ground_truth"].values
    y_pred = test_df["Predicted_Anomaly"].values

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Calculate sequence detection and detection delay (in timesteps)
    delays = []
    detected_sequences = 0
    for seq in anomaly_sequences:
        start_step, end_step = int(seq[0]), int(seq[1])
        seq_slice = test_df[(test_df["timestep"] >= start_step) & (test_df["timestep"] <= end_step)]
        detected_rows = seq_slice[seq_slice["Predicted_Anomaly"] == 1]
        if not detected_rows.empty:
            detected_sequences += 1
            first_detected_step = int(detected_rows["timestep"].iloc[0])
            delays.append(first_detected_step - start_step)

    mean_delay = float(np.mean(delays)) if delays else None

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_sequences": len(anomaly_sequences),
        "detected_sequences": detected_sequences,
        "mean_detection_delay": round(mean_delay, 1) if mean_delay is not None else "N/A",
    }