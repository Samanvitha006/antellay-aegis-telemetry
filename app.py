import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from telemanom_engine import (
    BENCHMARK_CHANNELS,
    load_channel_data,
    run_detectors,
    evaluate_predictions,
)

# ==========================================
# 1. PAGE CONFIGURATION & CORPORATE CSS
# ==========================================
st.set_page_config(
    page_title="Antellay Space | AEGIS Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .company-logo {
        font-size: 1.1em;
        font-weight: 700;
        color: #38BDF8;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: -15px;
    }
    .product-title {
        font-size: 2.8em;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: -10px;
    }
    .product-subtitle {
        font-size: 1.15em;
        font-weight: 400;
        color: #94A3B8;
        margin-bottom: 20px;
    }
    .source-badge {
        background-color: #1E293B;
        border-left: 4px solid #38BDF8;
        padding: 10px 15px;
        border-radius: 4px;
        font-size: 0.9em;
        color: #CBD5E1;
        margin-bottom: 20px;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1E293B;
    }
    hr {
        border-color: #1E293B;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if "current_index" not in st.session_state:
    st.session_state.current_index = 250
if "stream_active" not in st.session_state:
    st.session_state.stream_active = False

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ Telemetry Source & Model")
    selected_channel = st.selectbox(
        "Select Telemanom Benchmark Channel",
        BENCHMARK_CHANNELS,
        index=0,
        help="Public channels from NASA SMAP satellite & MSL Curiosity rover.",
    )
    detector_type = st.selectbox(
        "Select Detector Model",
        [
            "Isolation Forest (Unsupervised ML)",
            "Rolling Robust Baseline (Statistical)",
        ],
        index=0,
    )
    threshold_pct = st.slider(
        "Training Tail Threshold Percentile (%)",
        min_value=0.5,
        max_value=10.0,
        value=1.0,
        step=0.5,
        help="Calibrated strictly on held-out training data to prevent test leakage.",
    )

    st.divider()
    st.markdown("### 📡 Display Mode")
    view_mode = st.radio(
        "Telemetry Inspection Mode",
        ["Full Held-Out Test Split", "Simulated Playback Window"],
        index=0,
    )

    if view_mode == "Simulated Playback Window":
        colA, colB = st.columns(2)
        if colA.button("▶️ Play", use_container_width=True):
            st.session_state.stream_active = True
            st.rerun()
        if colB.button("⏸️ Pause", use_container_width=True):
            st.session_state.stream_active = False
            st.rerun()

    st.divider()
    severity_filter = st.multiselect(
        "Filter Triage Log by Severity",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    st.caption("© 2026 Antellay Space Systems. Benchmark Validation Build.")

# ==========================================
# 4. CACHED DATA LOADING & MODEL EXECUTION
# ==========================================
@st.cache_data(show_spinner="Loading NASA Telemanom benchmark channel from Hugging Face...")
def get_evaluated_channel(chan_id: str, model_name: str, thresh: float):
    bundle = load_channel_data(chan_id)
    evaluated_test = run_detectors(
        bundle["train_df"],
        bundle["test_df"],
        bundle["feature_cols"],
        detector_type=model_name,
        threshold_pct=thresh,
    )
    metrics = evaluate_predictions(evaluated_test, bundle["anomaly_sequences"])
    return bundle, evaluated_test, metrics

bundle, test_df, eval_metrics = get_evaluated_channel(selected_channel, detector_type, threshold_pct)
primary_col = "value" if "value" in bundle["feature_cols"] else bundle["feature_cols"][0]

if view_mode == "Simulated Playback Window":
    if st.session_state.stream_active:
        if st.session_state.current_index < len(test_df) - 1:
            st.session_state.current_index += 25
        else:
            st.session_state.current_index = 250
    end_idx = min(len(test_df), st.session_state.current_index)
    start_idx = max(0, end_idx - 400)
    display_df = test_df.iloc[start_idx:end_idx]
else:
    st.session_state.stream_active = False
    display_df = test_df

anomalies_df = display_df[display_df["Predicted_Anomaly"] == 1]
filtered_anomalies = anomalies_df[anomalies_df["Severity"].isin(severity_filter)]
latest_row = display_df.iloc[-1]

# ==========================================
# 5. HEADER & DATA PROVENANCE DISCLOSURE
# ==========================================
st.markdown("<div class='company-logo'>ANTELLAY SPACE</div>", unsafe_allow_html=True)
st.markdown("<div class='product-title'>AEGIS</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='product-subtitle'>Spacecraft Telemetry Anomaly Detection — Public Benchmark Validation</div>",
    unsafe_allow_html=True,
)

mission_type = (
    "SMAP Earth-Observation Satellite"
    if bundle["spacecraft"] == "SMAP"
    else f"{bundle['spacecraft']} Spacecraft / Curiosity Rover"
)

st.markdown(
    f"<div class='source-badge'>"
    f"<b>DATA PROVENANCE DISCLOSURE:</b> Displaying <b>PUBLIC BENCHMARK TELEMETRY</b> "
    f"(Dataset: <code>NASA Telemanom / appleparan</code> | Mission: <b>{mission_type}</b> | "
    f"Channel: <code>{selected_channel}</code> | Features: <code>{len(bundle['feature_cols'])}</code>). "
    f"Values are <b>anonymized and scaled (dimensionless)</b> and timestamps are relative index steps. "
    f"Not a live spacecraft feed; do not interpret as physical °C, V, or W units."
    f"</div>",
    unsafe_allow_html=True,
)


# ==========================================
# 6. CLEAN EXECUTIVE METRICS & EVALUATION
# ==========================================
st.markdown("#### 📊 Live Channel Telemetry State")
col1, col2, col3 = st.columns(3)
col1.metric("Observed Value", f"{latest_row[primary_col]:.4f}")
col2.metric("Rolling Baseline", f"{latest_row['baseline_value']:.4f}")
col3.metric("Anomaly Score", f"{latest_row['Anomaly_Score']:.4f}")

st.markdown("#### 🎯 Model Evaluation Scorecard (Held-Out Test Split)")

# Use clean columns with custom formatting for the performance scorecard
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Precision", f"{eval_metrics['precision']:.3f}")
m2.metric("Recall", f"{eval_metrics['recall']:.3f}")
m3.metric("F1 Score", f"{eval_metrics['f1']:.3f}")
m4.metric("FPR", f"{eval_metrics['fpr']:.3f}")
m5.metric("Sequences Caught", f"{eval_metrics['detected_sequences']}/{eval_metrics['total_sequences']}")
m6.metric("Detection Delay", f"{eval_metrics['mean_detection_delay']}")

st.divider()

# ==========================================
# 7. INTERACTIVE TELEMETRY & INTERVAL PLOT
# ==========================================
st.markdown(
    f"### 📡 Telemetry Channel `{selected_channel}` ({bundle['spacecraft']}) — Model: `{detector_type}`"
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=display_df["timestep"],
        y=display_df["baseline_value"],
        mode="lines",
        name="Rolling Robust Baseline",
        line=dict(color="#64748B", width=1.5, dash="dot"),
    )
)

fig.add_trace(
    go.Scatter(
        x=display_df["timestep"],
        y=display_df[primary_col],
        mode="lines",
        name="Observed Telemetry (Scaled/Anonymized)",
        line=dict(color="#0EA5E9", width=2),
    )
)

if not anomalies_df.empty:
    fig.add_trace(
        go.Scatter(
            x=anomalies_df["timestep"],
            y=anomalies_df[primary_col],
            mode="markers",
            name="Predicted Anomaly Flag",
            marker=dict(color="#EF4444", size=8, symbol="circle", line=dict(color="white", width=1)),
            hovertemplate=(
                "<b>Timestep:</b> %{x}<br>"
                "<b>Scaled Value:</b> %{y:.4f}<br>"
                "<b>Score:</b> " + anomalies_df["Anomaly_Score"].astype(str) + "<br>"
                "<b>Severity:</b> " + anomalies_df["Severity"] + "<br>"
                "<b>Supporting Evidence:</b> " + anomalies_df["Supporting_Evidence"] + "<extra></extra>"
            ),
        )
    )

min_ts, max_ts = display_df["timestep"].min(), display_df["timestep"].max()
for idx, seq in enumerate(bundle["anomaly_sequences"]):
    s_start, s_end = int(seq[0]), int(seq[1])
    if s_end >= min_ts and s_start <= max_ts:
        fig.add_vrect(
            x0=max(s_start, min_ts),
            x1=min(s_end, max_ts),
            fillcolor="rgba(234, 179, 8, 0.20)",
            layer="below",
            line_width=1,
            line_color="rgba(234, 179, 8, 0.6)",
            annotation_text=f"Ground Truth #{idx+1}" if idx == 0 else "",
            annotation_position="top left",
        )

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#0F172A",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(showgrid=True, gridcolor="#1E293B", title="Anonymized Timestep Index (Held-Out Test Split)"),
    yaxis=dict(showgrid=True, gridcolor="#1E293B", title="Scaled Telemetry Value (Dimensionless)"),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, width='stretch')

# ==========================================
# 8. ROOT-CAUSE HYPOTHESIS & TRIAGE LOG
# ==========================================
st.markdown("### 📋 Root-Cause Triage & Correlated Signal Hypotheses")
st.caption(
    "All root-cause attributions below are statistical hypotheses derived from correlated channel/command "
    "deviations. Human engineering review is mandatory before any spacecraft action."
)

if filtered_anomalies.empty:
    st.info("No predicted anomalies match the current severity filters in this window.")
else:
    for _, row in filtered_anomalies.tail(8).iterrows():
        badge = (
            "🔴 HIGH SEVERITY"
            if row["Severity"] == "High"
            else ("🟠 MEDIUM SEVERITY" if row["Severity"] == "Medium" else "🟡 LOW SEVERITY")
        )
        gt_match = "✅ True Positive (Inside Ground-Truth Interval)" if row["ground_truth"] == 1 else "⚠️ False Positive (Outside Ground-Truth Interval)"

        with st.expander(
            f"{badge} | Timestep: {int(row['timestep'])} | Channel: {selected_channel} | {gt_match}"
        ):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Observed / Baseline:** `{row[primary_col]:.4f}` / `{row['baseline_value']:.4f}`")
            c2.markdown(f"**Anomaly Score:** `{row['Anomaly_Score']:.4f}` (Threshold: `{row['Threshold_Used']}`)")
            c3.markdown(f"**Primary Channel |Z|-Score:** `{row['Primary_Z_Score']}`")

            st.markdown(f"**Supporting Correlated Evidence:** `{row['Supporting_Evidence']}`")
            st.markdown(f"**Diagnostic Hypothesis:** {row['Hypothesis']}")
            st.warning(f"**Operational Gate:** {row['Triage_Action']}")

if st.session_state.stream_active:
    time.sleep(0.6)
    st.rerun()



 
