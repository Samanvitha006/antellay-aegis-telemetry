import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION & CORPORATE CSS
# ==========================================
st.set_page_config(
    page_title="Antellay Space | AEGIS Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Corporate Dark Theme CSS
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Branding */
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
        font-size: 1.2em;
        font-weight: 400;
        color: #94A3B8;
        margin-bottom: 30px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1E293B;
    }

    /* Divider */
    hr {
        border-color: #1E293B;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. CACHED DATA & AI PIPELINE
# ==========================================
@st.cache_data
def run_telemetry_pipeline():
    np.random.seed(42)
    n_minutes = 2880
    timestamps = pd.date_range(start="2026-09-19 00:00:00", periods=n_minutes, freq="min")
    minute_of_orbit = np.array([i % 90 for i in range(n_minutes)])
    is_sunlit = minute_of_orbit < 60

    # Base nominal data
    solar = np.where(is_sunlit, 180 + 10 * np.sin(minute_of_orbit / 10) + np.random.normal(0, 2, n_minutes), 0.0)
    power = 130 + np.random.normal(0, 3, n_minutes)
    curr = np.where(is_sunlit, (solar - power) / 28.0, 0.0) - np.where(~is_sunlit, power / 28.0,
                                                                       0.0) + np.random.normal(0, 0.2, n_minutes)
    volt = 28.0 + 2.5 * np.sin(minute_of_orbit / 15) + np.random.normal(0, 0.1, n_minutes)
    temp = np.where(is_sunlit, 22.0 + 4.0 * np.sin(minute_of_orbit / 20),
                    12.0 - 2.0 * np.cos(minute_of_orbit / 15)) + np.random.normal(0, 0.3, n_minutes)

    df = pd.DataFrame({
        "Timestamp": timestamps, "Temperature_C": temp, "Battery_Voltage_V": volt,
        "Battery_Current_A": curr, "Power_Consumption_W": power, "Solar_Panel_Output_W": solar
    })

    # Inject anomaly scenario
    df.loc[2600:2640, "Temperature_C"] += np.linspace(5, 35, 41)
    df.loc[2600:2640, "Battery_Voltage_V"] -= np.linspace(1, 8, 41)
    df.loc[2600:2640, "Power_Consumption_W"] += np.linspace(10, 50, 41)

    features = ["Temperature_C", "Battery_Voltage_V", "Battery_Current_A", "Power_Consumption_W",
                "Solar_Panel_Output_W"]

    train_end, val_end = int(n_minutes * 0.70), int(n_minutes * 0.85)
    train_df, val_df, test_df = df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_val = scaler.transform(val_df[features])
    X_test = scaler.transform(test_df[features])

    model = IsolationForest(n_estimators=150, random_state=42, contamination=0.01).fit(X_train)
    dynamic_threshold = np.percentile(model.decision_function(X_val), 1.0)

    test_scores = model.decision_function(X_test)
    test_df['Predicted_Anomaly'] = (test_scores < dynamic_threshold).astype(int)
    test_df['Anomaly_Score'] = np.round(-test_scores, 3)

    score_p98, score_p90 = np.percentile(test_df['Anomaly_Score'], 98), np.percentile(test_df['Anomaly_Score'], 90)
    test_df["Severity"] = test_df.apply(lambda r: "Nominal" if r["Predicted_Anomaly"] == 0 else (
        "High" if r["Anomaly_Score"] > score_p98 else ("Medium" if r["Anomaly_Score"] > score_p90 else "Low")), axis=1)

    means, stds = train_df[features].mean(), train_df[features].std().replace(0, 1e-6)

    def find_culprit(row):
        return "None" if row["Predicted_Anomaly"] == 0 else np.abs((row[features] - means) / stds).idxmax()

    test_df["Root_Cause_Parameter"] = test_df.apply(find_culprit, axis=1)

    def get_diagnostics(row):
        if row['Predicted_Anomaly'] == 0: return "Nominal operations.", "Maintain standard mission profile."
        param = row['Root_Cause_Parameter']
        if param == "Temperature_C":
            return "Critical thermal variance detected. High correlation with voltage anomalies.", "SYSTEM ACTION: Isolate primary battery subsystem. Initiate Safe Mode protocol."
        elif param == "Battery_Voltage_V":
            return "Severe voltage degradation.", "SYSTEM ACTION: Inspect for unmetered load or parasitic drain."
        return f"Unexpected deviation in {param}.", "SYSTEM ACTION: Review raw telemetry matrices."

    test_df[["Possible_Interpretation", "System_Recommendation"]] = test_df.apply(get_diagnostics, axis=1,
                                                                                  result_type="expand")
    return test_df


test_df = run_telemetry_pipeline()
anomalies_df = test_df[test_df["Predicted_Anomaly"] == 1]
latest_data = test_df.iloc[-1]

# ==========================================
# 3. HEADER & ENTERPRISE BRANDING
# ==========================================
st.markdown("<div class='company-logo'>ANTELLAY SPACE</div>", unsafe_allow_html=True)
st.markdown("<div class='product-title'>AEGIS</div>", unsafe_allow_html=True)
st.markdown("<div class='product-subtitle'>AI-Based Satellite Health Monitoring & Anomaly Detection</div>",
            unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🎛️ Operations Console")
    st.divider()
    selected_sensor = st.selectbox(
        "Select Telemetry Stream",
        ["Temperature_C", "Battery_Voltage_V", "Power_Consumption_W", "Solar_Panel_Output_W"],
        index=0
    )
    severity_filter = st.multiselect(
        "Filter Anomalies by Severity",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )
    st.divider()
    st.caption("© 2026 Antellay Space Systems. All rights reserved.")

filtered_anomalies = anomalies_df[anomalies_df["Severity"].isin(severity_filter)]

# Status Banner
if not anomalies_df.empty:
    st.error(
        f"⚠️ **ATTENTION REQUIRED:** {len(anomalies_df)} anomalies detected in current telemetry window. Immediate review recommended.")
else:
    st.success("✅ **SYSTEM NOMINAL:** All spacecraft subsystems operating within defined parameters.")

# Executive KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ Thermal Core", f"{latest_data['Temperature_C']:.2f} °C", "+0.2°C")
col2.metric("⚡ Bus Voltage", f"{latest_data['Battery_Voltage_V']:.2f} V", "-0.05V")
col3.metric("🔋 Power Load", f"{latest_data['Power_Consumption_W']:.2f} W", "+1.2W")
col4.metric("☀️ Array Output", f"{latest_data['Solar_Panel_Output_W']:.2f} W", "0.0W")

st.divider()

# ==========================================
# 4. INTERACTIVE TELEMETRY VIEW
# ==========================================
st.markdown(f"### 📡 Interactive Telemetry Analysis: `{selected_sensor}`")

fig = go.Figure()

# Corporate Blue Line for Nominal Data
fig.add_trace(go.Scatter(
    x=test_df['Timestamp'], y=test_df[selected_sensor],
    mode='lines', name='Baseline Telemetry',
    line=dict(color='#0EA5E9', width=2)
))

# Severe Red Markers for AI Anomalies
fig.add_trace(go.Scatter(
    x=anomalies_df['Timestamp'], y=anomalies_df[selected_sensor],
    mode='markers', name='AI Detection Flag',
    marker=dict(color='#EF4444', size=8, symbol='circle', line=dict(color='white', width=1)),
    hovertemplate="<b>Timestamp:</b> %{x}<br>" +
                  "<b>Reading:</b> %{y:.2f}<br>" +
                  "<b>Root Cause:</b> " + anomalies_df['Root_Cause_Parameter'] + "<br>" +
                  "<b>Severity:</b> " + anomalies_df['Severity'] + "<extra></extra>"
))

# Enterprise Dark Chart Layout
fig.update_layout(
    template="plotly_dark",
    plot_bgcolor='#0F172A',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(showgrid=True, gridcolor='#1E293B', title="Mission Elapsed Time (UTC)"),
    yaxis=dict(showgrid=True, gridcolor='#1E293B', title="Telemetry Value"),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. DIAGNOSTIC LOG & ACTION CENTER
# ==========================================
st.markdown("### 📋 AI Diagnostic Log & Triage")

if filtered_anomalies.empty:
    st.info("No anomalies match the current operational filters.")
else:
    for index, row in filtered_anomalies.tail(5).iterrows():
        # Corporate color coding for severity
        if row['Severity'] == "High":
            badge = "🔴 CRITICAL"
        elif row['Severity'] == "Medium":
            badge = "🟠 WARNING"
        else:
            badge = "🟡 ADVISORY"

        with st.expander(f"{badge} | Timestamp: {row['Timestamp']} | Fault Domain: {row['Root_Cause_Parameter']}"):
            st.markdown(f"**AI Confidence Score:** `{row['Anomaly_Score']}`")
            st.markdown(f"**Diagnostic Interpretation:** {row['Possible_Interpretation']}")
            st.markdown(f"**Recommended Mitigation:** `{row['System_Recommendation']}`")

    st.caption("Displaying top 5 most recent fault events. Expand rows for automated mitigation instructions.")