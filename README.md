# AEGIS: AI-Based Satellite Health Monitoring 🛰️

This repository contains the prototype for the ANTELLAY Space technical assignment. It features a complete machine learning pipeline (Isolation Forest) for multivariate telemetry anomaly detection and an interactive Streamlit dashboard for operators.

## 🎥 Live Demonstration
**[Insert the link to your 2-minute Loom/Screen Recording video here]**

## 📁 Project Structure
* `1_EDA_and_Model_Training.ipynb`: Jupyter notebook containing data generation, preprocessing, model training, and root-cause analysis.
* `app.py`: The interactive Streamlit dashboard code.
* `requirements.txt`: Python package dependencies.

## ⚙️ Local Setup & Execution Instructions

To run this project locally, please follow these steps to set up a clean Python environment. Python 3.9+ is recommended.
**1. Clone the repository**
```bash
git clone https://github.com/Samanvitha006/antellay-aegis-telemetry.git
cd antellay-aegis-telemetry
graph TD
    A[<b>1. PROBLEM ANALYSIS</b><br>• Literature Survey<br>• Data Needs Identification<br>• Data Exploration] --> B
    B[<b>2. DATA PREPARATION</b><br>• Data Generation<br>• Data Cleansing<br>• Feature Scaling<br>• Data Split] --> C
    C[<b>3. TRAINING MODEL</b><br>• Initialize Isolation Forest<br>• Fit to Training Data<br>• Extract Validation Scores<br>• Calculate Dynamic Threshold] --> D
    D[<b>4. MODEL EVALUATION</b><br>• Predict on Test Data<br>• Isolate Multivariate Anomalies<br>• Calculate Z-Scores for Root Cause<br>• Assign Severity Classifications] --> E
    E[<b>5. OPERATIONAL DASHBOARD</b><br>• KPI Determination<br>• Dashboard Requirements<br>• Streamlit UI Integration]

    style A fill:#0B0F19,stroke:#0EA5E9,stroke-width:2px,color:#fff
    style B fill:#0B0F19,stroke:#0EA5E9,stroke-width:2px,color:#fff
    style C fill:#0B0F19,stroke:#0EA5E9,stroke-width:2px,color:#fff
    style D fill:#0B0F19,stroke:#0EA5E9,stroke-width:2px,color:#fff
    style E fill:#0B0F19,stroke:#0EA5E9,stroke-width:2px,color:#fff
