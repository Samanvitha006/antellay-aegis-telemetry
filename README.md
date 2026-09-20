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
# 2. Install required dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit dashboard
streamlit run app.py
