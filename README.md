# 🟡 Gold Price Forecasting Dashboard

A Final Year Project focused on forecasting gold prices using a multivariate LSTM model, incorporating financial indices such as Crude Oil, S&P500, and USD Index. This project includes both model training in Jupyter notebooks and a deployed interactive dashboard built with Streamlit.

---

## 📊 Project Overview

The goal of this project is to develop a deep learning model that accurately predicts future gold prices using multiple economic indicators. The model is built using LSTM (Long Short-Term Memory) neural networks due to their effectiveness in time series forecasting.

The dashboard allows users to:

- Visualize historical gold price trends
- Interact with prediction results
- Understand the influence of related indices on gold prices

---

## 🗂️ Repository Structure

```bash
gold-price-forecasting-dashboard/
│
├── ModelTraining/
│   ├── FYP2 - Multivariate - 3 index.ipynb
│   ├── FYP2 - Multivariate - Crude Oil.ipynb
│   ├── FYP2 - Multivariate - S&P500.ipynb
│   ├── FYP2 - Multivariate - USD Index.ipynb
│   ├── FYP2 - Retrieve Dataset.ipynb
│   ├── dataset1.csv
│   ├── dataset2.csv
│   ├── dataset3.csv
│   └── dataset4.csv
│
├── dashboard.py          # Streamlit dashboard app
├── lstm_model.h5         # Trained LSTM model
├── scaler.pkl            # Data scaler for preprocessing
└── README.md             # Project description
```

---

## 🚀 Try the Dashboard

You can view the live app here:
👉 https://lbx-fyp.streamlit.app

---

## 🔧 Run Locally

1. Clone the repository

```bash
git clone https://github.com/bx0518/FYP-Gold-Price-Forecasting.git
cd FYP-Gold-Price-Forecasting
```

2. Install dependencies

```bash
pip install -r streamlit pandas numpy tensorflow scikit-learn matplotlib seaborn plotly joblib yfinance
```

3. Run the Streamlit app

```bash
streamlit run dashboard.py
```
