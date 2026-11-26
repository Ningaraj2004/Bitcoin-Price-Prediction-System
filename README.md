# ₿ Bitcoin Price Prediction System

A machine learning and deep learning-based application for Bitcoin price forecasting and market trend analysis using historical cryptocurrency data.

## 🚀 Live Demo

[View Live Demo](https://bitcoin-price-prediction-system.streamlit.app/)

## Overview

This project analyzes historical Bitcoin market data to predict next-day closing prices and market direction. It combines traditional machine learning algorithms with LSTM neural networks and presents the results through an interactive Streamlit dashboard.

## Features

- Bitcoin historical market data collection and preprocessing
- Technical indicator calculation (SMA, EMA, RSI, MACD)
- Next-day Bitcoin price prediction
- Market direction classification (UP/DOWN)
- Multiple machine learning model comparisons
- LSTM-based time-series forecasting
- Interactive market analysis dashboard
- Model evaluation with performance metrics and visualizations
- Automated testing and data leakage checks

## Technologies Used

| Category         | Technologies            |
| ---------------- | ----------------------- |
| Programming      | Python                  |
| Data Analysis    | Pandas, NumPy           |
| Machine Learning | Scikit-learn, XGBoost   |
| Deep Learning    | TensorFlow, Keras, LSTM |
| Visualization    | Plotly, Matplotlib      |
| Dashboard        | Streamlit               |
| Testing          | Pytest                  |

## Machine Learning Models

**Regression**

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LSTM Neural Network

**Classification**

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

## Project Structure

```text
bitcoin-price-prediction/
│
├── app/                  # Streamlit dashboard
├── config/               # Project configuration
├── data/                 # Raw and processed datasets
├── models/
│   └── artifacts/        # Trained models and metrics
├── notebooks/            # Exploratory data analysis
├── reports/              # Evaluation reports and charts
├── scripts/              # Data and training pipelines
├── src/                  # Core ML/DL implementation
├── tests/                # Automated tests
│
├── requirements.txt
├── README.md
└── PROJECT_SPEC.md
```

## Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/bitcoin-price-prediction.git
cd bitcoin-price-prediction
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

## Run the Application

Launch the interactive dashboard:

```bash
streamlit run app/app.py
```

Open the local URL displayed in your terminal to explore the application.

## Model Evaluation

Models are evaluated using chronological training, validation, and testing datasets.

| Task                  | Evaluation Metrics                       |
| --------------------- | ---------------------------------------- |
| Price Prediction      | MAE, MSE, RMSE, R²                       |
| Market Classification | Accuracy, Precision, Recall, F1, ROC-AUC |

The project also compares model performance against baseline predictions to provide a more meaningful evaluation.

## Testing

Run the automated test suite:

```bash
pytest -v
```

## Future Improvements

- Return-based price forecasting
- Sentiment analysis using cryptocurrency news
- Additional market indicators
- Improved time-series forecasting models
- Live cryptocurrency market monitoring

## Disclaimer

This project is developed for educational and research purposes only. Predictions are based on historical market data and should not be considered financial or investment advice. Cryptocurrency markets are highly volatile.

---

**Developed using Python, Machine Learning and Deep Learning.**
