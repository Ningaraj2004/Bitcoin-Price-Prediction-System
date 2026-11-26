# ₿ Advanced Bitcoin Price Prediction & Market Analysis System

An end-to-end, production-quality platform that predicts Bitcoin's
**next-day closing price** (regression) and **next-day market direction**
(UP/DOWN classification) using classical machine learning, a deep-learning
LSTM, technical-indicator analysis, and an interactive Streamlit dashboard.

> ⚠️ **Financial-risk disclaimer:** this project is for education and
> research only. It is **not** financial advice. Cryptocurrency markets are
> extremely volatile; models trained on historical data can and will be
> wrong, sometimes catastrophically. Never invest money you cannot afford
> to lose, and never treat any output of this system as a trading signal.

---

## Table of Contents

1. [Project overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Technologies used](#technologies-used)
5. [Installation](#installation)
6. [Download data](#download-data)
7. [Train models](#train-models)
8. [Run tests](#run-tests)
9. [Launch the dashboard](#launch-the-dashboard)
10. [Model descriptions](#model-descriptions)
11. [Evaluation methodology](#evaluation-methodology)
12. [Measured results](#measured-results)
13. [Screenshots](#screenshots)
14. [GitHub upload](#github-upload)
15. [Free deployment (Streamlit Community Cloud)](#free-deployment-streamlit-community-cloud)
16. [Known limitations](#known-limitations)
17. [Financial-risk disclaimer](#financial-risk-disclaimer)

---

## Project overview

The system covers the complete machine-learning lifecycle:

| Stage | What it does |
| --- | --- |
| **Data collection** | Downloads BTC-USD daily OHLCV from Yahoo Finance (yfinance), validates quality, handles missing values/duplicates, sorts chronologically, and excludes the still-forming current-day candle |
| **Feature engineering** | 22 leak-free features: SMA 7/14/30, EMA 12/26, RSI 14, MACD + signal, daily returns, rolling volatility, lagged closes/volumes, high-low range, open-close change |
| **Modeling** | 3 regression models (Linear Regression, Random Forest, XGBoost), 3 classifiers (Logistic Regression, Random Forest, XGBoost), 1 LSTM regressor (TensorFlow/Keras) |
| **Evaluation** | Chronological train/val/test splits, honest baselines (persistence / majority-class), full metric suites, comparison tables and charts saved to `reports/` |
| **Prediction** | A `BitcoinPredictor` façade loads saved artifacts (never retrains) and produces next-day price and direction predictions with probabilities |
| **Dashboard** | 7-page interactive Streamlit app: market overview, candlestick history, technical indicators, price prediction, direction prediction, model comparison, model evaluation |

## Features

- ✅ Real BTC-USD data via `yfinance` — no fabricated prices, ever
- ✅ Leak-free feature engineering (verified by automated tests)
- ✅ Chronological splits — time-series data is **never shuffled**
- ✅ Preprocessing fitted on training data only (scalers live inside sklearn Pipelines)
- ✅ Persistence & majority-class baselines for honest comparison
- ✅ LSTM with early stopping on a chronological validation set
- ✅ Graceful degradation: the app runs fine if the LSTM or any artifact is missing
- ✅ Cached data/model loading — no retraining on dashboard load
- ✅ 73 automated pytest tests (leakage, alignment, training, prediction, app)
- ✅ Genuine evaluation reports & figures generated from real predictions
- ✅ Relative paths everywhere — deployable as-is

## Architecture

```
bitcoin-price-prediction/
├── config/                  # Central settings: paths, hyperparameters, logging
│   └── settings.py
├── data/
│   ├── raw/                 # Untouched Yahoo Finance download (btc_usd_raw.csv)
│   └── processed/           # Cleaned OHLCV + engineered feature dataset
├── notebooks/               # Exploratory analysis starter notebook
├── src/btc_prediction/      # Core library package
│   ├── data/collector.py            # fetch / validate / clean / save / load
│   ├── features/indicators.py       # SMA, EMA, RSI, MACD, volatility, lags…
│   ├── features/build_features.py   # feature matrix + Next_Day_Close / Direction
│   ├── models/                      # sklearn / XGBoost / LSTM definitions
│   ├── training/                    # chronological splits + training pipelines
│   ├── evaluation/                  # metrics, baselines, comparison reports, figures
│   └── prediction/predictor.py      # artifact-based prediction façade
├── app/                     # Streamlit dashboard
│   ├── app.py                       # entry point (streamlit run app/app.py)
│   └── components/                  # charts, cached loaders, page sections
├── models/artifacts/        # Saved .joblib models, .keras LSTM, metrics JSON
├── reports/                 # Comparison tables (CSV), figures (PNG), summary JSON
├── scripts/                 # CLI pipeline: download → features → train → run all
├── tests/                   # pytest suite (offline, deterministic fixtures)
├── .streamlit/config.toml   # Theme + server configuration
├── requirements.txt
├── PROJECT_SPEC.md          # Detailed system specification
└── README.md
```

## Technologies used

| Category | Technology |
| --- | --- |
| Language | Python 3.11 / 3.12 |
| Data | pandas, NumPy, yfinance |
| Machine learning | scikit-learn, XGBoost, joblib |
| Deep learning | TensorFlow / Keras (LSTM) |
| Visualization | Plotly (dashboard), Matplotlib (report figures) |
| Dashboard | Streamlit |
| Testing | pytest |

## Installation

```bash
# 1. Clone / unzip the project
cd bitcoin-price-prediction

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# CPU-only machines can use the lighter TensorFlow build:
pip install tensorflow-cpu
```

## Download data

```bash
python scripts/download_data.py            # uses cache when present
python scripts/download_data.py --refresh  # force re-download from Yahoo
```

Raw and processed CSVs are stored separately under `data/raw/` and
`data/processed/`. The pipeline validates the download (row counts, missing
values, duplicate dates, high ≥ low, positive prices) and drops the
incomplete candle of the current day before anything is trained.

## Train models

```bash
python scripts/train_models.py          # all models incl. LSTM
python scripts/train_models.py --skip-lstm   # classical ML only (faster)
```

Or run the whole pipeline in one command:

```bash
python scripts/run_pipeline.py          # download → features → train → smoke test
```

Artifacts land in `models/artifacts/`, evaluation tables and figures in
`reports/`. Training metadata (dates, splits, metrics) is written to
`models/artifacts/training_metadata.json`.

## Run tests

```bash
pytest                       # full suite (offline, uses synthetic data)
pytest -v                    # verbose
pytest tests/test_leakage.py # just the leakage checks
```

The LSTM tests are skipped automatically if TensorFlow is not installed.

## Launch the dashboard

```bash
streamlit run app/app.py
```

Then open the printed local URL (default `http://localhost:8501`). The
sidebar provides page navigation, a **Refresh market data** button, project
information, and per-page model selection.

## Model descriptions

### Regression — next-day closing price

| Model | Notes |
| --- | --- |
| **Linear Regression** | StandardScaler (fit on train only) inside a Pipeline; interpretable linear baseline |
| **Random Forest** | 300 trees, `min_samples_leaf=2`; captures non-linear interactions |
| **XGBoost** | 400 boosting rounds, lr 0.05, depth 6, histogram tree method |
| **LSTM (Keras)** | Lookback 60 days → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1); MinMax scaling fit on train; Adam(1e-3), MSE loss, early stopping (patience 10) restoring best weights |

### Classification — UP/DOWN direction

| Model | Notes |
| --- | --- |
| **Logistic Regression** | Scaled Pipeline; outputs calibrated-ish probabilities |
| **Random Forest** | 300 trees; `predict_proba` for the dashboard gauge |
| **XGBoost** | 300 rounds, lr 0.05, depth 4, logistic objective |

### Targets

- `Next_Day_Close = Close.shift(-1)` (regression)
- `Direction = 1` if next-day close > today's close else `0` (classification)

## Evaluation methodology

1. **Chronological 70/15/15 split** into train / validation / test. The test
   set is strictly the most recent ~15% of days — the models never see it
   during training, and the split is never shuffled.
2. **Selection on validation, reporting on test.** The "best" model is chosen
   by validation performance; all headline numbers shown in the dashboard
   and reports are computed on the untouched test set.
3. **Baselines.** Regression is compared to the *persistence baseline*
   ("tomorrow = today"); classification to the *majority-class baseline*.
   A model that cannot beat its baseline is reported as such — no sugarcoating.
4. **Metrics.** Regression: MAE, MSE, RMSE, R². Classification: accuracy,
   precision, recall, F1, ROC-AUC, confusion matrix.
5. **Leakage control.** Every feature at day *t* uses only data from days
   ≤ *t*; automated tests perturb future data and assert past features do
   not change. Scalers are fitted on the training block only.
6. **All reported numbers are computed from real predictions** stored in
   `models/artifacts/*_metrics.json`. Nothing is hard-coded.

## Measured results

The numbers below were **actually measured** on 2026-11-26 with data
2014-09-17 → 2026-11-24 (4,421 model rows; chronological test split
2025-01-29 → 2026-11-23, 664 days). They are reproduced in
`models/artifacts/training_metadata.json`, `reports/evaluation_summary.json`
and on the dashboard's Model Comparison page — every run regenerates them.

**Regression — next-day close (test set):**

| Model | MAE (USD) | RMSE (USD) | R² |
| --- | --- | --- | --- |
| Persistence Baseline | 1,367 | 1,935 | 0.9880 |
| Linear Regression | 1,398 | 1,965 | 0.9876 |
| Random Forest | 23,774 | 29,424 | −1.7690 |
| XGBoost | 28,787 | 33,921 | −2.6802 |
| LSTM | 12,718 | 14,538 | 0.3224 |

Honest reading of these results:

- Next-day close is dominated by today's close, so the *persistence
  baseline* is extremely strong; Linear Regression essentially matches it.
- Random Forest / XGBoost / LSTM operate on **price levels** and cannot
  extrapolate beyond the price range seen in training (2025-2026 prices
  left the training range), which is the classic tree/NN regression trap
  for trending financial series. This is a genuine, well-known limitation —
  we report it rather than hide it. Predicting *returns* instead of levels
  is the standard fix and a natural next step.
- The reported figures are exactly why the project insists on baselines:
  without them, an R² of −2.7 would be invisible.

**Classification — UP/DOWN direction (test set):**

| Model | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- |
| Majority-Class Baseline | 0.486 | 0.655 | n/a |
| Logistic Regression | 0.502 | 0.103 | 0.493 |
| Random Forest | 0.505 | 0.035 | 0.488 |
| XGBoost | 0.514 | 0.277 | 0.500 |

Directional accuracy sits at ~50-51% against a 48.6% majority baseline and
ROC-AUC ≈ 0.50: **for daily BTC direction, the models add essentially no
tradeable edge**, which is the realistic outcome for this problem and the
honest conclusion of this project.

Test suite: **73 tests, all passing** (including Streamlit AppTest runs of
all seven dashboard pages).

## Screenshots

*(placeholders — replace after launching locally)*

| Page | Screenshot |
| --- | --- |
| Market Overview | `docs/screenshots/market_overview.png` |
| Historical Price Analysis | `docs/screenshots/historical.png` |
| Technical Indicators | `docs/screenshots/indicators.png` |
| Bitcoin Price Prediction | `docs/screenshots/price_prediction.png` |
| Market Direction Prediction | `docs/screenshots/direction.png` |
| Model Comparison | `docs/screenshots/comparison.png` |
| Model Evaluation | `docs/screenshots/evaluation.png` |

## GitHub upload

```bash
cd bitcoin-price-prediction
git init
git add .
git commit -m "Bitcoin price prediction system: data, features, ML+LSTM, dashboard, tests"
git branch -M main
git remote add origin https://github.com/<your-user>/bitcoin-price-prediction.git
git push -u origin main
```

Recommended: add a short description + topics (`bitcoin`, `streamlit`,
`machine-learning`, `lstm`, `time-series`) on the repo page, and drop your
screenshots into `docs/screenshots/` before pushing.

## Free deployment (Streamlit Community Cloud)

1. Push the repository to GitHub (steps above).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, **main file path**: `app/app.py`.
4. Deploy. That's it — no API keys or secrets are needed:
   - all paths are computed relative to the project root,
   - data is downloaded at runtime via yfinance (and a small CSV snapshot is
     committed as an offline fallback),
   - model artifacts are committed in `models/artifacts/` (a few MB),
   - missing artifacts produce friendly in-app guidance instead of crashes.

Keep the repository lean: `.gitignore` excludes venvs, caches, notebooks
checkpoints and ZIPs; the committed CSVs (~10 years of daily data) are only
a few hundred KB.

## Known limitations

- **Direction prediction is hard.** Daily crypto direction is close to a
  coin flip; treat classification metrics above the majority baseline as a
  small, not tradable, edge.
- **Level-based models cannot extrapolate.** Random Forest, XGBoost and the
  LSTM are trained on price *levels*; when real prices move outside the
  training range their predictions degrade sharply (see Measured results).
  Predicting returns instead of levels is the standard remedy.
- **No transaction costs, slippage or liquidity modelling** — outputs are
  not backtested strategies.
- **Daily granularity only.** Intraday dynamics, order-book state, funding
  rates and on-chain features are out of scope.
- **No external features** (macro data, sentiment, halving cycles).
- **Yahoo Finance data quirks** — occasional gaps/candles are cleaned, but
  upstream revisions can slightly change historical values between refreshes.
- **LSTM needs TensorFlow**; on machines without it the app degrades
  gracefully and hides the LSTM option.
- The current-day candle is excluded by design, so "latest close" can be
  one day behind a live exchange feed.

## Financial-risk disclaimer

This software is provided **for educational purposes only**, with no
warranty of any kind (see spirit of the MIT license in the repo). Nothing
in this repository constitutes financial, investment, or trading advice.
Past performance is not indicative of future results. Crypto-asset trading
can result in the total loss of your capital. Consult a licensed financial
advisor before making investment decisions, and independently verify any
data or output produced by this system.
