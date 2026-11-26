# PROJECT_SPEC — Advanced Bitcoin Price Prediction & Market Analysis System

Version 1.0 · Python 3.11/3.12 · Status: implemented and tested

This document describes the system architecture, data contracts, modeling
decisions and reliability guarantees of the implemented system. For user-facing
setup and usage instructions see `README.md`.

---

## 1. System overview

The system is a modular ML platform composed of five layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Data collection        src/btc_prediction/data/collector.py  │
│  Yahoo Finance → validate → clean → save raw + processed CSVs           │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2 — Feature engineering    src/btc_prediction/features/          │
│  OHLCV → 22 technical/lag features + 2 leak-free targets                │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 3 — Modeling               src/btc_prediction/models/ + training/│
│  3 regressors, 3 classifiers, 1 LSTM; chronological train/val/test      │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 4 — Evaluation             src/btc_prediction/evaluation/        │
│  Metrics, baselines, comparison tables, PNG figures, JSON summaries     │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 5 — Serving                src/btc_prediction/prediction/ + app/ │
│  BitcoinPredictor façade → Streamlit dashboard (7 pages)                │
└─────────────────────────────────────────────────────────────────────────┘
```

The CLI pipeline (`scripts/`) chains layers 1–4; the dashboard consumes the
persisted artifacts through layer 5 without ever retraining.

## 2. Data contract

| Item | Value |
| --- | --- |
| Source | Yahoo Finance, ticker `BTC-USD`, daily interval |
| History | 2014-09-17 → yesterday (UTC); configurable via `--start` |
| Raw file | `data/raw/btc_usd_raw.csv` (untouched download, for audit) |
| Processed file | `data/processed/btc_usd_daily.csv` (cleaned OHLCV) |
| Feature file | `data/processed/btc_usd_features.csv` (reference copy) |
| Columns | `Open, High, Low, Close, Volume` (float64) |
| Index | Sorted, unique, tz-naive daily dates |

Cleaning rules (in order): chronological sort → duplicate dates dropped
(keep last) → rows with an all-NaN OHLC block dropped → remaining isolated
NaNs forward-filled, still-missing rows dropped → **incomplete current-day
candle dropped** (a candle dated "today" UTC is still forming and must never
be used for training or as "latest close").

Failure handling: network/Yahoo failures raise `DataDownloadError` with an
actionable message; validation produces a quality report (row counts,
duplicates, missing values, high<low violations, non-positive prices) that
is logged and stored in `reports/training_log.txt`.

## 3. Feature engineering contract

All 22 features use only information available at the end of day *t*:

| Group | Features |
| --- | --- |
| Levels | `Close`, `Volume` |
| Trend | `SMA_7`, `SMA_14`, `SMA_30`, `EMA_12`, `EMA_26` |
| Momentum | `RSI_14` (Wilder smoothing), `MACD`, `MACD_Signal` (9-day EMA of MACD) |
| Returns/volatility | `Daily_Return_Pct`, `Rolling_Volatility_7`, `Rolling_Volatility_30` (std-dev of daily % returns) |
| Lags | `Close_Lag_1/2/3/5`, `Volume_Lag_1/2` |
| Range | `High_Low_Range`, `High_Low_Range_Pct`, `Open_Close_Pct` |

Targets: `Next_Day_Close = Close.shift(-1)`; `Direction = 1` when
`Next_Day_Close > Close` else `0`. The canonical feature order is the
`FEATURE_COLUMNS` list in `src/btc_prediction/features/build_features.py`
and is persisted to `models/artifacts/feature_columns.json` at training time
so inference and training can never drift apart.

**Leakage policy** (enforced by tests in `tests/test_leakage.py`):

1. No feature may reference future rows — perturbing all data after day *k*
   must leave features for days < *k* bit-identical.
2. Targets are excluded from `FEATURE_COLUMNS` by construction and by test.
3. Scalers (Pipeline `StandardScaler`, LSTM `MinMaxScaler`) are fitted on the
   training block only.
4. The final row (no next-day close) is dropped from the training dataset;
   the prediction layer recomputes that row's features separately.

## 4. Model architecture

### 4.1 Regression (next-day close)

| Model | Configuration | Artifact |
| --- | --- | --- |
| Linear Regression | `StandardScaler` + `LinearRegression` inside one Pipeline | `regression_linear_regression.joblib` |
| Random Forest | 300 trees, `min_samples_leaf=2`, seed 42 | `regression_random_forest.joblib` |
| XGBoost | 400 rounds, lr 0.05, depth 6, subsample/colsample 0.9, `hist` | `regression_xgboost.joblib` |
| LSTM (Keras) | see 4.3 | `lstm_model.keras` |

### 4.2 Classification (UP/DOWN)

| Model | Configuration | Artifact |
| --- | --- | --- |
| Logistic Regression | Scaled Pipeline, max_iter 2000 | `classification_logistic_regression.joblib` |
| Random Forest | 300 trees, `min_samples_leaf=2` | `classification_random_forest.joblib` |
| XGBoost | 300 rounds, lr 0.05, depth 4, `binary:logistic` | `classification_xgboost.joblib` |

### 4.3 LSTM (TensorFlow/Keras)

- Input: sliding sequences of `lookback=60` scaled feature days →
  `X[i] = features[i-60:i]`, `y[i] = scaled_close_target[i]`; sequences never
  contain their target row.
- Network: `LSTM(64, return_sequences=True)` → `Dropout(0.2)` →
  `LSTM(32)` → `Dropout(0.2)` → `Dense(16, relu)` → `Dense(1)`.
- Feature scaling: `MinMaxScaler` on features; separate `MinMaxScaler` on the
  next-day close target; both fitted on the training block only.
- Training: Adam(1e-3), MSE loss, batch 32, max 100 epochs,
  `EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)`,
  chronological validation set.
- Graceful degradation: if TensorFlow is missing or data is insufficient,
  `LstmUnavailableError` is raised and caught; every other component
  continues working and the dashboard hides the LSTM option.

## 5. Evaluation design

- **Splits**: contiguous 70% train / 15% validation / 15% test by date.
  Shuffled splits raise an error by construction.
- **Selection rule**: regression → lowest validation RMSE; classification →
  highest validation F1. The chosen name is stored in the metrics JSON.
- **Baselines**: persistence ("tomorrow = today") for regression;
  majority-class for classification. Both are reported in every table.
- **Metrics**: regression MAE / MSE / RMSE / R²; classification accuracy /
  precision / recall / F1 / ROC-AUC / confusion matrix (ROC-AUC reported as
  NaN when the split has a single class, never faked).
- **Outputs**: `reports/regression_comparison.csv`,
  `reports/classification_comparison.csv`,
  `reports/evaluation_summary.json`, and PNG figures in `reports/figures/`
  (actual-vs-predicted, residual analysis, confusion matrices, ROC curves,
  metric bar charts) — all rendered from the real test-set predictions saved
  during the training run.

## 6. Serving layer

`BitcoinPredictor` (src/btc_prediction/prediction/predictor.py):

- loads fitted artifacts lazily and caches them in-process;
- recomputes the training-time feature engineering on the latest processed
  data (single source of truth, no feature drift);
- `predict_next_day_close(model)` → predicted close, implied % change, the
  target date, and the last observed close/date;
- `predict_direction(model)` → UP/DOWN, `P(UP)`/`P(DOWN)`, confidence, and a
  plain-language explanation grounded in the current RSI / MACD / SMA state;
- raises `PredictionError` with recovery instructions when artifacts are
  missing instead of crashing.

## 7. Dashboard

`app/app.py` (entry point `streamlit run app/app.py`) with seven pages:
Market Overview · Historical Price Analysis · Technical Indicators ·
Bitcoin Price Prediction · Market Direction Prediction · Model Comparison ·
Model Evaluation. Sidebar provides navigation, a forced data-refresh button
(cache-aware) and project info. Expensive operations are cached
(`@st.cache_data` for data, `@st.cache_resource` for artifacts, cached
test-set predictions per model). Every page degrades gracefully when
artifacts or data are missing. Theme is configured in
`.streamlit/config.toml`.

## 8. Reliability and testing

- Deterministic seeds everywhere (`RANDOM_SEED = 42`, TF seed set per run).
- Structured logging through `config.settings.get_logger`.
- 73 pytest tests, all offline and deterministic:
  `tests/test_data_collection.py` (validation/cleaning/persistence),
  `tests/test_features.py` (indicator math),
  `tests/test_leakage.py` (future-perturbation invariance, target alignment),
  `tests/test_models.py` (training, metrics, split discipline),
  `tests/test_lstm.py` (sequence construction, training, persistence;
  skipped without TensorFlow),
  `tests/test_prediction.py` (artifact-based predictions),
  `tests/test_app.py` (app compilation, page registry, chart builders,
  no-hardcoded-paths rule).
- The dashboard never retrains; models load from `models/artifacts/`.

## 9. Deployment

- Entry point: `app/app.py`; `requirements.txt` pins compatible ranges.
- All paths derive from `config.settings.PROJECT_ROOT` (relative).
- No credentials, no machine-specific absolute paths (enforced by a test).
- Data refresh on the cloud: sidebar button triggers a fresh yfinance
  download; the committed CSV snapshot acts as an offline fallback.
- Repository size: source + a few hundred KB of CSVs + a few MB of model
  artifacts; venvs, caches and ZIPs are gitignored.

## 10. Repository map

See the *Architecture* section of `README.md` for the full tree, and
`config/settings.py` for every tunable constant (windows, lookback, split
fractions, artifact names).
