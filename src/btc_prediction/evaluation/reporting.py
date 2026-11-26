"""Generate comparison tables and evaluation charts from real metrics.

Everything here reads the metric JSON files written during training and
produces human-readable reports (CSV/JSON tables + PNG figures) under
``reports/``. No numbers are invented; if a report is missing it means the
models were not trained yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from config.settings import (
    ARTIFACTS_DIR,
    CLASSIFICATION_METRICS_PATH,
    FIGURES_DIR,
    LSTM_METRICS_PATH,
    REGRESSION_METRICS_PATH,
    REPORTS_DIR,
    ensure_directories,
    get_logger,
)

logger = get_logger(__name__)


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        logger.warning("Metrics file missing: %s (train the models first).", path)
        return None
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------

def build_regression_comparison(
    metrics: Optional[dict] = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Optional[pd.DataFrame]:
    """Test-set comparison table for regression models + baseline + LSTM."""
    metrics = metrics or _load_json(Path(artifacts_dir) / "regression_metrics.json")
    if metrics is None:
        return None

    rows = []
    for name, entry in metrics.items():
        if name.startswith("_"):
            continue
        test = entry.get("test", {})
        rows.append(
            {
                "Model": name,
                "MAE": test.get("MAE"),
                "MSE": test.get("MSE"),
                "RMSE": test.get("RMSE"),
                "R2": test.get("R2"),
            }
        )
    lstm_payload = _load_json(LSTM_METRICS_PATH)
    if lstm_payload:
        test = lstm_payload.get("test", {})
        rows.append(
            {
                "Model": "LSTM (Deep Learning)",
                "MAE": test.get("MAE"),
                "MSE": test.get("MSE"),
                "RMSE": test.get("RMSE"),
                "R2": test.get("R2"),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("MAE").reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df


def build_classification_comparison(
    metrics: Optional[dict] = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Optional[pd.DataFrame]:
    """Test-set comparison table for classification models + baseline."""
    metrics = metrics or _load_json(Path(artifacts_dir) / "classification_metrics.json")
    if metrics is None:
        return None

    rows = []
    for name, entry in metrics.items():
        if name.startswith("_"):
            continue
        test = entry.get("test", {})
        rows.append(
            {
                "Model": name,
                "Accuracy": test.get("Accuracy"),
                "Precision": test.get("Precision"),
                "Recall": test.get("Recall"),
                "F1": test.get("F1"),
                "ROC_AUC": test.get("ROC_AUC"),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("F1", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df


def save_comparison_tables(out_dir: Path = REPORTS_DIR):
    """Write comparison tables as CSV into reports/ (returns paths)."""
    ensure_directories()
    paths = {}
    reg = build_regression_comparison()
    if reg is not None:
        path = Path(out_dir) / "regression_comparison.csv"
        reg.to_csv(path, index=False)
        paths["regression"] = path
    clf = build_classification_comparison()
    if clf is not None:
        path = Path(out_dir) / "classification_comparison.csv"
        clf.to_csv(path, index=False)
        paths["classification"] = path
    logger.info("Saved comparison tables: %s", list(paths.values()))
    return paths


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true, y_pred, title: str, filename: str, out_dir: Path = FIGURES_DIR
) -> Path:
    """Actual vs predicted line chart for a regression model on the test set."""
    ensure_directories()
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.plot(y_true, label="Actual", color="#1f77b4", linewidth=1.4)
    ax.plot(y_pred, label="Predicted", color="#F7931A", linewidth=1.2, alpha=0.9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Test-set trading day (chronological)")
    ax.set_ylabel("Next-day close (USD)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.12), ncol=2, frameon=False)
    ax.grid(alpha=0.25)
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_residuals(y_true, y_pred, title: str, filename: str, out_dir: Path = FIGURES_DIR) -> Path:
    """Error analysis: residual distribution + residual vs actual scatter."""
    ensure_directories()
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].hist(residuals, bins=40, color="#F7931A", edgecolor="white")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Residual distribution", fontsize=11)
    axes[0].set_xlabel("Prediction error (USD)")
    axes[0].set_ylabel("Days")
    axes[1].scatter(y_true, residuals, s=8, alpha=0.5, color="#1f77b4")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Residuals vs actual close", fontsize=11)
    axes[1].set_xlabel("Actual next-day close (USD)")
    axes[1].set_ylabel("Prediction error (USD)")
    fig.suptitle(title, fontsize=12)
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, title: str, filename: str, out_dir: Path = FIGURES_DIR) -> Path:
    """Annotated confusion matrix heat map (0=DOWN, 1=UP)."""
    ensure_directories()
    fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    cm = np.asarray(cm)
    im = ax.imshow(cm, cmap="Oranges")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1], labels=["DOWN (0)", "UP (1)"])
    ax.set_yticks([0, 1], labels=["DOWN (0)", "UP (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc_curve(y_true, y_proba, title: str, filename: str, out_dir: Path = FIGURES_DIR) -> Path:
    """ROC curve for a classifier with its AUC in the legend."""
    from sklearn.metrics import roc_curve, auc

    ensure_directories()
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        logger.warning("ROC curve skipped: test set has a single class.")
        return Path(out_dir) / filename
    fpr, tpr, _ = roc_curve(y_true, np.asarray(y_proba))
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    ax.plot(fpr, tpr, color="#F7931A", linewidth=2,
            label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1,
            label="Random guessing")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_metric_bars(comparison: pd.DataFrame, metric: str, title: str,
                     filename: str, out_dir: Path = FIGURES_DIR) -> Path:
    """Horizontal bar chart comparing one metric across models."""
    ensure_directories()
    data = comparison.dropna(subset=[metric]).sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(data) + 1.6), constrained_layout=True)
    colors = ["#F7931A" if "Baseline" not in m and "LSTM" not in m
              else "#7f7f7f" if "Baseline" in m else "#1f77b4"
              for m in data["Model"]]
    ax.barh(data["Model"], data[metric], color=colors)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(metric)
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(data[metric]):
        ax.text(v, i, f" {v:.4g}", va="center", fontsize=9)
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def generate_all_reports(
    artifacts_dir: Path = ARTIFACTS_DIR,
    out_dir: Path = REPORTS_DIR,
    dataset: Optional[pd.DataFrame] = None,
    predictions: Optional[dict] = None,
) -> dict:
    """Produce every report/figure that has the data to exist.

    ``predictions`` optionally maps ``model_name -> {"y_true", "y_pred"}``
    (regression) and ``{"y_true", "y_proba"}`` (classification) so figures
    can be rendered directly from the training run.
    """
    ensure_directories()
    predictions = predictions or {}
    produced = {}

    reg = build_regression_comparison(artifacts_dir=artifacts_dir)
    clf = build_classification_comparison(artifacts_dir=artifacts_dir)

    tables = save_comparison_tables(out_dir)
    produced.update(tables)

    reg_metrics = _load_json(Path(artifacts_dir) / "regression_metrics.json")
    if reg_metrics and predictions:
        for name, pred in predictions.get("regression", {}).items():
            slug = name.lower().replace(" ", "_")
            produced[f"pred_vs_actual_{slug}"] = plot_actual_vs_predicted(
                pred["y_true"], pred["y_pred"],
                f"{name}: actual vs predicted next-day close (test set)",
                f"regression_{slug}_actual_vs_predicted.png", out_dir / "figures",
            )
            produced[f"residuals_{slug}"] = plot_residuals(
                pred["y_true"], pred["y_pred"],
                f"{name}: error analysis (test set)",
                f"regression_{slug}_error_analysis.png", out_dir / "figures",
            )

    clf_metrics = _load_json(Path(artifacts_dir) / "classification_metrics.json")
    if clf_metrics and predictions:
        for name, pred in predictions.get("classification", {}).items():
            slug = name.lower().replace(" ", "_")
            produced[f"cm_{slug}"] = plot_confusion_matrix(
                pred.get("confusion_matrix", [[0, 0], [0, 0]]),
                f"{name}: confusion matrix (test set)",
                f"classification_{slug}_confusion_matrix.png", out_dir / "figures",
            )
            if pred.get("y_proba") is not None:
                produced[f"roc_{slug}"] = plot_roc_curve(
                    pred["y_true"], pred["y_proba"],
                    f"{name}: ROC curve (test set)",
                    f"classification_{slug}_roc_curve.png", out_dir / "figures",
                )

    if reg is not None:
        produced["bar_regression_mae"] = plot_metric_bars(
            reg, "MAE", "Regression models: test-set MAE (lower is better)",
            "regression_mae_comparison.png", out_dir / "figures",
        )
        produced["bar_regression_r2"] = plot_metric_bars(
            reg, "R2", "Regression models: test-set R2 (higher is better)",
            "regression_r2_comparison.png", out_dir / "figures",
        )
    if clf is not None:
        produced["bar_classification_f1"] = plot_metric_bars(
            clf, "F1", "Classification models: test-set F1 (higher is better)",
            "classification_f1_comparison.png", out_dir / "figures",
        )

    # One consolidated evaluation summary for humans.
    summary = {
        "regression_comparison": reg.to_dict(orient="records") if reg is not None else None,
        "classification_comparison": clf.to_dict(orient="records") if clf is not None else None,
    }
    with open(Path(out_dir) / "evaluation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    produced["evaluation_summary"] = Path(out_dir) / "evaluation_summary.json"

    logger.info("Generated %d reports/figures.", len(produced))
    return produced
