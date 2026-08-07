"""Metrics and figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # No display in CI or on a headless machine.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config


def score(y_true, y_pred) -> dict:
    """Return MAE, RMSE and R² as a plain dict."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def results_table(scores: dict) -> str:
    """Format ``{model_name: metrics}`` as a markdown table."""
    frame = pd.DataFrame(scores).T
    frame.index.name = "Model"
    return frame.round(3).to_markdown()


def plot_predicted_vs_actual(y_true, y_pred, model_name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.2, edgecolors="none")

    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="crimson")

    ax.set_xlabel("Actual delivery time (min)")
    ax.set_ylabel("Predicted delivery time (min)")
    ax.set_title(f"{model_name}: predicted vs actual")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, model_name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residuals, bins=60, edgecolor="none")
    ax.axvline(0, linestyle="--", linewidth=1.5, color="crimson")
    ax.set_xlabel("Residual: actual - predicted (min)")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name}: residual distribution")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "residuals.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(
    model, X_test, y_test, model_name: str, top_n: int = 15
):
    """Plot the most influential features, returning them as a Series.

    Uses the model's built-in impurity importances when available, otherwise
    falls back to permutation importance (HistGradientBoosting exposes no
    ``feature_importances_``). Permutation importance also has the advantage of
    not inflating the score of high-cardinality numeric features.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = list(X_test.columns)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        label = "Impurity-based importance"
    else:
        result = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=-1
        )
        importances = pd.Series(result.importances_mean, index=feature_names)
        label = "Permutation importance (mean decrease in R²)"

    top = importances.sort_values(ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.values)
    ax.set_xlabel(label)
    ax.set_title(f"{model_name}: top {top_n} features")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    return top.iloc[::-1]
