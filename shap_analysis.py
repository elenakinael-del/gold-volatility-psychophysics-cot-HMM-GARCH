"""
shap_analysis.py
================
SHAP-based model explainability for the Gold Psychophysics framework.

Usage
-----
    from shap_analysis import run_shap_analysis, narrative_explanation
    shap_vals, explainer = run_shap_analysis(rf_model, X_train, X_test, feature_names)
    print(narrative_explanation(shap_vals, X_test, feature_names, idx=-1))
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[shap_analysis] WARNING: 'shap' not installed. "
          "Run:  pip install shap\n"
          "Falling back to permutation importance.")

from sklearn.inspection import permutation_importance


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_shap_analysis(
    model,
    X_train,
    X_test,
    feature_names: Optional[List[str]] = None,
    max_background: int = 200,
) -> Tuple[np.ndarray, object]:
    """
    Compute SHAP values for a fitted sklearn-compatible model.

    Returns
    -------
    shap_values : np.ndarray  shape (n_test, n_features)
    explainer   : shap.TreeExplainer or None
    """
    if feature_names is None:
        if hasattr(X_train, "columns"):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    X_train_arr = _to_array(X_train)
    X_test_arr  = _to_array(X_test)

    if SHAP_AVAILABLE:
        print(f"[shap_analysis] Computing SHAP values for {X_test_arr.shape[0]} samples ...")
        n_bg  = min(max_background, X_train_arr.shape[0])
        bg    = X_train_arr[np.random.choice(X_train_arr.shape[0], n_bg, replace=False)]

        explainer   = shap.TreeExplainer(model, data=bg, feature_names=feature_names)
        shap_values = explainer.shap_values(X_test_arr)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        print(f"[shap_analysis] SHAP values computed. Shape: {shap_values.shape}")
        return shap_values, explainer

    else:
        print("[shap_analysis] Using permutation importance fallback ...")
        perm = permutation_importance(
            model, X_test_arr, model.predict(X_test_arr),
            n_repeats=10, random_state=42,
        )
        fake_shap = np.tile(perm.importances_mean, (X_test_arr.shape[0], 1))
        return fake_shap, None


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_shap_summary(
    shap_values: np.ndarray,
    X,
    feature_names: List[str],
    max_display: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Beeswarm summary plot (bar fallback if shap not installed)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    if SHAP_AVAILABLE:
        plt.sca(ax)
        shap.summary_plot(
            shap_values,
            features=_to_array(X),
            feature_names=feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        ax.set_title("SHAP Feature Impact -- Gold Volatility Model",
                     color="white", fontsize=13, pad=12)
    else:
        _bar_fallback(ax, shap_values, feature_names, max_display)

    _dark_theme_ax(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[shap_analysis] SHAP summary saved -> {save_path}")

    return fig


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X,
    feature_names: List[str],
    idx: int = -1,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Waterfall chart explaining a single prediction."""
    row_shap = shap_values[idx]
    row_X    = _to_array(X)[idx]

    order    = np.argsort(np.abs(row_shap))[::-1][:15]
    names    = [feature_names[i] for i in order]
    vals     = row_shap[order]
    feat_val = row_X[order]

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    ax.barh(range(len(vals)), vals, color=colors, edgecolor="none", height=0.65)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(
        [f"{n} = {v:.4f}" for n, v in zip(names, feat_val)],
        color="white", fontsize=9,
    )
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value  (red = increases vol forecast, green = decreases)",
                  color="white", fontsize=9)
    ax.set_title(f"SHAP Waterfall -- Prediction Explanation (idx={idx})",
                 color="white", fontsize=12)

    _dark_theme_ax(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[shap_analysis] Waterfall saved -> {save_path}")

    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    X,
    feature: str,
    feature_names: List[str],
    interaction_feature: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """SHAP dependence plot for a single feature."""
    X_arr = _to_array(X)
    if feature not in feature_names:
        raise ValueError(f"'{feature}' not in feature_names.")

    fidx   = feature_names.index(feature)
    x_vals = X_arr[:, fidx]
    s_vals = shap_values[:, fidx]

    if interaction_feature and interaction_feature in feature_names:
        iidx       = feature_names.index(interaction_feature)
        i_vals     = X_arr[:, iidx]
        scatter_kw = dict(c=i_vals, cmap="RdYlGn", alpha=0.7, s=18)
        clabel     = interaction_feature
    else:
        scatter_kw = dict(c="#3498db", alpha=0.6, s=18)
        clabel     = None

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    sc = ax.scatter(x_vals, s_vals, **scatter_kw)
    if clabel:
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(clabel, color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feature, color="white")
    ax.set_ylabel(f"SHAP value for {feature}", color="white")
    ax.set_title(f"SHAP Dependence -- {feature}", color="white", fontsize=12)

    _dark_theme_ax(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[shap_analysis] Dependence plot saved -> {save_path}")

    return fig


def plot_feature_importance_comparison(
    shap_values: np.ndarray,
    rf_importances: Dict[str, float],
    feature_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side: SHAP mean|abs| vs RF feature importance."""
    shap_imp = np.abs(shap_values).mean(axis=0)
    shap_imp = shap_imp / shap_imp.sum()

    rf_imp = np.array([rf_importances.get(f, 0) for f in feature_names])
    rf_imp = rf_imp / rf_imp.sum() if rf_imp.sum() > 0 else rf_imp

    order = np.argsort(shap_imp)[::-1][:12]
    names = [feature_names[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f0f1a")

    for ax, vals, title, color in [
        (ax1, shap_imp[order], "SHAP Mean |Value|",     "#e74c3c"),
        (ax2, rf_imp[order],   "RF Feature Importance", "#3498db"),
    ]:
        ax.set_facecolor("#0f0f1a")
        ax.barh(range(len(vals)), vals[::-1], color=color, edgecolor="none")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names[::-1], color="white", fontsize=9)
        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333355")

    fig.suptitle("Feature Importance: SHAP vs RF", color="white", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[shap_analysis] Importance comparison saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Narrative explanation
# ---------------------------------------------------------------------------
def narrative_explanation(
    shap_values: np.ndarray,
    X,
    feature_names: List[str],
    idx: int = -1,
    top_n: int = 6,
) -> str:
    """
    Returns a human-readable string explaining the model's prediction.

    Example
    -------
    Today's prediction driven by:

      + VolShock            (+0.0031)   Increases forecast vol
      - DealerPressure      (-0.0012)   Decreases forecast vol
    """
    row   = shap_values[idx]
    order = np.argsort(np.abs(row))[::-1][:top_n]

    lines = ["", "Today's prediction driven by:", ""]
    for i in order:
        sign  = "+" if row[i] > 0 else "-"
        arrow = "Increases forecast vol" if row[i] > 0 else "Decreases forecast vol"
        name  = feature_names[i]
        lines.append(f"  {sign} {name:<30s} ({row[i]:+.4f})   {arrow}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _to_array(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)


def _dark_theme_ax(ax: plt.Axes) -> None:
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333355")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def _bar_fallback(ax, shap_values, feature_names, max_display):
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:max_display]
    vals     = mean_abs[order]
    names    = [feature_names[i] for i in order]
    ax.barh(range(len(vals)), vals[::-1], color="#e74c3c", edgecolor="none")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names[::-1], color="white", fontsize=9)
    ax.set_title("Feature Importance (SHAP fallback -- mean |value|)",
                 color="white", fontsize=12)


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(0)
    n, p = 500, 8
    fnames = ["RV20", "DXY", "US10Y", "VolShock", "PositionShock",
              "SpecPressure", "PsychophysicalPositioning", "CrowdingIndex"]
    X = pd.DataFrame(np.random.randn(n, p), columns=fnames)
    y = X["RV20"] * 2 + X["VolShock"] * 0.5 + np.random.randn(n) * 0.3

    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X[:400], y[:400])

    sv, ex = run_shap_analysis(rf, X[:400], X[400:], fnames)
    print(narrative_explanation(sv, X[400:], fnames, idx=-1))
    plot_shap_summary(sv, X[400:], fnames, save_path="shap_summary_test.png")
    print("Done.")
