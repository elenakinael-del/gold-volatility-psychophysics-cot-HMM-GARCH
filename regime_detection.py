"""
regime_detection.py
===================
Hidden Markov Model regime detection for gold volatility.

States
------
    0  = Calm         (low volatility, stable positioning)
    1  = Transitional (rising volatility, positioning shifts)
    2  = Crisis       (high volatility, extreme positioning)

Usage
-----
    from regime_detection import fit_regimes, add_regime_features
    result = fit_regimes(df, n_states=3)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

# Optional: hmmlearn for proper HMM; fallback to Gaussian Mixture if absent
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Colour palette for regimes
# ---------------------------------------------------------------------------
REGIME_COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]   # green / amber / red
REGIME_LABELS = ["Calm", "Transitional", "Crisis"]
REGIME_ALPHAS = [0.25, 0.30, 0.35]


# ---------------------------------------------------------------------------
# Core fitting function
# ---------------------------------------------------------------------------
def fit_regimes(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit an HMM (or GMM fallback) on the supplied feature columns and
    attach regime assignments + probabilities to the DataFrame.

    Parameters
    ----------
    df          : DataFrame with a DatetimeIndex
    features    : columns to use for regime detection
    n_states    : number of hidden states (default 3)
    n_iter      : EM iterations
    random_state: reproducibility seed

    Returns
    -------
    df  with new columns:
        Regime              int
        RegimeLabel         str
        RegimeProb_0/1/2    float
    """
    if features is None:
        candidates = ["RV20", "VolSurprise", "PositionShock",
                      "SpecPressure", "PsychophysicalPositioning"]
        features = [c for c in candidates if c in df.columns]

    if not features:
        raise ValueError("No valid feature columns found for regime detection.")

    print(f"[regime_detection] Fitting {n_states}-state model on: {features}")

    X = df[features].copy().ffill().bfill()
    X = (X - X.mean()) / X.std().clip(lower=1e-9)
    X_arr = X.values

    if HMM_AVAILABLE:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        model.fit(X_arr)
        raw_states = model.predict(X_arr)
        probs      = model.predict_proba(X_arr)
    else:
        print("[regime_detection] hmmlearn not found -- using GaussianMixture fallback.")
        model = GaussianMixture(
            n_components=n_states,
            covariance_type="full",
            n_init=5,
            random_state=random_state,
        )
        model.fit(X_arr)
        raw_states = model.predict(X_arr)
        probs      = model.predict_proba(X_arr)

    # Re-order states by ascending mean volatility
    vol_col = "RV20" if "RV20" in features else features[0]
    vol_idx = features.index(vol_col)
    means   = [X_arr[raw_states == s, vol_idx].mean() for s in range(n_states)]
    order   = np.argsort(means)
    remap   = {old: new for new, old in enumerate(order)}

    states          = np.array([remap[s] for s in raw_states])
    probs_reordered = probs[:, order]

    df = df.copy()
    df["Regime"]      = states
    df["RegimeLabel"] = [REGIME_LABELS[min(s, len(REGIME_LABELS) - 1)] for s in states]

    for i in range(n_states):
        df[f"RegimeProb_{i}"] = probs_reordered[:, i]

    _print_regime_stats(df, n_states)
    return df


# ---------------------------------------------------------------------------
# Add derivative regime features for ML models
# ---------------------------------------------------------------------------
def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
        RegimeChange    bool  -- regime changed this period
        RegimeDuration  int   -- consecutive periods in current regime
        CrisisProb      float -- alias for RegimeProb_2
    """
    df = df.copy()

    if "Regime" not in df.columns:
        print("[regime_detection] Warning: 'Regime' column not found. Run fit_regimes first.")
        return df

    df["RegimeChange"]   = df["Regime"].diff().ne(0).astype(int)
    df["RegimeDuration"] = (
        df.groupby((df["Regime"] != df["Regime"].shift()).cumsum()).cumcount() + 1
    )

    if "RegimeProb_2" in df.columns:
        df["CrisisProb"] = df["RegimeProb_2"]

    return df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_regimes(
    df: pd.DataFrame,
    price_col: str = "GoldPrice",
    vol_col: str = "RV20",
    figsize: Tuple[int, int] = (16, 9),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    3-panel chart:
        Panel 1 -- Gold price shaded by regime
        Panel 2 -- Realised volatility shaded by regime
        Panel 3 -- Regime probability stacked area
    """
    if "Regime" not in df.columns:
        raise ValueError("Run fit_regimes() first.")

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333355")

    regimes = df["Regime"].values
    dates   = df.index

    def shade_regimes(ax):
        start = 0
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i - 1] or i == len(regimes) - 1:
                end = i
                r = int(regimes[start])
                ax.axvspan(
                    dates[start], dates[min(end, len(dates) - 1)],
                    alpha=REGIME_ALPHAS[min(r, 2)],
                    color=REGIME_COLORS[min(r, 2)],
                    linewidth=0,
                )
                start = i

    # Panel 1: Price
    ax0 = axes[0]
    if price_col in df.columns:
        ax0.plot(dates, df[price_col], color="#f4d03f", linewidth=1.2, label="Gold Price")
        ax0.set_ylabel("Price (USD)", color="white")
    shade_regimes(ax0)
    ax0.set_title("Gold -- Volatility Regime Detection (HMM)", color="white", fontsize=14)

    # Panel 2: Volatility
    ax1 = axes[1]
    if vol_col in df.columns:
        ax1.plot(dates, df[vol_col] * 100, color="#e74c3c", linewidth=1.0, label="RV20 (%)")
        ax1.set_ylabel("RV20 (%)", color="white")
    shade_regimes(ax1)
    ax1.legend(facecolor="#0f0f1a", labelcolor="white")

    # Panel 3: Probabilities
    ax2 = axes[2]
    prob_cols = [c for c in [f"RegimeProb_{i}" for i in range(3)] if c in df.columns]
    prob_data = [df[c].values for c in prob_cols]
    colors_   = REGIME_COLORS[:len(prob_cols)]
    labels_   = REGIME_LABELS[:len(prob_cols)]
    if prob_data:
        ax2.stackplot(dates, *prob_data, labels=labels_, colors=colors_, alpha=0.85)
    ax2.set_ylabel("Regime Probability", color="white")
    ax2.set_ylim(0, 1)
    ax2.legend(facecolor="#0f0f1a", labelcolor="white", loc="upper left")

    patches = [
        mpatches.Patch(color=REGIME_COLORS[i], label=REGIME_LABELS[i])
        for i in range(len(REGIME_COLORS))
    ]
    axes[0].legend(handles=patches, facecolor="#0f0f1a", labelcolor="white", loc="upper left")

    for ax in axes:
        ax.yaxis.label.set_color("white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[regime_detection] Chart saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _print_regime_stats(df: pd.DataFrame, n_states: int) -> None:
    print("\n[regime_detection] Regime Statistics")
    print("-" * 40)
    for s in range(n_states):
        mask  = df["Regime"] == s
        count = mask.sum()
        pct   = 100 * count / len(df)
        label = REGIME_LABELS[min(s, len(REGIME_LABELS) - 1)]
        print(f"  State {s} ({label:14s}): {count:5d} weeks  ({pct:.1f}%)")
    print("-" * 40 + "\n")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n = 800
    dates = pd.date_range("2006-01-01", periods=n, freq="W")
    df_test = pd.DataFrame({
        "RV20":          np.abs(np.random.randn(n) * 0.01 + 0.015),
        "SpecPressure":  np.random.randn(n) * 0.1,
        "PositionShock": np.random.randn(n),
    }, index=dates)

    result = fit_regimes(df_test)
    result = add_regime_features(result)
    print(result[["Regime", "RegimeLabel", "RegimeDuration"]].tail(10))
