"""
main.py
=======
Gold Psychophysics v3 — Full Pipeline
======================================
Runs end-to-end:
    1. Load & process CFTC COT data
    2. Download gold price + macro data (yfinance)
    3. Merge COT + price into weekly master frame
    4. Feature engineering (psychophysical features)
    5. HAR-RV baseline model
    6. Random Forest model (baseline + augmented)
    7. HMM Regime Detection
    8. SHAP explainability
    9. Charts
   10. PDF report

Usage
-----
    python3 main.py --cot "/path/to/C_Disagg06_25 2.txt"

    # If you already have gold_cot_processed.csv:
    python3 main.py --cot gold_cot_processed.csv --skip-cot-rebuild
"""

from __future__ import annotations

import argparse
import warnings
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all systems
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional imports with graceful fallbacks
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("[main] yfinance not found. Run:  pip install yfinance")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[main] scikit-learn not found. Run:  pip install scikit-learn")

# Local modules
from cot_loader import load_gold_cot, cot_summary
from regime_detection import fit_regimes, add_regime_features, plot_regimes
from shap_analysis import run_shap_analysis, narrative_explanation
from shap_analysis import plot_shap_summary, plot_shap_waterfall, plot_feature_importance_comparison
from pdf_export import generate_report


# ===========================================================================
# 1. PRICE & MACRO DATA
# ===========================================================================
def download_price_data(start: str = "2006-01-01", end: str = "2026-06-30") -> pd.DataFrame:
    """Download weekly Gold, DXY, US10Y via yfinance."""
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance required. Run:  pip install yfinance")

    print("[main] Downloading price data ...")
    tickers = {
        "GoldPrice": "GC=F",
        "DXY":       "DX-Y.NYB",
        "US10Y":     "^TNX",
    }
    frames = {}
    for name, ticker in tickers.items():
        try:
            raw = yf.download(ticker, start=start, end=end,
                              interval="1d", progress=False, auto_adjust=True)
            if raw.empty:
                print(f"  [warn] No data for {ticker}")
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = name
            frames[name] = close
        except Exception as e:
            print(f"  [warn] {ticker} failed: {e}")

    if not frames:
        raise RuntimeError("Could not download any price data.")

    daily = pd.concat(frames.values(), axis=1)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Realised Volatility (20-day, annualised)
    gold_ret    = np.log(daily["GoldPrice"] / daily["GoldPrice"].shift(1))
    daily["RV5"]  = gold_ret.rolling(5).std()  * np.sqrt(252)
    daily["RV20"] = gold_ret.rolling(20).std() * np.sqrt(252)
    daily["RV60"] = gold_ret.rolling(60).std() * np.sqrt(252)

    # Resample to weekly (Friday close)
    weekly = daily.resample("W-FRI").last()

    # Volatility surprise: log(RV20 / EWM_63(RV20))
    weekly["VolSurprise"] = np.log(
        weekly["RV20"] / weekly["RV20"].ewm(span=63, adjust=False).mean().clip(lower=1e-6)
    )
    weekly["VolShock"] = weekly["VolSurprise"]   # alias used in SHAP narrative

    print(f"[main] Price data: {weekly.index.min().date()} -> {weekly.index.max().date()}  "
          f"({len(weekly)} weeks)")
    return weekly


# ===========================================================================
# 2. MERGE
# ===========================================================================
def merge_data(price: pd.DataFrame, cot: pd.DataFrame) -> pd.DataFrame:
    """Merge weekly price + COT on nearest date (tolerance = 5 days)."""
    print("[main] Merging price + COT data ...")

    # COT reports are typically Tuesday; price is Friday — merge_asof handles this
    price_sorted = price.sort_index()
    cot_sorted   = cot.sort_index()

    merged = pd.merge_asof(
        price_sorted,
        cot_sorted,
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta("5D"),
        direction="backward",
    )

    # Forward-fill COT columns (COT is weekly, price daily gaps already resampled)
    cot_cols = cot_sorted.columns.tolist()
    merged[cot_cols] = merged[cot_cols].ffill()

    merged = merged.dropna(subset=["RV20", "MMNet"])
    print(f"[main] Merged frame: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


# ===========================================================================
# 3. TARGET VARIABLE
# ===========================================================================
def build_target(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """
    Target: forward-looking RV20 over `horizon` weeks.
    horizon=4 means we're forecasting 1-month-ahead volatility.
    """
    df = df.copy()
    df["FutureRV"] = df["RV20"].shift(-horizon)
    df = df.dropna(subset=["FutureRV"])
    print(f"[main] Target = RV20 shifted {horizon} weeks forward. "
          f"Rows after target alignment: {len(df)}")
    return df


# ===========================================================================
# 4. FEATURE SET
# ===========================================================================
FEATURES_BASELINE = ["RV5", "RV20", "RV60"]

FEATURES_MACRO = ["DXY", "US10Y"]

FEATURES_PSYCH = [
    "PerceivedVol",
    "VolSurprise",
    "VolShock",
    "PsychophysicalPositioning",
    "PP_signed",
    "PositionShock",
    "PosMomentum",
]

FEATURES_COT = [
    "SpecPressure",
    "DealerPressure",
    "ProdPressure",
    "CrowdingIndex",
    "DivergenceIndex",
]

FEATURES_REGIME = ["Regime", "CrisisProb", "RegimeDuration"]

# PerceivedVol needs to be computed from merged data
def add_perceived_vol(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rv_ref = df["RV20"].rolling(52, min_periods=13).mean()
    df["PerceivedVol"] = np.log(df["RV20"].clip(lower=1e-6) / rv_ref.clip(lower=1e-6))
    return df


def get_feature_list(df: pd.DataFrame, include_regime: bool = True) -> List[str]:
    """Return all available features that exist in the dataframe."""
    all_features = (
        FEATURES_BASELINE +
        FEATURES_MACRO +
        FEATURES_PSYCH +
        FEATURES_COT +
        (FEATURES_REGIME if include_regime else [])
    )
    available = [f for f in all_features if f in df.columns]
    missing   = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"[main] Features not available (skipped): {missing}")
    return available


# ===========================================================================
# 5. TRAIN / TEST SPLIT
# ===========================================================================
def train_test_split_ts(
    df: pd.DataFrame,
    features: List[str],
    target: str = "FutureRV",
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Temporal split — no shuffling."""
    n_test  = int(len(df) * test_ratio)
    n_train = len(df) - n_test

    X = df[features].copy()
    y = df[target].copy()

    X_train = X.iloc[:n_train].values
    X_test  = X.iloc[n_train:].values
    y_train = y.iloc[:n_train].values
    y_test  = y.iloc[n_train:].values

    idx_train = df.index[:n_train]
    idx_test  = df.index[n_train:]

    print(f"[main] Train: {idx_train[0].date()} -> {idx_train[-1].date()}  ({n_train} weeks)")
    print(f"[main] Test:  {idx_test[0].date()}  -> {idx_test[-1].date()}   ({n_test} weeks)")
    return X_train, X_test, y_train, y_test, idx_train, idx_test


# ===========================================================================
# 6. HAR-RV BASELINE
# ===========================================================================
def fit_har_rv(df: pd.DataFrame, target: str = "FutureRV", test_ratio: float = 0.2) -> float:
    """
    HAR-RV: RV_{t+h} = a + b*RV_t + c*RV_weekly + d*RV_monthly + e
    Returns out-of-sample R2.
    """
    if not SKLEARN_AVAILABLE:
        return float("nan")

    df = df.copy().dropna(subset=["RV5", "RV20", "RV60", target])
    X  = df[["RV5", "RV20", "RV60"]].values
    y  = df[target].values

    n_test  = int(len(df) * test_ratio)
    n_train = len(df) - n_test

    model = LinearRegression()
    model.fit(X[:n_train], y[:n_train])
    preds = model.predict(X[n_train:])
    r2    = r2_score(y[n_train:], preds)
    print(f"[main] HAR-RV R2 (out-of-sample): {r2:.4f}")
    return r2


# ===========================================================================
# 7. RANDOM FOREST
# ===========================================================================
def fit_random_forest(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    feature_names: List[str],
    label: str = "RF",
) -> Tuple[object, float, Dict[str, float]]:
    """Fit Random Forest, return (model, r2, feature_importance_dict)."""
    if not SKLEARN_AVAILABLE:
        return None, float("nan"), {}

    print(f"[main] Fitting {label} ({X_train.shape[1]} features, {X_train.shape[0]} train rows) ...")
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        max_features=0.6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2    = r2_score(y_test, preds)
    fi    = dict(zip(feature_names, model.feature_importances_))

    print(f"[main] {label} R2 (out-of-sample): {r2:.4f}")
    print(f"[main] Top features:")
    for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"         {feat:<35s}  {imp * 100:.1f}%")

    return model, r2, fi


# ===========================================================================
# 8. VISUALISATIONS
# ===========================================================================
def plot_results_dashboard(
    df: pd.DataFrame,
    idx_test: pd.DatetimeIndex,
    y_test: np.ndarray,
    preds: np.ndarray,
    fi: Dict[str, float],
    shap_values: Optional[np.ndarray],
    feature_names: List[str],
    save_dir: Path,
) -> Dict[str, plt.Figure]:
    """Generate all charts and save to save_dir."""
    figs = {}

    # ---- 1. Actual vs Predicted ----------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.plot(idx_test, y_test * 100,  color="#e74c3c", linewidth=1.2, label="Actual RV")
    ax.plot(idx_test, preds * 100,   color="#f4d03f", linewidth=1.0,
            linestyle="--", label="RF Forecast", alpha=0.85)
    ax.set_title("Gold Volatility — Actual vs Forecast", color="white", fontsize=13)
    ax.set_ylabel("Annualised RV (%)", color="white")
    ax.legend(facecolor="#0f0f1a", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333355")
    plt.tight_layout()
    path = save_dir / "forecast.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    figs["forecast"] = fig
    print(f"[main] Saved -> {path}")

    # ---- 2. Feature Importance bar chart --------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor("#0f0f1a")
    ax2.set_facecolor("#0f0f1a")
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:12]
    names_ = [x[0] for x in fi_sorted][::-1]
    vals_  = [x[1] * 100 for x in fi_sorted][::-1]
    bars   = ax2.barh(names_, vals_, color="#D4AF37", edgecolor="none")
    ax2.set_xlabel("Importance (%)", color="white")
    ax2.set_title("Random Forest — Feature Importance", color="white", fontsize=12)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333355")
    plt.tight_layout()
    path2 = save_dir / "feature_importance.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    figs["feature_importance"] = fig2
    print(f"[main] Saved -> {path2}")

    # ---- 3. SHAP summary (if available) ---------------------------------
    if shap_values is not None:
        X_test_df = df.loc[idx_test, feature_names]
        fig3 = plot_shap_summary(
            shap_values, X_test_df, feature_names,
            save_path=str(save_dir / "shap_summary.png"),
        )
        figs["shap_summary"] = fig3

        fig4 = plot_shap_waterfall(
            shap_values, X_test_df, feature_names, idx=-1,
            save_path=str(save_dir / "shap_waterfall.png"),
        )
        figs["shap_waterfall"] = fig4

    return figs


def plot_positioning_history(df: pd.DataFrame, save_dir: Path) -> plt.Figure:
    """3-panel: Gold price / MMNet positioning / Speculative pressure."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.patch.set_facecolor("#0f0f1a")

    panels = [
        ("GoldPrice",   "Gold Price (USD)",          "#f4d03f"),
        ("MMNet",       "Managed Money Net (lots)",   "#3498db"),
        ("SpecPressure","Speculative Pressure (% OI)","#e74c3c"),
    ]

    for ax, (col, ylabel, color) in zip(axes, panels):
        ax.set_facecolor("#0f0f1a")
        if col in df.columns:
            ax.plot(df.index, df[col], color=color, linewidth=1.0)
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333355")

    axes[0].set_title("Gold — Price, Positioning & Speculative Pressure (2006–2026)",
                      color="white", fontsize=13)
    plt.tight_layout()
    path = save_dir / "positioning_history.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[main] Saved -> {path}")
    return fig


# ===========================================================================
# 9. MAIN PIPELINE
# ===========================================================================
def run_pipeline(cot_path: str, horizon: int = 4, skip_cot_rebuild: bool = False) -> None:

    save_dir = Path("outputs")
    save_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 65)
    print("  GOLD PSYCHOPHYSICS v3 — FULL PIPELINE")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # Step 1: Load COT
    # ------------------------------------------------------------------
    processed_csv = Path("gold_cot_processed.csv")
    if skip_cot_rebuild and processed_csv.exists():
        print("[main] Loading cached COT from gold_cot_processed.csv ...")
        cot = pd.read_csv(processed_csv, index_col="Date", parse_dates=True)
    else:
        cot = load_gold_cot(cot_path)
        cot.to_csv(processed_csv)
        print(f"[main] COT saved -> {processed_csv}")

    cot_summary(cot)

    # ------------------------------------------------------------------
    # Step 2: Download price data
    # ------------------------------------------------------------------
    start_date = str(cot.index.min().date())
    price = download_price_data(start=start_date)

    # ------------------------------------------------------------------
    # Step 3: Merge
    # ------------------------------------------------------------------
    df = merge_data(price, cot)
    df = add_perceived_vol(df)

    # ------------------------------------------------------------------
    # Step 4: Regime detection (before feature selection so regime
    #         columns are available as features)
    # ------------------------------------------------------------------
    df = fit_regimes(df, n_states=3)
    df = add_regime_features(df)

    # Plot positioning history
    plot_positioning_history(df, save_dir)

    # ------------------------------------------------------------------
    # Step 5: Target variable
    # ------------------------------------------------------------------
    df = build_target(df, horizon=horizon)

    # ------------------------------------------------------------------
    # Step 6: HAR-RV baseline
    # ------------------------------------------------------------------
    print("\n--- HAR-RV Baseline ---")
    r2_har = fit_har_rv(df, target="FutureRV")

    # ------------------------------------------------------------------
    # Step 7a: RF — macro only (no psychophysics, no COT)
    # ------------------------------------------------------------------
    print("\n--- RF Baseline (no COT/psychophysics) ---")
    features_base = [f for f in FEATURES_BASELINE + FEATURES_MACRO if f in df.columns]
    df_clean      = df.dropna(subset=features_base + ["FutureRV"])
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split_ts(
        df_clean, features_base
    )
    rf_base, r2_base, fi_base = fit_random_forest(
        X_tr, X_te, y_tr, y_te, features_base, label="RF-Baseline"
    )

    # ------------------------------------------------------------------
    # Step 7b: RF — full augmented (all features)
    # ------------------------------------------------------------------
    print("\n--- RF Augmented (full psychophysical + COT + regime) ---")
    features_full = get_feature_list(df, include_regime=True)
    df_full       = df.dropna(subset=features_full + ["FutureRV"])
    X_tr2, X_te2, y_tr2, y_te2, idx_tr2, idx_te2 = train_test_split_ts(
        df_full, features_full
    )
    rf_aug, r2_aug, fi_aug = fit_random_forest(
        X_tr2, X_te2, y_tr2, y_te2, features_full, label="RF-Augmented"
    )

    preds_aug = rf_aug.predict(X_te2)

    # ------------------------------------------------------------------
    # Step 8: SHAP
    # ------------------------------------------------------------------
    print("\n--- SHAP Analysis ---")
    shap_values, _ = run_shap_analysis(rf_aug, X_tr2, X_te2, features_full)
    narrative = narrative_explanation(shap_values, X_te2, features_full, idx=-1)
    print(narrative)

    # ------------------------------------------------------------------
    # Step 9: Charts
    # ------------------------------------------------------------------
    print("\n--- Generating Charts ---")
    figs = plot_results_dashboard(
        df_full, idx_te2, y_te2, preds_aug, fi_aug,
        shap_values, features_full, save_dir,
    )

    regime_fig = plot_regimes(
        df, price_col="GoldPrice", vol_col="RV20",
        save_path=str(save_dir / "regimes.png"),
    )
    figs["regime_chart"] = regime_fig

    # ------------------------------------------------------------------
    # Step 10: Regime stats for report
    # ------------------------------------------------------------------
    regime_stats = {}
    for label in ["Calm", "Transitional", "Crisis"]:
        mask = df["RegimeLabel"] == label
        regime_stats[label] = {
            "count":  int(mask.sum()),
            "pct":    100 * mask.sum() / len(df),
            "avg_rv": float(df.loc[mask, "RV20"].mean()) if mask.sum() > 0 else 0.0,
        }

    # Hypothesis verdicts (simple heuristics)
    hyp = {
        "H1: Volatility Perception": {
            "supported": "PerceivedVol" in fi_aug and fi_aug.get("PerceivedVol", 0) > 0.01,
            "note":      f"PerceivedVol importance: {fi_aug.get('PerceivedVol', 0)*100:.1f}%",
        },
        "H2: Positioning Perception": {
            "supported": fi_aug.get("PsychophysicalPositioning", 0) > 0.01,
            "note":      f"PP importance: {fi_aug.get('PsychophysicalPositioning', 0)*100:.1f}%",
        },
        "H3: Position Shock -> Vol": {
            "supported": fi_aug.get("PositionShock", 0) > 0.01,
            "note":      f"PositionShock importance: {fi_aug.get('PositionShock', 0)*100:.1f}%",
        },
        "H4: Psych > HAR-RV": {
            "supported": r2_aug > r2_har,
            "note":      f"R2 delta: {r2_aug - r2_har:+.4f}",
        },
    }

    # ------------------------------------------------------------------
    # Step 11: PDF Report
    # ------------------------------------------------------------------
    print("\n--- Generating PDF Report ---")
    results = {
        "r2_har":             r2_har,
        "r2_rf":              r2_base,
        "r2_augmented":       r2_aug,
        "feature_importance": fi_aug,
        "shap_narrative":     narrative,
        "regime_stats":       regime_stats,
        "hypothesis_results": hyp,
        "data_summary": {
            "n_obs":     len(df_full),
            "cot_start": str(cot.index.min().date()),
            "cot_end":   str(cot.index.max().date()),
        },
    }

    chart_map = {
        "regime_chart":       figs.get("regime_chart"),
        "shap_summary":       figs.get("shap_summary"),
        "shap_waterfall":     figs.get("shap_waterfall"),
        "feature_importance": figs.get("feature_importance"),
    }

    pdf_path = generate_report(results, charts=chart_map,
                               output_path="GoldPsychophysics.pdf")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  HAR-RV R2          : {r2_har:.4f}")
    print(f"  RF Baseline R2     : {r2_base:.4f}")
    print(f"  RF Augmented R2    : {r2_aug:.4f}   <-- full psychophysical model")
    print(f"  Improvement vs HAR : {r2_aug - r2_har:+.4f}")
    print(f"\n  Regime breakdown:")
    for label, v in regime_stats.items():
        print(f"    {label:<14s}: {v['count']:4d} weeks ({v['pct']:.1f}%),  "
              f"avg RV = {v['avg_rv']*100:.2f}%")
    print(f"\n  Output files in:  ./{save_dir}/")
    print(f"  PDF report:       {pdf_path}")
    print("=" * 65 + "\n")


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Psychophysics v3 Pipeline")
    parser.add_argument(
        "--cot",
        type=str,
        default="gold_cot_processed.csv",
        help="Path to CFTC COT file or pre-processed gold_cot_processed.csv",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="Forecast horizon in weeks (default: 4 = 1 month ahead)",
    )
    parser.add_argument(
        "--skip-cot-rebuild",
        action="store_true",
        help="Skip re-parsing the raw COT file if gold_cot_processed.csv exists",
    )
    args = parser.parse_args()

    run_pipeline(
        cot_path=args.cot,
        horizon=args.horizon,
        skip_cot_rebuild=args.skip_cot_rebuild,
    )
