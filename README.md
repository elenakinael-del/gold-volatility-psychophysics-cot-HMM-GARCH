# Gold Psychophysics — Reproducible Gold-Volatility Research

An exploratory, reproducible research pipeline testing whether CFTC positioning and psychophysically scaled features improve **four-week-ahead realised-volatility** forecasts for COMEX gold futures. This is not an investment strategy or a claim of tradable alpha.

## Research question

Does the augmented feature set improve out-of-sample volatility forecasts over a HAR-RV and macro baseline? The pipeline reports an untouched final time holdout and expanding-window fold metrics. Results are evidence for this sample only—not causal proof or a guarantee of persistence.

## What This Does

- Loads and processes CFTC COT disaggregated data for COMEX Gold Futures
- Downloads gold price, DXY, and US10Y macro data via yfinance
- Engineers **psychophysical features** based on the Weber-Fechner law (human perception of change relative to context)
- Builds a HAR-RV baseline + Random Forest model augmented with COT features
- Produces HMM regimes as an **in-sample descriptive diagnostic**; fitted regime labels are excluded from forecasting features to avoid future-data leakage
- Runs full SHAP explainability analysis on feature importance
- Generates a PDF research report with all charts

## Pipeline

```
COT Data → Feature Engineering → HAR-RV + Random Forest → HMM Regimes → SHAP → PDF Report
```

## Outputs

| File | Description |
|---|---|
| `outputs/regimes.png` | HMM regime chart overlaid on gold price |
| `outputs/forecast.png` | Model forecast vs actual |
| `outputs/shap_summary.png` | SHAP beeswarm — feature importance |
| `outputs/shap_waterfall.png` | Single prediction explanation |
| `outputs/feature_importance.png` | RF vs baseline feature comparison |
| `outputs/positioning_history.png` | COT positioning over time |
| `outputs/walk_forward_metrics.csv` | Expanding-window MAE, RMSE, and R² for five validation folds |
| `GoldPsychophysics.pdf` | Full auto-generated research report |

## Setup

```bash
git clone https://github.com/elenakinael-del/gold-volatility-psychophysics-cot-HMM-GARCH.git
cd gold-volatility-psychophysics-cot-HMM-GARCH
pip install numpy pandas matplotlib scikit-learn yfinance shap hmmlearn
```

## Run

```bash
python main.py --cot "path/to/C_Disagg.txt"

# If you already have the processed CSV:
python main.py --cot gold_cot_processed.csv --skip-cot-rebuild
```

## Key Concepts

**Weber-Fechner Law in Markets**: Human perception of price change is logarithmic, not linear. This project encodes that into features — the *perceived* magnitude of a COT positioning shift is scaled by the existing baseline, not the raw absolute change.

**HMM Regime Detection**: Uses a 3-state Hidden Markov Model to identify Bull, Bear, and Transition regimes from volatility + positioning features.

**SHAP Explainability**: Every model prediction is explained at the feature level, making the "black box" interpretable for research purposes.

## Data Sources

- CFTC COT Disaggregated Reports: [cftc.gov](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)
- Gold Futures (GC=F), DXY, US10Y: Yahoo Finance via yfinance

## Reproducibility and limitations

- The model uses chronological splits only; it never shuffles observations.
- COT reports are released after the Tuesday reporting date. Any live use must align features to their public release timestamps; this repository is research code, not a live trading system.
- `GC=F` and Yahoo Finance are convenient research series, not exchange-quality execution data. No trading costs, slippage, or deployable trading backtest is claimed.
- The data snapshot and external downloads can change. Save a dated raw COT extract and the generated `outputs/walk_forward_metrics.csv` with any reported result.

## Research Context

This project is part of broader research on psychophysical models in financial markets, connected to:
> *Forecasting Volatility in Gold Futures Contracts: HAR Models, Options-Implied Volatility and the Limits of Directional Inference* — SSRN, June 2026
> [ssrn.com/abstract=6978741](https://ssrn.com/abstract=6978741)

---
*Built by Elena Hysa | Psychology × Quantitative Finance*
