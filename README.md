# Gold Psychophysics — COT-Driven Regime Detection & SHAP Analysis

A full quantitative pipeline combining CFTC Commitments of Traders (COT) data with psychophysical market indicators to detect regimes and forecast gold futures positioning.

## What This Does

- Loads and processes CFTC COT disaggregated data for COMEX Gold Futures
- Downloads gold price, DXY, and US10Y macro data via yfinance
- Engineers **psychophysical features** based on the Weber-Fechner law (human perception of change relative to context)
- Builds a HAR-RV baseline + Random Forest model augmented with COT features
- Detects market regimes using Hidden Markov Models (HMM)
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
| `GoldPsychophysics.pdf` | Full auto-generated research report |

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/gold-psychophysics-cot.git
cd gold-psychophysics-cot
pip install -r requirements.txt  # if available, else: pip install numpy pandas matplotlib scikit-learn yfinance shap hmmlearn
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

## Research Context

This project is part of broader research on psychophysical models in financial markets, connected to:
> *Forecasting Volatility in Gold Futures Contracts: HAR Models, Options-Implied Volatility and the Limits of Directional Inference* — SSRN, June 2026
> [ssrn.com/abstract=6978741](https://ssrn.com/abstract=6978741)

---
*Built by Elena Hysa | Psychology × Quantitative Finance*
