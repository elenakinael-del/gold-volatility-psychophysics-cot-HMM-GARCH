"""
cot_loader.py
=============
Loads and processes CFTC Disaggregated COT data for Gold (COMEX).
Filters for "GOLD - COMMODITY EXCHANGE INC." rows, extracts all
key positioning variables, and computes psychophysical features.

Usage
-----
    from cot_loader import load_gold_cot
    cot = load_gold_cot("C_Disagg06_25_2.txt")
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


# ---------------------------------------------------------------------------
# Column mapping from the CFTC disaggregated file header
# ---------------------------------------------------------------------------
COT_COLS = {
    "Market_and_Exchange_Names":          "Market",
    "Report_Date_as_YYYY-MM-DD":          "Date",
    "Open_Interest_All":                  "OpenInterest",
    "Prod_Merc_Positions_Long_All":       "ProdLong",
    "Prod_Merc_Positions_Short_All":      "ProdShort",
    "Swap_Positions_Long_All":            "SwapLong",
    "Swap__Positions_Short_All":          "SwapShort",
    "M_Money_Positions_Long_All":         "MMLong",
    "M_Money_Positions_Short_All":        "MMShort",
    "Other_Rept_Positions_Long_All":      "OtherLong",
    "Other_Rept_Positions_Short_All":     "OtherShort",
    "Change_in_Open_Interest_All":        "ChangeOI",
    "Change_in_M_Money_Long_All":         "ChangeMMLong",
    "Change_in_M_Money_Short_All":        "ChangeMMShort",
}

GOLD_MARKET_NAME = "GOLD - COMMODITY EXCHANGE INC."

# Rolling windows (weeks)
WINDOW_SHORT  = 26   # ~6 months
WINDOW_LONG   = 52   # ~1 year
WINDOW_SHOCK  = 52   # z-score normalisation window
EWM_SPAN      = 26   # EWM for position momentum


def load_gold_cot(filepath: str) -> pd.DataFrame:
    """
    Load CFTC disaggregated file and return a clean weekly Gold COT DataFrame
    with all psychophysical features attached.

    Parameters
    ----------
    filepath : str or Path
        Path to the CFTC .txt (CSV) disaggregated file
        (e.g. C_Disagg06_25_2.txt).

    Returns
    -------
    pd.DataFrame  (index = Date, weekly)
    """
    filepath = Path(filepath)
    print(f"[cot_loader] Reading {filepath.name}  ...")

    # ------------------------------------------------------------------
    # 1. Read raw CSV
    # ------------------------------------------------------------------
    raw = pd.read_csv(
        filepath,
        low_memory=False,
        encoding="latin-1",
    )

    # ------------------------------------------------------------------
    # 2. Filter Gold rows only
    # ------------------------------------------------------------------
    gold = raw[raw["Market_and_Exchange_Names"].str.strip() == GOLD_MARKET_NAME].copy()
    print(f"[cot_loader] Gold rows found: {len(gold):,}")

    if gold.empty:
        raise ValueError(
            f"No rows matching '{GOLD_MARKET_NAME}' found in {filepath.name}.\n"
            "Check the market name or file path."
        )

    # ------------------------------------------------------------------
    # 3. Select and rename columns
    # ------------------------------------------------------------------
    gold = gold.rename(columns=COT_COLS)
    keep = list(COT_COLS.values())
    gold = gold[keep].copy()

    # ------------------------------------------------------------------
    # 4. Parse date and sort chronologically
    # ------------------------------------------------------------------
    gold["Date"] = pd.to_datetime(gold["Date"])
    gold = gold.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    gold = gold.set_index("Date")

    # Numeric coercion (some fields may be strings with spaces)
    for col in gold.columns:
        gold[col] = pd.to_numeric(gold[col], errors="coerce")

    print(f"[cot_loader] Date range: {gold.index.min().date()} -> {gold.index.max().date()}")

    # ------------------------------------------------------------------
    # 5. Primary positioning variables
    # ------------------------------------------------------------------
    gold["MMNet"]        = gold["MMLong"] - gold["MMShort"]
    gold["SwapNet"]      = gold["SwapLong"] - gold["SwapShort"]
    gold["ProducerNet"]  = gold["ProdLong"] - gold["ProdShort"]
    gold["OtherNet"]     = gold["OtherLong"] - gold["OtherShort"]

    # ------------------------------------------------------------------
    # 6. Normalised pressures  (% of Open Interest)
    # ------------------------------------------------------------------
    oi = gold["OpenInterest"].replace(0, np.nan)
    gold["SpecPressure"]   = gold["MMNet"]      / oi
    gold["DealerPressure"] = gold["SwapNet"]     / oi
    gold["ProdPressure"]   = gold["ProducerNet"] / oi

    # ------------------------------------------------------------------
    # 7. Psychophysical Positioning  (Weber-Fechner: log ratio to ref)
    #    PP_t = log( |MMNet_t| / MMNet_rolling_mean )
    # ------------------------------------------------------------------
    mm_ref = gold["MMNet"].abs().rolling(WINDOW_LONG, min_periods=13).mean()
    gold["PsychophysicalPositioning"] = np.log(
        gold["MMNet"].abs().clip(lower=1) / mm_ref.clip(lower=1)
    )

    # Signed version (preserves direction)
    gold["PP_signed"] = np.sign(gold["MMNet"]) * gold["PsychophysicalPositioning"]

    # ------------------------------------------------------------------
    # 8. Position Shock  (z-score of weekly change)
    # ------------------------------------------------------------------
    mm_chg = gold["MMNet"].diff()
    mm_std = mm_chg.rolling(WINDOW_SHOCK, min_periods=13).std()
    gold["PositionShock"] = mm_chg / mm_std.clip(lower=1e-6)

    # ------------------------------------------------------------------
    # 9. Position momentum  (EWM trend)
    # ------------------------------------------------------------------
    gold["MMNet_EWM"]   = gold["MMNet"].ewm(span=EWM_SPAN, adjust=False).mean()
    gold["PosMomentum"] = gold["MMNet"] - gold["MMNet_EWM"]

    # ------------------------------------------------------------------
    # 10. Crowding index  (how extreme is current positioning vs history)
    # ------------------------------------------------------------------
    roll_min  = gold["MMNet"].rolling(WINDOW_LONG, min_periods=26).min()
    roll_max  = gold["MMNet"].rolling(WINDOW_LONG, min_periods=26).max()
    roll_rng  = (roll_max - roll_min).clip(lower=1e-6)
    gold["CrowdingIndex"] = (gold["MMNet"] - roll_min) / roll_rng

    # ------------------------------------------------------------------
    # 11. Dealer-vs-Speculator divergence
    # ------------------------------------------------------------------
    gold["DivergenceIndex"] = gold["SpecPressure"] - gold["DealerPressure"]

    # ------------------------------------------------------------------
    # 12. Drop early NaN rows (before rolling windows fill)
    # ------------------------------------------------------------------
    gold = gold.dropna(subset=["PositionShock", "PsychophysicalPositioning"])

    print(f"[cot_loader] Clean rows after feature engineering: {len(gold):,}")
    print(f"[cot_loader] Features: {list(gold.columns)}")
    return gold


# ---------------------------------------------------------------------------
# Quick diagnostics helper
# ---------------------------------------------------------------------------
def cot_summary(cot: pd.DataFrame) -> None:
    """Print a brief summary of the loaded COT frame."""
    print("\n" + "=" * 60)
    print("GOLD COT SUMMARY")
    print("=" * 60)
    print(f"Observations : {len(cot):,}")
    print(f"Date range   : {cot.index.min().date()} -> {cot.index.max().date()}")
    print("\nDescriptive stats (key features):")
    cols = [
        "MMNet", "SpecPressure", "DealerPressure",
        "PsychophysicalPositioning", "PositionShock", "CrowdingIndex",
    ]
    print(cot[[c for c in cols if c in cot.columns]].describe().round(4).to_string())
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "C_Disagg06_25_2.txt"
    cot  = load_gold_cot(path)
    cot_summary(cot)
    cot.to_csv("gold_cot_processed.csv")
    print("[cot_loader] Saved -> gold_cot_processed.csv")
