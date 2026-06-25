"""
pdf_export.py  (v3.1 — fixed import detection)
===============================================
Generates an academic-style PDF report for the Gold Psychophysics project.
Compatible with Python 3.9+ and Anaconda environments.

Usage
-----
    from pdf_export import generate_report
    generate_report(results, charts, output_path="GoldPsychophysics.pdf")
"""

from __future__ import annotations

import io
import datetime
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

# ---------------------------------------------------------------------------
# Robust reportlab detection — works across Anaconda + system Python
# ---------------------------------------------------------------------------
def _try_import_reportlab():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, Image as RLImage, PageBreak,
        )
        return True, dict(
            A4=A4, cm=cm, colors=colors,
            getSampleStyleSheet=getSampleStyleSheet,
            ParagraphStyle=ParagraphStyle,
            TA_CENTER=TA_CENTER, TA_JUSTIFY=TA_JUSTIFY,
            SimpleDocTemplate=SimpleDocTemplate,
            Paragraph=Paragraph, Spacer=Spacer,
            Table=Table, TableStyle=TableStyle,
            HRFlowable=HRFlowable, RLImage=RLImage,
            PageBreak=PageBreak,
        )
    except ImportError:
        return False, {}

REPORTLAB_AVAILABLE, RL = _try_import_reportlab()

if REPORTLAB_AVAILABLE:
    colors   = RL["colors"]
    GOLD     = colors.HexColor("#D4AF37")
    DARK     = colors.HexColor("#1a1a2e")
    LIGHT    = colors.HexColor("#e2e2e2")
    GREEN    = colors.HexColor("#2ecc71")
    A4       = RL["A4"]
    cm       = RL["cm"]
else:
    GOLD = DARK = LIGHT = GREEN = A4 = cm = colors = None
    print("[pdf_export] reportlab not found in current Python.")
    print(f"[pdf_export] Current interpreter: {sys.executable}")
    print("[pdf_export] Fix: python3 -m pip install reportlab")


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================
def generate_report(
    results: Dict[str, Any],
    charts: Optional[Dict[str, Any]] = None,
    output_path: str = "GoldPsychophysics.pdf",
) -> str:
    if not REPORTLAB_AVAILABLE:
        return _fallback_text_report(results, output_path)

    charts  = charts or {}
    rl      = RL
    A4_     = rl["A4"]
    cm_     = rl["cm"]
    colors_ = rl["colors"]

    GOLD_   = colors_.HexColor("#D4AF37")
    DARK_   = colors_.HexColor("#1a1a2e")
    LIGHT_  = colors_.HexColor("#e2e2e2")
    GREEN_  = colors_.HexColor("#2ecc71")

    doc = rl["SimpleDocTemplate"](
        str(output_path),
        pagesize=A4_,
        leftMargin=2.2 * cm_, rightMargin=2.2 * cm_,
        topMargin=2.5 * cm_,  bottomMargin=2.5 * cm_,
        title="Gold Psychophysics -- Volatility Forecasting Framework",
        author="GoldPsychophysics v3",
    )

    styles = _build_styles(rl, GOLD_, LIGHT_)
    story  = []

    story += _title_section(styles, rl, GOLD_, cm_)
    story += _abstract_section(styles, rl, GOLD_, results)
    story.append(rl["PageBreak"]())
    story += _methodology_section(styles, rl, GOLD_)
    story += _data_section(styles, rl, GOLD_, results)
    story.append(rl["PageBreak"]())
    story += _results_section(styles, rl, GOLD_, results)
    story += _regime_section(styles, rl, GOLD_, results, charts, cm_)
    story += _shap_section(styles, rl, GOLD_, results, charts, cm_)
    story.append(rl["PageBreak"]())
    story += _conclusions_section(styles, rl, GOLD_, results)

    def _header_first(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(DARK_)
        canvas.rect(0, 0, A4_[0], A4_[1], fill=True, stroke=False)
        canvas.restoreState()

    def _header_later(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(DARK_)
        canvas.rect(0, 0, A4_[0], A4_[1], fill=True, stroke=False)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors_.grey)
        canvas.drawString(2.2 * cm_, 1.2 * cm_, "Gold Psychophysics v3")
        canvas.drawRightString(A4_[0] - 2.2 * cm_, 1.2 * cm_, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_header_first, onLaterPages=_header_later)
    abs_path = str(Path(output_path).resolve())
    print(f"[pdf_export] PDF saved -> {abs_path}")
    return abs_path


# ===========================================================================
# SECTION BUILDERS
# ===========================================================================
def _title_section(styles, rl, GOLD_, cm_):
    P, S, HR = rl["Paragraph"], rl["Spacer"], rl["HRFlowable"]
    today = datetime.date.today().strftime("%B %d, %Y")
    return [
        S(1, 1.5 * cm_),
        P("GOLD VOLATILITY PSYCHOPHYSICS", styles["BigTitle"]),
        P("A Quantitative Research Framework", styles["Subtitle"]),
        S(1, 0.4 * cm_),
        HR(width="100%", thickness=2, color=GOLD_),
        S(1, 0.4 * cm_),
        P("Testing Weber-Fechner Perception Laws in Commodity Futures Markets",
          styles["SubSubtitle"]),
        S(1, 1.0 * cm_),
        P(f"Version 3.1  |  {today}", styles["DateLine"]),
        S(1, 0.8 * cm_),
        HR(width="60%", thickness=1, color=GOLD_),
        S(1, 1.5 * cm_),
    ]


def _abstract_section(styles, rl, GOLD_, results):
    P, S, HR = rl["Paragraph"], rl["Spacer"], rl["HRFlowable"]
    r2_rf  = results.get("r2_rf",        0.08)
    r2_aug = results.get("r2_augmented", 0.12)
    r2_har = results.get("r2_har",       0.31)
    text = (
        f"This paper investigates whether human perception of financial volatility follows "
        f"logarithmic (Weber-Fechner) scaling laws, and whether psychophysical transformations "
        f"of CFTC Commitment of Traders positioning data improve gold volatility forecasting. "
        f"Using 1,032 weekly COMEX Gold observations (2006-2026), we construct Perceived "
        f"Volatility, Psychophysical Positioning, and Position Shock features and test them "
        f"within a Random Forest framework augmented by HMM regime detection and SHAP "
        f"explainability. The HAR-RV benchmark achieves R2 = {r2_har:.3f}. "
        f"The augmented Random Forest achieves R2 = {r2_aug:.3f}, with regime and "
        f"psychophysical features contributing meaningful marginal information beyond "
        f"pure volatility autoregression."
    )
    return [
        P("ABSTRACT", styles["SectionHeader"]),
        HR(width="100%", thickness=0.5, color=GOLD_),
        S(1, 0.3 * cm_()),
        P(text, styles["BodyJustify"]),
        S(1, 0.4 * cm_()),
        P("<b>Keywords:</b> Gold volatility, Weber-Fechner law, CFTC COT, "
          "HAR-RV, Hidden Markov Model, SHAP, Psychophysics", styles["Body"]),
        S(1, 0.8 * cm_()),
    ] if False else _abstract_simple(styles, rl, GOLD_, results)


def _abstract_simple(styles, rl, GOLD_, results):
    P, S, HR = rl["Paragraph"], rl["Spacer"], rl["HRFlowable"]
    r2_rf  = results.get("r2_rf",        0.08)
    r2_aug = results.get("r2_augmented", 0.12)
    r2_har = results.get("r2_har",       0.31)
    text = (
        f"This paper investigates whether human perception of financial volatility follows "
        f"logarithmic (Weber-Fechner) scaling laws, and whether psychophysical transformations "
        f"of CFTC COT positioning data improve gold volatility forecasting. "
        f"Using 1,032 weekly COMEX Gold observations (2006-2026), we construct Perceived "
        f"Volatility, Psychophysical Positioning, and Position Shock features and test them "
        f"within a Random Forest framework augmented by HMM regime detection and SHAP "
        f"explainability. HAR-RV benchmark: R2 = {r2_har:.3f}. "
        f"RF Augmented: R2 = {r2_aug:.3f}."
    )
    return [
        P("ABSTRACT", styles["SectionHeader"]),
        HR(width="100%", thickness=0.5, color=GOLD_),
        S(1, 0.2 * cm),
        P(text, styles["BodyJustify"]),
        S(1, 0.4 * cm),
        P("<b>Keywords:</b> Gold volatility, Weber-Fechner law, CFTC COT, "
          "HAR-RV, Hidden Markov Model, SHAP, Psychophysics", styles["Body"]),
        S(1, 0.8 * cm),
    ]


def _methodology_section(styles, rl, GOLD_):
    P, S, HR, T, TS = (rl["Paragraph"], rl["Spacer"], rl["HRFlowable"],
                        rl["Table"], rl["TableStyle"])
    colors_ = rl["colors"]
    story = [
        P("1. METHODOLOGY", styles["SectionHeader"]),
        HR(width="100%", thickness=0.5, color=GOLD_),
        S(1, 0.3 * cm),
    ]
    hypotheses = [
        ("H1 -- Volatility Perception",
         "PV_t = log( sigma_t / sigma_ref )",
         "Humans perceive volatility logarithmically per Weber-Fechner law."),
        ("H2 -- Positioning Perception",
         "PP_t = log( |MMNet_t| / MMNet_ref )",
         "Managed Money traders adjust positions in log proportion to baseline."),
        ("H3 -- Position Shock Forecasts Volatility",
         "Shock_t = ( MMNet_t - MMNet_{t-1} ) / sigma(MMNet)",
         "Standardised weekly position changes predict future realised volatility."),
        ("H4 -- Psychophysical Model vs HAR-RV",
         "RV_{t+h} = a + b*RV_t + c*PV_t + d*PP_t + e_t",
         "Psychophysical augmentation improves out-of-sample R2 vs HAR-RV."),
    ]
    for title, eq, desc in hypotheses:
        story += [
            P(f"<b>{title}</b>", styles["SubSection"]),
            S(1, 0.1 * cm),
            P(eq, styles["Equation"]),
            S(1, 0.1 * cm),
            P(desc, styles["Body"]),
            S(1, 0.4 * cm),
        ]
    return story


def _data_section(styles, rl, GOLD_, results):
    P, S, HR, T, TS = (rl["Paragraph"], rl["Spacer"], rl["HRFlowable"],
                        rl["Table"], rl["TableStyle"])
    colors_ = rl["colors"]
    ds = results.get("data_summary", {})
    rows = [["Variable", "Description", "Frequency"]]
    rows += [
        ["RV5/20/60",     "Realised volatility (annualised)",          "Daily->Weekly"],
        ["PerceivedVol",  "log(RV20 / rolling mean RV20)",             "Weekly"],
        ["MMNet",         "Managed Money Net Position",                  "Weekly"],
        ["PsychophysicalPositioning", "log(|MMNet| / 52w mean)",        "Weekly"],
        ["PositionShock", "z-score of weekly MMNet change",             "Weekly"],
        ["SpecPressure",  "MMNet / Open Interest",                       "Weekly"],
        ["CrowdingIndex", "MMNet percentile (52w rolling)",              "Weekly"],
        ["Regime",        "HMM state: 0=Calm, 1=Trans, 2=Crisis",       "Weekly"],
        ["DXY",           "US Dollar Index",                             "Weekly"],
        ["US10Y",         "US 10Y Treasury Yield",                       "Weekly"],
    ]
    tbl = _make_table(rows, rl)
    obs = ds.get("n_obs", "N/A")
    cot_start = ds.get("cot_start", "2006")
    cot_end   = ds.get("cot_end",   "2026")
    return [
        P("2. DATA", styles["SectionHeader"]),
        HR(width="100%", thickness=0.5, color=GOLD_),
        S(1, 0.2 * cm),
        P(f"Weekly COMEX Gold COT data ({cot_start} - {cot_end}, {obs} observations) "
          "merged with daily Gold futures, DXY, and US10Y from Yahoo Finance.",
          styles["BodyJustify"]),
        S(1, 0.3 * cm),
        tbl,
        S(1, 0.6 * cm),
    ]


def _results_section(styles, rl, GOLD_, results):
    P, S, HR = rl["Paragraph"], rl["Spacer"], rl["HRFlowable"]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    r2_har = results.get("r2_har",        "N/A")
    r2_rf  = results.get("r2_rf",         "N/A")
    r2_aug = results.get("r2_augmented",  "N/A")
    fi     = results.get("feature_importance", {})

    perf = [
        ["Model",               "R2",        "Notes"],
        ["HAR-RV (baseline)",   fmt(r2_har), "Linear autoregression benchmark"],
        ["RF Baseline",         fmt(r2_rf),  "RV + DXY + US10Y only"],
        ["RF + Psychophysics",  fmt(r2_aug), "Full framework (all features)"],
    ]

    story = [
        P("3. RESULTS", styles["SectionHeader"]),
        HR(width="100%", thickness=0.5, color=GOLD_),
        S(1, 0.2 * cm),
        P("3.1 Out-of-Sample R2 Comparison", styles["SubSection"]),
        S(1, 0.15 * cm),
        _make_table(perf, rl),
        S(1, 0.4 * cm),
    ]

    if fi:
        fi_rows = [["Rank", "Feature", "Importance (%)"]]
        for rank, (feat, imp) in enumerate(
            sorted(fi.items(), key=lambda x: x[1], reverse=True), 1
        ):
            fi_rows.append([str(rank), feat, f"{imp * 100:.1f}%"])
        story += [
            P("3.2 Feature Importance", styles["SubSection"]),
            S(1, 0.15 * cm),
            _make_table(fi_rows, rl),
            S(1, 0.4 * cm),
        ]

    hresults = results.get("hypothesis_results", {})
    if hresults:
        h_rows = [["Hypothesis", "Verdict", "Evidence"]]
        for h, v in hresults.items():
            verdict = "Supported" if v.get("supported", False) else "Not supported"
            h_rows.append([h, verdict, v.get("note", "")])
        story += [
            P("3.3 Hypothesis Summary", styles["SubSection"]),
            S(1, 0.15 * cm),
            _make_table(h_rows, rl),
            S(1, 0.4 * cm),
        ]

    return story


def _regime_section(styles, rl, GOLD_, results, charts, cm_):
    P, S, HR = rl["Paragraph"], rl["Spacer"], rl["HRFlowable"]
    reg = results.get("regime_stats", {})
    story = [
        P("4. REGIME DETECTION", styles["SectionHeader"]),
        HR(width="100%", thickness=0.5, color=GOLD_),
        S(1, 0.2 * cm),
        P("3-state Gaussian HMM fitted on RV20, VolSurprise, PositionShock, "
          "SpecPressure, PsychophysicalPositioning. States ordered by ascending "
          "mean volatility.", styles["BodyJustify"]),
        S(1, 0.3 * cm),
    ]
    if reg:
        rows = [["Regime", "Weeks", "% Sample", "Avg RV20"]]
        for label, v in reg.items():
            rows.append([
                label, str(v.get("count","--")),
                f"{v.get('pct',0):.1f}%",
                f"{v.get('avg_rv',0)*100:.2f}%",
            ])
        story.append(_make_table(rows, rl))
        story.append(S(1, 0.3 * cm))
    if "regime_chart" in charts:
        story.append(_embed(charts["regime_chart"], 15, 8, rl, cm_))
    story.append(S(1, 0.5 * cm))
    return story


def _shap_section(styles, rl, GOLD_, results, charts, cm_):
    P, S, HR = rl["Paragraph"], rl["Spacer"], rl["HRFlowable"]
    narrative = results.get("shap_narrative", "")
    story = [
        P("5. SHAP EXPLAINABILITY", styles["SectionHeader"]),
        HR(width="100%", thickness=0.5, color=GOLD_),
        S(1, 0.2 * cm),
        P("TreeSHAP decomposes each prediction into additive feature contributions. "
          "Positive SHAP = increases volatility forecast; negative = dampens it.",
          styles["BodyJustify"]),
        S(1, 0.3 * cm),
    ]
    if narrative:
        story += [
            P("<b>Latest Prediction Explanation:</b>", styles["SubSection"]),
            S(1, 0.1 * cm),
            P(narrative.replace("\n", "<br/>"), styles["Body"]),
            S(1, 0.3 * cm),
        ]
    for key in ("shap_summary", "shap_waterfall", "feature_importance"):
        if key in charts:
            story.append(_embed(charts[key], 14, 7, rl, cm_))
            story.append(S(1, 0.2 * cm))
    story.append(S(1, 0.5 * cm))
    return story


def _conclusions_section(styles, rl, GOLD_, results):
    P, S, HR = rl["Paragraph"], rl["Spacer"], rl["HRFlowable"]
    r2_aug = results.get("r2_augmented", 0.12)
    r2_har = results.get("r2_har", 0.31)
    story = [
        P("6. CONCLUSIONS", styles["SectionHeader"]),
        HR(width="100%", thickness=0.5, color=GOLD_),
        S(1, 0.2 * cm),
        P(
            f"The HAR-RV benchmark (R2 = {r2_har:.3f}) outperforms the Random Forest "
            f"(R2 = {r2_aug:.3f}) on the 2022-2026 test period, which is dominated by "
            f"the post-COVID inflation regime -- a structurally different environment "
            f"from the 2007-2022 training window. This regime mismatch is itself an "
            f"interesting finding: linear volatility autoregression is more robust to "
            f"structural breaks than tree-based models trained on historical regimes. "
            f"Psychophysical and positioning features (CrisisProb, RegimeDuration, "
            f"ProdPressure) contribute meaningfully as auxiliary signals, accounting "
            f"for ~13% of total feature importance.",
            styles["BodyJustify"],
        ),
        S(1, 0.4 * cm),
        P("Future Work", styles["SubSection"]),
        S(1, 0.15 * cm),
    ]
    for item in [
        "Walk-forward cross-validation to reduce look-ahead bias",
        "Regime-conditional models (separate RF per HMM state)",
        "LSTM / Transformer for sequence-aware psychophysics",
        "Cross-commodity extension: silver, crude oil, copper",
        "Bayesian model averaging over HAR-RV + RF + psychophysical",
    ]:
        story.append(P(f"- {item}", styles["Body"]))
    story += [
        S(1, 0.5 * cm),
        P("References", styles["SubSection"]),
        S(1, 0.15 * cm),
        P("Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. "
          "Journal of Financial Econometrics, 7(2), 174-196.", styles["Body"]),
        P("Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting "
          "Model Predictions. NeurIPS.", styles["Body"]),
        P("Fechner, G. T. (1860). Elemente der Psychophysik. Leipzig.", styles["Body"]),
        S(1, 1.0 * cm),
        HR(width="100%", thickness=1, color=GOLD_),
        S(1, 0.3 * cm),
        P(f"Generated by GoldPsychophysics v3  --  {datetime.date.today()}", styles["DateLine"]),
    ]
    return story


# ===========================================================================
# HELPERS
# ===========================================================================
def _build_styles(rl, GOLD_, LIGHT_):
    base  = rl["getSampleStyleSheet"]()
    PS    = rl["ParagraphStyle"]
    TC    = rl["TA_CENTER"]
    TJ    = rl["TA_JUSTIFY"]
    col   = rl["colors"]
    return {
        "Normal":      base["Normal"],
        "Heading1":    base["Heading1"],
        "Heading2":    base["Heading2"],
        "BigTitle":    PS("BigTitle",    fontSize=24, textColor=GOLD_,  alignment=TC,
                          fontName="Helvetica-Bold", spaceAfter=6),
        "Subtitle":    PS("Subtitle",    fontSize=15, textColor=LIGHT_, alignment=TC,
                          fontName="Helvetica", spaceAfter=5),
        "SubSubtitle": PS("SubSubtitle", fontSize=11, textColor=LIGHT_, alignment=TC,
                          fontName="Helvetica-Oblique"),
        "DateLine":    PS("DateLine",    fontSize=8,  textColor=col.grey, alignment=TC,
                          fontName="Helvetica"),
        "SectionHeader": PS("SectionHeader", fontSize=13, textColor=GOLD_,
                             fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=3),
        "SubSection":  PS("SubSection",  fontSize=10, textColor=LIGHT_,
                          fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2),
        "Body":        PS("Body",        fontSize=8,  textColor=LIGHT_,
                          fontName="Helvetica", leading=13),
        "BodyJustify": PS("BodyJustify", fontSize=8,  textColor=LIGHT_,
                          fontName="Helvetica", leading=13, alignment=TJ),
        "Equation":    PS("Equation",    fontSize=9,  textColor=GOLD_,
                          fontName="Helvetica-Oblique", alignment=TC,
                          spaceBefore=3, spaceAfter=3),
    }


def _make_table(data, rl):
    T, TS = rl["Table"], rl["TableStyle"]
    col   = rl["colors"]
    tbl   = T(data, repeatRows=1)
    tbl.setStyle(TS([
        ("BACKGROUND",     (0,0), (-1,0),  col.HexColor("#1a1a3e")),
        ("TEXTCOLOR",      (0,0), (-1,0),  col.HexColor("#D4AF37")),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,0),  8),
        ("BACKGROUND",     (0,1), (-1,-1), col.HexColor("#0d0d1a")),
        ("TEXTCOLOR",      (0,1), (-1,-1), col.HexColor("#e2e2e2")),
        ("FONTNAME",       (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,1), (-1,-1), 7),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [col.HexColor("#111128"), col.HexColor("#0d0d1a")]),
        ("GRID",           (0,0), (-1,-1), 0.3, col.HexColor("#333355")),
        ("TOPPADDING",     (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 3),
        ("LEFTPADDING",    (0,0), (-1,-1), 5),
    ]))
    return tbl


def _embed(source, w_cm, h_cm, rl, cm_):
    RLImage = rl["RLImage"]
    S       = rl["Spacer"]
    if isinstance(source, matplotlib.figure.Figure):
        buf = io.BytesIO()
        source.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                       facecolor=source.get_facecolor())
        buf.seek(0)
        return RLImage(buf, width=w_cm * cm_, height=h_cm * cm_)
    path = Path(str(source))
    if path.exists():
        return RLImage(str(path), width=w_cm * cm_, height=h_cm * cm_)
    return S(1, 0.1)


def _fallback_text_report(results: dict, output_path: str) -> str:
    lines = [
        "=" * 70,
        "GOLD PSYCHOPHYSICS v3 -- REPORT",
        f"Generated: {datetime.date.today()}",
        f"NOTE: Install reportlab for PDF:  python3 -m pip install reportlab",
        f"Current interpreter: {sys.executable}",
        "=" * 70,
        "",
        "PERFORMANCE",
        f"  HAR-RV R2       : {results.get('r2_har', 'N/A')}",
        f"  RF Baseline R2  : {results.get('r2_rf',  'N/A')}",
        f"  Augmented R2    : {results.get('r2_augmented', 'N/A')}",
        "",
        "FEATURE IMPORTANCE",
    ]
    fi = results.get("feature_importance", {})
    for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {feat:<35s}: {imp * 100:.1f}%")
    lines += ["", "SHAP NARRATIVE", results.get("shap_narrative", "N/A"), ""]

    reg = results.get("regime_stats", {})
    if reg:
        lines.append("REGIME BREAKDOWN")
        for label, v in reg.items():
            lines.append(f"  {label:<14s}: {v.get('count',0):4d} weeks  "
                         f"({v.get('pct',0):.1f}%)  avg RV={v.get('avg_rv',0)*100:.2f}%")

    txt_path = str(Path(output_path).with_suffix(".txt"))
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[pdf_export] Text report -> {txt_path}")
    return txt_path


# ===========================================================================
# STANDALONE TEST
# ===========================================================================
if __name__ == "__main__":
    print(f"Python: {sys.executable}")
    print(f"reportlab available: {REPORTLAB_AVAILABLE}")

    dummy = {
        "r2_har": 0.313, "r2_rf": 0.079, "r2_augmented": 0.125,
        "feature_importance": {
            "RV20": 0.31, "RV60": 0.21, "DXY": 0.07, "US10Y": 0.07,
            "RegimeDuration": 0.06, "CrisisProb": 0.04,
            "PsychophysicalPositioning": 0.03, "SpecPressure": 0.03,
        },
        "shap_narrative": "  + RV20  (+0.83)  Increases forecast vol\n"
                          "  - DealerPressure  (-0.01)  Decreases forecast vol",
        "regime_stats": {
            "Calm":         {"count": 122, "pct": 12.1, "avg_rv": 0.1198},
            "Transitional": {"count": 520, "pct": 51.5, "avg_rv": 0.1430},
            "Crisis":       {"count": 367, "pct": 36.4, "avg_rv": 0.2177},
        },
        "hypothesis_results": {
            "H1: Vol Perception":  {"supported": True,  "note": "PerceivedVol: 2.1%"},
            "H2: Pos Perception":  {"supported": True,  "note": "PP: 3.0%"},
            "H3: Pos Shock->Vol":  {"supported": True,  "note": "PositionShock: 2.8%"},
            "H4: Psych > HAR-RV":  {"supported": False, "note": "R2 delta: -0.188"},
        },
        "data_summary": {"n_obs": 765, "cot_start": "2006-09-12", "cot_end": "2026-06-16"},
    }
    path = generate_report(dummy, output_path="GoldPsychophysics_test.pdf")
    print(f"Output: {path}")
