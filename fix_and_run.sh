#!/bin/bash
# ============================================================
# fix_and_run.sh
# Run this once to fix the Python environment and launch the
# full Gold Psychophysics pipeline correctly.
# ============================================================

echo ""
echo "=== Step 1: Check which python3 has the packages ==="
python3 -c "import reportlab; print('reportlab OK:', reportlab.Version)"
python3 -c "import shap; print('shap OK:', shap.__version__)"
python3 -c "import hmmlearn; print('hmmlearn OK')"

echo ""
echo "=== Step 2: Install any missing packages ==="
python3 -m pip install shap reportlab hmmlearn yfinance --quiet

echo ""
echo "=== Step 3: Run the pipeline ==="
python3 main.py --skip-cot-rebuild

echo ""
echo "=== Step 4: Check outputs ==="
ls -lh outputs/
ls -lh GoldPsychophysics.pdf 2>/dev/null || echo "PDF not found — check above for errors"
