#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv_thesis"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip

# Prefer repo requirements if present
if [ -f "$ROOT/requirements_final.txt" ]; then
  python -m pip install -r "$ROOT/requirements_final.txt"
fi

# Install package + optional extras (if available)
python -m pip install -e ".[reports,dev]" || python -m pip install -e .

# Run the full pipeline (backtest mode by default)
python -m arima_garch \
  --plot-dir thesis_results \
  --forecast-mode backtest \
  --coins bitcoin,ethereum,dogecoin,solana \
  | tee "$ROOT/output_run.log"
