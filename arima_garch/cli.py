"""CLI entrypoint for the ARIMA-GARCH pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

from . import pipeline


def _load_config(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object/dict")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ARIMA-GARCH crypto analysis")
    parser.add_argument("--config", type=str, help="Path to JSON config override")
    parser.add_argument("--workdir", type=str, help="Working directory to run in")
    parser.add_argument("--plot-dir", type=str, help="Override plot_dir_base")
    parser.add_argument("--forecast-mode", type=str, choices=["backtest", "horizon_evaluation", "future"], help="Forecast mode")
    parser.add_argument("--coins", type=str, help="Comma-separated coin IDs (e.g., bitcoin,ethereum)")
    parser.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="End date YYYY-MM-DD (or empty for latest)")

    args = parser.parse_args()

    if args.workdir:
        os.chdir(args.workdir)

    override: Dict[str, Any] = {}
    if args.config:
        override.update(_load_config(Path(args.config)))

    if args.plot_dir:
        override["plot_dir_base"] = args.plot_dir
    if args.forecast_mode:
        override["forecast_mode"] = args.forecast_mode
    if args.coins:
        override["coins_to_analyze"] = [c.strip() for c in args.coins.split(",") if c.strip()]
    if args.start_date:
        override["start_date"] = args.start_date
    if args.end_date is not None:
        # allow empty string to mean None
        override["end_date"] = args.end_date or None

    pipeline.main(config_override=override or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
