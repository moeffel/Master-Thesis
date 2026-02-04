#!/usr/bin/env python3
"""Split ARIMA GARCH FINAL.py into modules under arima_garch/."""
from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ARIMA GARCH FINAL.py"
OUT_DIR = ROOT / "arima_garch"


def extract_block(lines: list[str], start_idx: int) -> str:
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        if lines[i].startswith("def "):
            end_idx = i
            break
    return "\n".join(lines[start_idx:end_idx]).rstrip()


def extract_def(source: str, func_name: str) -> str:
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"def {func_name}("):
            return extract_block(lines, i)
    raise ValueError(f"Function not found: {func_name}")


def inject_after_line(text: str, marker: str, insertion: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if marker in line:
            lines.insert(i + 1, insertion)
            return "\n".join(lines)
    return text


def extract_config(source: str) -> str:
    lines = source.splitlines()
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.startswith("CRYPTO_COINS ="):
            start = i
        if start is not None and line.startswith("# --- Import Libraries ---"):
            end = i
            break
    if start is None or end is None:
        raise ValueError("Config block not found")
    block = lines[start:end]
    return "\n".join(block).rstrip()


def write_module(path: Path, header: str, body: str):
    header_txt = textwrap.dedent(header).strip()
    body_txt = textwrap.dedent(body).strip()
    path.write_text(header_txt + "\n\n" + body_txt + "\n", encoding="utf-8")


def main() -> int:
    source = SRC.read_text(encoding="utf-8", errors="ignore")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # __init__.py
    (OUT_DIR / "__init__.py").write_text('"""ARIMA-GARCH package (modularized)."""\n', encoding="utf-8")

    # config.py
    config_body = extract_config(source)
    config_header = """
    # Configuration and symbols
    """
    write_module(OUT_DIR / "config.py", config_header, config_body)

    # logging_utils.py
    logging_header = """
    # Logging utilities
    import logging
    """
    logging_body = """
    log = logging.getLogger("arima_garch")

    class CoinLogFilter(logging.Filter):
        def __init__(self, coin_id="N/A"):
            super().__init__()
            self.coin_id = coin_id
        def filter(self, record):
            record.coin_id = self.coin_id
            return True

    def setup_logger():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)-12s - %(levelname)-8s - %(funcName)-25s [%(coin_id)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        _log = logging.getLogger("arima_garch")
        _filter = CoinLogFilter()
        _log.addFilter(_filter)
        return _log, _filter
    """
    write_module(OUT_DIR / "logging_utils.py", logging_header, logging_body)

    # stats.py
    stats_header = """
    # Descriptive statistics and stationarity tests
    import logging
    import warnings
    from typing import Dict, Any
    import numpy as np
    import pandas as pd
    try:
        from scipy.stats import skew, kurtosis as scipy_kurtosis, jarque_bera
        scipy_skew_kurt_available = True
    except Exception:
        skew = None
        scipy_kurtosis = None
        jarque_bera = None
        scipy_skew_kurt_available = False
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except Exception:
        adfuller = None
        kpss = None

    log = logging.getLogger(__name__)
    """
    adf_def = extract_def(source, "adf_test")
    adf_def = inject_after_line(
        adf_def,
        'log.debug("Performing ADF test...")',
        "    if adfuller is None:\n        return {'p_value':np.nan,'is_stationary':False,'test_statistic':np.nan,'critical_values':{}, 'error': 'adfuller unavailable'}",
    )
    kpss_def = extract_def(source, "kpss_test")
    kpss_def = inject_after_line(
        kpss_def,
        "log.debug(f\"Performing KPSS test (reg='{regression}')...\")",
        "    if kpss is None:\n        return {'p_value':np.nan,'is_stationary':False,'test_statistic':np.nan,'critical_values':{}, 'lags':None, 'error': 'kpss unavailable'}",
    )
    stats_body = "\n\n".join([
        extract_def(source, "compute_descriptive_stats"),
        adf_def,
        kpss_def,
        extract_def(source, "difference_series"),
    ])
    write_module(OUT_DIR / "stats.py", stats_header, stats_body)

    # metrics.py
    metrics_header = """
    # Forecast error metrics
    import logging
    import numpy as np
    import pandas as pd
    log = logging.getLogger(__name__)
    """
    metrics_body = "\n\n".join([
        extract_def(source, "mean_absolute_error"),
        extract_def(source, "mean_squared_error"),
        extract_def(source, "root_mean_squared_error"),
        extract_def(source, "mean_squared_error_variance"),
        extract_def(source, "root_mean_squared_error_variance"),
        extract_def(source, "qlike_loss_calc"),
        extract_def(source, "qlike_loss"),
        extract_def(source, "mean_absolute_percentage_error"),
    ])
    write_module(OUT_DIR / "metrics.py", metrics_header, metrics_body)

    # risk.py
    risk_header = """
    # Risk metrics and backtesting
    import logging
    from typing import Any, Dict, Optional, Tuple
    import numpy as np
    import pandas as pd
    try:
        from scipy.stats import chi2, norm, t as student_t, skewnorm
        scipy_chi2_available = True
    except Exception:
        chi2 = None
        norm = None
        student_t = None
        skewnorm = None
        scipy_chi2_available = False
    log = logging.getLogger(__name__)
    """
    risk_body = "\n\n".join([
        extract_def(source, "value_at_risk"),
        extract_def(source, "calculate_parametric_var_es"),
        extract_def(source, "expected_shortfall"),
        extract_def(source, "kupiec_test"),
        extract_def(source, "christoffersen_test"),
    ])
    write_module(OUT_DIR / "risk.py", risk_header, risk_body)

    # diagnostics.py
    diagnostics_header = """
    # Diagnostic statistical tests
    import logging
    from typing import Any, Dict, Union, Optional, Callable
    import numpy as np
    import pandas as pd
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from .metrics import qlike_loss_calc
    try:
        from dieboldmariano import dm_test as lib_dm_test, NegativeVarianceException
        DIEBOLDMARIANO_LIB_AVAILABLE = True
    except Exception:
        lib_dm_test = None
        NegativeVarianceException = None
        DIEBOLDMARIANO_LIB_AVAILABLE = False
    log = logging.getLogger(__name__)
    """
    diagnostics_body = "\n\n".join([
        extract_def(source, "ljung_box_test"),
        extract_def(source, "arch_test"),
        extract_def(source, "diebold_mariano_test"),
    ])
    write_module(OUT_DIR / "diagnostics.py", diagnostics_header, diagnostics_body)

    # data.py
    data_header = """
    # Data acquisition and preprocessing
    import logging
    from typing import Optional, Tuple
    import numpy as np
    import pandas as pd
    try:
        import yfinance as yf
    except Exception:
        yf = None
    from .config import CRYPTO_SYMBOLS
    log = logging.getLogger(__name__)
    """
    fetch_def = extract_def(source, "fetch_data_yahoo")
    fetch_def = inject_after_line(
        fetch_def,
        "log.info(f\"Fetching data for {coin_id} from Yahoo ({start} to {end})...\")",
        "    if yf is None:\n        raise ImportError(\"yfinance is not installed. Install with `pip install yfinance`.\")",
    )
    preprocess_def = extract_def(source, "preprocess_data")
    preprocess_def = preprocess_def.replace(
        "df_copy['log_return'].replace([np.inf, -np.inf], np.nan, inplace=True)",
        "df_copy['log_return'] = df_copy['log_return'].replace([np.inf, -np.inf], np.nan)",
    )
    data_body = "\n\n".join([
        fetch_def,
        preprocess_def,
        extract_def(source, "train_val_test_split"),
    ])
    write_module(OUT_DIR / "data.py", data_header, data_body)

    # modeling.py
    modeling_header = """
    # Model fitting, forecasting, and tuning
    import logging
    from typing import Tuple, Dict, Any, Optional, Union, List
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    from arch import arch_model
    from .metrics import mean_squared_error, mean_squared_error_variance, qlike_loss, root_mean_squared_error, mean_absolute_error
    from .risk import calculate_parametric_var_es
    from .diagnostics import diebold_mariano_test
    from .stats import adf_test, kpss_test, difference_series
    log = logging.getLogger(__name__)
    """
    modeling_body = "\n\n".join([
        extract_def(source, "fit_arima_garch"),
        extract_def(source, "forecast_arima_garch"),
        extract_def(source, "evaluate_candidate_on_val"),
        extract_def(source, "handle_differencing"),
        extract_def(source, "auto_tune_arima_garch"),
        extract_def(source, "invert_price_forecast"),
    ])
    write_module(OUT_DIR / "modeling.py", modeling_header, modeling_body)

    # plotting.py
    plotting_header = """
    # Plotting utilities
    import logging
    from typing import Any, Optional, List, Dict
    import os
    import math
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.table import Table
        PLOT_AVAILABLE = True
    except Exception:
        matplotlib = None
        plt = None
        Table = None
        PLOT_AVAILABLE = False
    try:
        import dataframe_image as dfi
        DATAFRAME_IMAGE_AVAILABLE = True
    except Exception:
        dfi = None
        DATAFRAME_IMAGE_AVAILABLE = False
    import numpy as np
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except Exception:
        plot_acf = None
        plot_pacf = None
    log = logging.getLogger(__name__)
    """
    plotting_body = "\n\n".join([
        extract_def(source, "get_col_widths"),
        extract_def(source, "create_parameter_table_png"),
        extract_def(source, "print_model_summary_console"),
        extract_def(source, "plot_parameter_stability"),
        extract_def(source, "plot_combined_qq"),
        extract_def(source, "plot_combined_acf_pacf"),
    ])
    write_module(OUT_DIR / "plotting.py", plotting_header, plotting_body)

    # pipeline.py
    pipeline_header = """
    # Main analysis pipeline
    import copy
    import logging
    import os
    import time
    from typing import Dict, Optional
    import numpy as np
    import pandas as pd
    from .config import CONFIG
    from .logging_utils import setup_logger
    from .stats import compute_descriptive_stats, adf_test, kpss_test
    from .data import fetch_data_yahoo, preprocess_data, train_val_test_split
    from .metrics import (
        mean_absolute_error,
        mean_squared_error,
        root_mean_squared_error,
        mean_absolute_percentage_error,
        qlike_loss,
        root_mean_squared_error_variance,
        mean_squared_error_variance,
    )
    from .risk import calculate_parametric_var_es, kupiec_test, christoffersen_test, scipy_chi2_available
    from .diagnostics import diebold_mariano_test, DIEBOLDMARIANO_LIB_AVAILABLE
    from .modeling import (
        fit_arima_garch,
        forecast_arima_garch,
        evaluate_candidate_on_val,
        handle_differencing,
        auto_tune_arima_garch,
        invert_price_forecast,
    )
    from .plotting import (
        create_parameter_table_png,
        print_model_summary_console,
        plot_parameter_stability,
        plot_combined_qq,
        plot_combined_acf_pacf,
        PLOT_AVAILABLE,
    )

    log, current_coin_filter = setup_logger()
    """
    pipeline_body = "\n\n".join([
        extract_def(source, "run_analysis_for_coin"),
        extract_def(source, "main"),
    ])
    # Make main accept optional config overrides
    pipeline_body = pipeline_body.replace(
        "def main():",
        "def main(config_override: Optional[Dict] = None):",
    )
    pipeline_body = inject_after_line(
        pipeline_body,
        "config_main = copy.deepcopy(CONFIG)",
        "    if config_override:\n        config_main.update(config_override)",
    )
    # Replace original __main__ block with CLI entrypoint
    marker = "\nif __name__ == \"__main__\":"
    if marker in pipeline_body:
        pipeline_body = pipeline_body.split(marker)[0].rstrip()
    pipeline_body = pipeline_body + \"\\n\\n\" + \"if __name__ == \\\"__main__\\\":\\n    from .cli import main as cli_main\\n    raise SystemExit(cli_main())\\n\"
    write_module(OUT_DIR / "pipeline.py", pipeline_header, pipeline_body)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
