# ARIMA-GARCH Time Series Analysis for Cryptocurrencies (v28.5)

## Overview

This Python script performs a comprehensive time series analysis of cryptocurrency price data using combined ARIMA-GARCH models. It automates data fetching from Yahoo Finance, preprocessing, optional automated model selection (tuning), forecasting under different scenarios, and detailed evaluation including statistical tests and risk metrics.

The script aims to provide a robust and configurable framework for analyzing cryptocurrency price dynamics, volatility, and associated risks (VaR/ES).

## Key Features

*   **Data Handling:**
    *   Fetches historical daily price data using `yfinance`.
    *   Handles missing values (time-based interpolation), removes duplicates, and calculates logarithmic returns.
    *   Performs stationarity tests (ADF, KPSS) on log returns to guide differencing decisions.
*   **Modeling:**
    *   Fits ARIMA models to the conditional mean of the (differenced) log returns.
    *   Fits various GARCH models (GARCH, EGARCH, GJR-GARCH, FIGARCH) to the residuals of the ARIMA fit to capture volatility clustering.
    *   Supports different conditional distributions for GARCH innovations (Normal, Student's t, potentially Skewed-t via `scipy.stats.skewnorm` approximation).
*   **Model Selection & Specification:**
    *   **Automated Tuning (`use_auto_tune=True`):**
        *   Searches over specified ranges of ARIMA (p, q) and GARCH (p, q) orders.
        *   Considers specified GARCH types (`garch_types_to_tune`).
        *   Optionally tests additional differencing orders (`tune_min_d`, `tune_max_d`).
        *   Selects the best model based on an information criterion (AIC or BIC) applied to fits on the Train+Validation data, **subject to parameter significance constraints**.
    *   **Manual Specification (`use_auto_tune=False`):** Allows direct setting of ARIMA (p,d,q) and GARCH (p,q) orders, GARCH type, and distribution.
    *   **Differencing:** Automatically determines the final differencing order (`d`) based on initial stationarity tests, tuning recommendations (if applicable), and manual overrides, constrained by `max_differencing_order`.
*   **Forecasting Modes (`forecast_mode`):**
    *   **`horizon_evaluation`:** Fits one model on Train+Validation data. Generates multi-step forecasts (defined in `evaluation_horizons`) from multiple rolling origins within the test set. Evaluates accuracy at each horizon.
    *   **`backtest`:** Performs a rolling 1-step-ahead forecast evaluation over the test set. Refits the model using a moving window (`fitting_window_size`) at specified intervals (`refit_interval`).
    *   **`future`:** Fits one model on all available data (Train+Validation) and generates forecasts for `future_forecast_horizon` steps into the future.
*   **Evaluation & Benchmarking:**
    *   Calculates standard forecast accuracy metrics:
        *   Price: MAE, RMSE, MAPE (Model vs. Naive).
        *   Variance: RMSE, QLIKE (Model vs. EWMA).
    *   Performs Diebold-Mariano (DM) tests (`dieboldmariano` library) to statistically compare predictive accuracy:
        *   Price: ARIMA-GARCH vs. Naive.
        *   Variance: ARIMA-GARCH vs. EWMA benchmark.
*   **Risk Management Evaluation:**
    *   Calculates parametric Value-at-Risk (VaR) and Expected Shortfall (ES) based on model forecasts (mean, volatility, distribution).
    *   Performs VaR backtesting using Kupiec's POF test and Christoffersen's conditional coverage test (requires `scipy`). Results are reported for both the model and the EWMA benchmark.
*   **Output & Reporting:**
    *   **Logging:** Detailed step-by-step logging of the analysis process.
    *   **Console Output:**
        *   Initial model parameters (from fit on Train+Val).
        *   Evaluation results printed during the run (mode-dependent).
        *   A final, formatted vertical summary per coin, detailing configuration, key results (accuracy, DM tests, VaR backtests), and generated recommendations.
    *   **File Output (in `plot_dir_base`/`coin_id`/):**
        *   Diagnostic plots (ACF, PACF, Q-Q plot) (`.png`).
        *   Initial Parameter Table (`.png`).
        *   Forecast Plots (Price & Volatility) (`.png`).
        *   VaR Threshold Plots (in `backtest` and `horizon_evaluation` modes) (`.png`).
        *   Parameter Stability Plot (in `backtest` mode) (`.png`).
        *   Detailed Forecast Results (`.csv`) - Contains actuals, forecasts (model & benchmark), variance, VaR/ES, violations per step/horizon.
    *   **File Output (in `plot_dir_base`/):**
        *   `FINAL_SUMMARY_All_Coins_[mode].csv`: Aggregated summary results across all analyzed coins in CSV format.
        *   `FINAL_SUMMARY_All_Coins_[mode].png`: PNG image table of the aggregated summary (requires `dataframe_image`).
*   **Robustness:**
    *   Includes error handling for data fetching, model fitting (convergence warnings, linear algebra errors), and numerical issues (e.g., division by zero, log(0)).
    *   Uses specific warning filters to reduce console noise from common `statsmodels` and `arch` warnings.
    *   Automatically changes the working directory to the script's location to ensure relative paths in `CONFIG` work reliably.

## Configuration (`CONFIG` Dictionary)

Modify these parameters directly in the script (`VAR Model - ARIMA GARCH FINAL.py`) to control the analysis:

**General & Data:**

*   `coins_to_analyze`: `List[str]` - List of cryptocurrency IDs (e.g., `"bitcoin"`, `"ethereum"`). Must have corresponding entries in `CRYPTO_SYMBOLS`.
*   `start_date`, `end_date`: `str` (YYYY-MM-DD) or `None` - The period for fetching historical data. `None` for `end_date` fetches up to the latest available data.
*   `split_ratios`: `Tuple[float, float, float]` - Ratios for Train, Validation, and Test sets (must sum to 1.0).
*   `min_data_length`: `int` - Minimum number of rows required *after preprocessing* to proceed with analysis for a coin.
*   `max_differencing_order`: `int` - The maximum order of differencing (`d`) allowed, even if suggested higher by tests or tuning. (Set to 0 to prevent differencing).
*   `adf_significance`, `kpss_significance`: `float` - Significance levels for the stationarity tests.

**Forecasting Mode & Parameters:**

*   `forecast_mode`: `str` - Choose the analysis type:
    *   `'horizon_evaluation'`: Multi-step evaluation on a fixed model.
    *   `'backtest'`: Rolling 1-step evaluation with refitting.
    *   `'future'`: Forecast future periods.
*   `evaluation_horizons`: `List[int]` - (Used in `horizon_evaluation`) List of forecast horizons (in days) to evaluate (e.g., `[1, 7, 30]`).
*   `fitting_window_size`: `int` - (Used in `backtest`) The number of days in the rolling window used for refitting the model.
*   `refit_interval`: `int` - (Used in `backtest`) How often (in steps/days) the model is refit. `1` means refit every day.
*   `future_forecast_horizon`: `int` - (Used in `future`) Number of days to forecast into the future.

**Model Selection (Auto-Tune):** (Used if `use_auto_tune = True`)

*   `use_auto_tune`: `bool` - If `True`, automatically search for the best model orders. If `False`, use `manual_...` parameters.
*   `tune_min_p`, `tune_max_p`: `int` - Range of ARIMA 'p' orders to test.
*   `tune_min_q`, `tune_max_q`: `int` - Range of ARIMA 'q' orders to test.
*   `tune_min_d`, `tune_max_d`: `int` - Range of *additional* differencing orders 'd' to test (applied *on top* of any differencing suggested by initial stationarity tests, capped by `max_differencing_order`). Set min/max to 0 to disable testing additional differencing during tuning.
*   `tune_min_gp`, `tune_max_gp`: `int` - Range of GARCH 'p' orders to test.
*   `tune_min_gq`, `tune_max_gq`: `int` - Range of GARCH 'q' orders to test.
*   `garch_types_to_tune`: `List[str]` - GARCH model types to include in the search (e.g., `['GARCH', 'GJR', 'FIGARCH']`). Note: 'EGARCH' can sometimes be unstable during tuning.
*   `tune_criterion`: `str` - Information criterion (`'AIC'` or `'BIC'`) used to rank candidate models during tuning.
*   `param_significance_level`: `float` - P-value threshold (e.g., 0.10 or 0.05) for checking parameter significance. Models with insignificant core parameters (highest lags, constants) are penalized during tuning.
*   `verbose_tuning`: `bool` - If `True`, print detailed progress and intermediate results during auto-tuning.

**Model Selection (Manual):** (Used if `use_auto_tune = False` or if auto-tuning fails)

*   `manual_arima_order`: `Tuple[int, int, int]` - (p, d, q) order for the ARIMA model. Note: The 'd' here acts as a manual override for differencing.
*   `manual_garch_order`: `Tuple[int, int]` - (p, q) order for the GARCH model.
*   `garch_distribution_manual_choice`: `str` - Assumed distribution for GARCH innovations (`'normal'`, `'t'`, `'skewt'`).
*   `garch_vol_model_manual`: `str` - The GARCH model type (`'GARCH'`, `'EGARCH'`, `'GJR'`, `'FIGARCH'`).

**Evaluation & Benchmarking:**

*   `dm_test_loss_type`: `str` - Loss function for Diebold-Mariano tests on *price/log return* forecasts (`'Squared Error'` or `'Absolute Error'`).
*   `dm_test_variance_loss_type`: `str` - Loss function for Diebold-Mariano tests on *variance* forecasts (`'QLIKE'` or `'Squared Error'`). `QLIKE` is generally preferred for variance.
*   `dm_test_alpha`: `float` - Significance level (alpha) for DM tests and VaR backtesting (Kupiec, Christoffersen).
*   `ewma_lambda`: `float` - Smoothing factor (lambda, between 0 and 1) for the Exponentially Weighted Moving Average (EWMA) variance benchmark.
*   `qlike_epsilon`: `float` - Small positive value to avoid division by zero or log(0) in QLIKE calculations.

**Output & Plotting:**

*   `plot_dir_base`: `str` - The **relative** or **absolute** path for the main results directory. Subdirectories for each coin will be created here. If relative, it's based on the script's location.
*   `generate_parameter_tables`: `bool` - Whether to save PNG images of model parameter tables.
*   `generate_console_parameter_output`: `bool` - Whether to print model parameter summaries to the console.
*   `generate_stability_plots`: `bool` - Whether to generate parameter stability plots (only relevant in `backtest` mode).

**Advanced/Technical:**

*   `default_scale_factor`: `float` - Factor used to scale returns before fitting (can sometimes help convergence). Forecasts are automatically scaled back.
*   `min_fitting_window_size`: `int` - Minimum data points required in the fitting window for the `backtest` mode to attempt a refit.
*   `min_test_set_size`: `int` - Minimum required size for the test set after splitting. Adjusted automatically if needed for `horizon_evaluation`.

## How to Run

1.  **Set up Environment:** Create and activate a Python virtual environment.
    ```bash
    python -m venv venv_crypto_ts
    # On Linux/macOS:
    source venv_crypto_ts/bin/activate
    # On Windows:
    .\venv_crypto_ts\Scripts\activate
    ```
2.  **Install Dependencies:** Use pip to install the required libraries. Create a `requirements.txt` file with the following content (or generate it from your environment if you installed manually):
    ```text
    # requirements.txt
    pandas>=1.3
    numpy>=1.20
    statsmodels>=0.13
    arch>=5.0
    yfinance>=0.2
    matplotlib>=3.5
    scipy>=1.7
    dieboldmariano>=1.0 
    dataframe_image>=0.1 
    # Add specific versions if needed, e.g., pandas==2.0.3
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `dieboldmariano` and `dataframe_image` are external libraries.*

3.  **Configure:** Open `VAR Model - ARIMA GARCH FINAL.py` and modify the `CONFIG` dictionary at the top to match your desired analysis (coins, dates, mode, tuning options, etc.).
4.  **Run Script:** Execute the script from your terminal. The script will handle changing the directory.
    ```bash
    python "VAR Model - ARIMA GARCH FINAL.py"
    ```

## Output Interpretation

*   **Console Summary:** Pay attention to the final vertical summary for each coin. It provides:
    *   Basic run info and data sizes.
    *   Selected model specification (ARIMA/GARCH orders, distribution).
    *   Key evaluation metrics (RMSE, MAPE, QLIKE) comparing the model (AG) to benchmarks (Naive/EWMA) for the relevant mode (`backtest` or `horizon_evaluation`).
    *   Diebold-Mariano test results (p-values and significance interpretation).
    *   VaR Backtesting results (Violation counts, Kupiec/Christoffersen p-values).
    *   Final recommendations based on the tests.
*   **CSV Files:**
    *   `[coin_id]_rolling_1step_results_d[d].csv` (backtest): Contains step-by-step actuals, forecasts, variance, VaR/ES, violations. Useful for detailed analysis or custom plotting.
    *   `[coin_id]_multi_horizon_eval_details_d[d].csv` (horizon): Contains actuals and forecasts for each evaluated horizon and origin date.
    *   `FINAL_SUMMARY_All_Coins_[mode].csv`: The aggregated data used for the final console summary.
*   **Plots:** Visualize forecasts, volatility, residuals, and parameter stability to assess model performance and fit.

## Dependencies

*   Python 3.7+
*   pandas
*   numpy
*   statsmodels
*   arch
*   yfinance
*   matplotlib
*   scipy
*   dieboldmariano (install via `pip install dieboldmariano`)
*   dataframe_image (install via `pip install dataframe_image`, may require `conda install -c conda-forge firefox geckodriver` or similar for backend)

See `requirements.txt` for specific versions.