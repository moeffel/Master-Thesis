# ARIMA GARCH FINAL.py
# v28.5

############################################################################################################
############################################################################################################
##################################### --- CONFIG---#########################################################
############################################################################################################
############################################################################################################

CRYPTO_COINS = ["bitcoin", "ethereum", "dogecoin", "solana"]

CONFIG = {
    "coins_to_analyze": ["bitcoin", "ethereum", "dogecoin", "solana"],
    "start_date": "2023-06-01",  # YYYY-MM-DD
    "end_date": "2025-04-01",    # None = today #"2025-04-01"
    "split_ratios": (0.70, 0.15, 0.15),  # Train, Validation, Test
    "forecast_mode": 'backtest',  # 'horizon_evaluation', 'backtest', 'future' 
    "evaluation_horizons": [1, 3, 7, 14,30],  # Days for horizon evaluation
    "dm_test_loss_type": "Squared Error",  # For price/return DM tests
    "dm_test_alpha": 0.1,  # Significance level for Diebold-Mariano tests
    "ewma_lambda": 0.94,  # Smoothing factor for EWMA benchmark volatility
    "dm_test_variance_loss_type": "QLIKE",  # qlike or squared error
    "fitting_window_size": 90,  # Days for rolling window in 'backtest' mode ##90
    "refit_interval": 7,  # Refit model every N steps in 'backtest' mode
    "future_forecast_horizon": 70,  # Days for 'future' mode forecast
    "use_auto_tune": True,  # Use automated model selection
    "compare_garch_dists": True,  # If not auto-tuning, compare dists
    "manual_arima_order": (1, 0, 1),  # Manual ARIMA order (p, d, q)
    "manual_garch_order": (1, 1),
    "garch_distribution_manual_choice": 't',
    "garch_vol_model_manual": 'FIGARCH', # GARCH, EGARCH, GJR, FIGARCH
    "tune_min_p": 0, "tune_max_p": 2,
    "tune_min_q": 0, "tune_max_q": 2,
    "tune_min_d": 0, "tune_max_d": 1,
    "tune_min_gp": 0, "tune_max_gp": 2,
    "tune_min_gq": 0, "tune_max_gq": 2,
    "garch_types_to_tune": ['GARCH', 'FIGARCH','GJR'], # EGARCH unstable in tuning
    "tune_criterion": 'BIC', # 'AIC' or 'BIC'
    "param_significance_level": 0.10,  # Significance level for parameter significance
    "plot_dir_base": "thesis_results", # Base directory for plots
    "generate_parameter_tables": True,
    "generate_console_parameter_output": True,
    "generate_stability_plots": True, # 
    "verbose_tuning": False,
    "default_scale_factor": 100.0, 
    "max_differencing_order": 0,
    "min_data_length": 250,
    "min_fitting_window_size": 60,  # Minimum fitting window size for backtest
    "min_test_set_size": 80,
    "adf_significance": 0.05,
    "kpss_significance": 0.05,
    "qlike_epsilon": 1e-8
}

CRYPTO_SYMBOLS = {
    "bitcoin":  "BTC-USD",
    "ethereum": "ETH-USD",
    "dogecoin": "DOGE-USD",
    "solana":   "SOL-USD"
}


############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

# --- Import Libraries ---
import warnings
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional, Any, Union, Callable
import itertools
import os
import time
import traceback
import copy
import math
import pandas as pd
try:
    from scipy.stats import chi2
    scipy_chi2_available = True
except ImportError:
    chi2 = None
    scipy_chi2_available = False
    print("Warning: Scipy chi2 distribution not available. Kupiec/Christoffersen tests will fail.")

# --- Suppress specific warnings ---
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='An unsupported index was provided')
warnings.filterwarnings('ignore', message='No supported index is available')
warnings.filterwarnings('ignore', message='The test statistic is outside of the range of p-values')
warnings.filterwarnings('ignore', category=HessianInversionWarning)

# --- Configure Matplotlib Backend ---
import matplotlib
try:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.table import Table
    PLOT_AVAILABLE = True
    print("Matplotlib backend set to 'Agg'. Plotting to file enabled.")
except Exception as e:
    print(f"Warning: Could not set matplotlib backend or import pyplot: {e}. Plotting disabled.")
    plt = None
    Table = None
    PLOT_AVAILABLE = False

# --- Library Imports ---
import numpy as np
from scipy.stats import norm
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
try:
    from dieboldmariano import dm_test as lib_dm_test, NegativeVarianceException
    DIEBOLDMARIANO_LIB_AVAILABLE = True
    print("dieboldmariano library found.")
except ImportError:
    lib_dm_test = None
    NegativeVarianceException = None
    DIEBOLDMARIANO_LIB_AVAILABLE = False
    print("Warning: dieboldmariano library not found. DM tests will be skipped.")

if PLOT_AVAILABLE and plt is not None:
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
else:
    sm = None
    plot_acf = None
    plot_pacf = None
try:
    from scipy.stats import skew, kurtosis as scipy_kurtosis, norm, t as student_t, skewnorm, jarque_bera
    scipy_stats_available = True
    scipy_skew_kurt_available = True
except ImportError:
    norm, student_t, skewnorm, skew, scipy_kurtosis = None, None, None, None, None
    scipy_stats_available = False
    scipy_skew_kurt_available = False
    print("Warning: Scipy unavailable, 'skewt' distribution skipped and pandas skew/kurtosis used.")

try:
    import dataframe_image as dfi
    DATAFRAME_IMAGE_AVAILABLE = True
    print("dataframe_image library found, PNG table export enabled.")
except ImportError:
    dfi = None
    DATAFRAME_IMAGE_AVAILABLE = False
    print("Warning: dataframe_image library not found. PNG table export disabled.")
    from sklearn.model_selection import TimeSeriesSplit

# --- Global Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)-12s - %(levelname)-8s - %(funcName)-25s [%(coin_id)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('ARIMA_GARCH_Analysis')

class CoinLogFilter(logging.Filter):
    def __init__(self, coin_id="N/A"):
        super().__init__()
        self.coin_id = coin_id
    def filter(self, record):
        record.coin_id = self.coin_id
        return True
current_coin_filter = CoinLogFilter()
log.addFilter(current_coin_filter)

# --- Configure Matplotlib Style and Default Params ---
try:
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Modern Style
    plt.style.use('seaborn-v0_8-darkgrid') # Seaborn style for better aesthetics

    # global parameter for all plots
    matplotlib.rcParams.update({
        'figure.figsize': (10, 6),    #standard figure size
        'figure.dpi': 100,            # Standard DPI
        'savefig.dpi': 300,           # DPI for saved figures
        'axes.titlesize': 14,         # titel size
        'axes.labelsize': 12,         # Axislabels
        'xtick.labelsize': 10,        # X‑Tick‑Labels
        'ytick.labelsize': 10,        # Y‑Tick‑Labels
        'legend.fontsize': 10,        # Legend text
        'axes.grid': True,            # show grid
        'grid.linestyle': '--',       # dashed grid lines
        'grid.alpha': 0.7,            # tramsparency of grid lines
        'axes.prop_cycle': cycler('color', plt.cm.Set2.colors)  # color cycle
    })
except Exception as style_e:
    log.warning(f"Could not set matplotlib style: {style_e}")
    
# --- Validate Config ---
config_check = copy.deepcopy(CONFIG) # Use copy for validation

if config_check['forecast_mode'] == 'horizon_evaluation':
    if not DIEBOLDMARIANO_LIB_AVAILABLE:
         log.warning("Diebold-Mariano Tests in 'horizon_evaluation' cannot be performed because the 'dieboldmariano' library is missing.")
    if not config_check.get('evaluation_horizons'):
         log.warning("evaluation_horizons is empty, setting default [1, 7,14, 30]")
         config_check['evaluation_horizons'] = [1, 7, 14, 30]
    valid_horizons = [h for h in config_check.get('evaluation_horizons', []) if isinstance(h, int) and h > 0]
    if not valid_horizons:
         log.warning("No valid positive horizons in evaluation_horizons, setting [1, 7, 14]") # Corrected default
         valid_horizons = [1, 7, 14, 30]
         config_check['evaluation_horizons'] = valid_horizons

    current_max_horizon = max(valid_horizons) if valid_horizons else 0
    min_required_test_size = current_max_horizon + 10 # Add buffer for DM test stability
    if config_check['min_test_set_size'] < min_required_test_size:
        log.warning(f"CONFIG CHECK: min_test_set_size ({config_check['min_test_set_size']}) too small for max horizon ({current_max_horizon}) + buffer. Setting to {min_required_test_size}.")
        config_check['min_test_set_size'] = min_required_test_size

if config_check['forecast_mode'] == 'backtest':
     if not DIEBOLDMARIANO_LIB_AVAILABLE:
         log.warning("Diebold-Mariano Tests in 'backtest' cannot be performed because the 'dieboldmariano' library is missing.")
     min_fit_window = config_check.get('min_fitting_window_size', 10)
     fit_window = config_check.get('fitting_window_size', 1)
     if fit_window < min_fit_window:
          log.warning(f"fitting_window_size ({fit_window}) < min_fitting_window_size ({min_fit_window}). Adjusting fitting_window_size.")
          config_check['fitting_window_size'] = min_fit_window
     if config_check.get('refit_interval', 1) < 1:
          log.warning(f"refit_interval ({config_check.get('refit_interval')}) must be >= 1. Setting to 1.")
          config_check['refit_interval'] = 1

if not isinstance(config_check.get('ewma_lambda'), (float, int)) or not (0 < config_check['ewma_lambda'] < 1):
    log.warning(f"Invalid ewma_lambda ({config_check.get('ewma_lambda')}). Must be in (0, 1). Setting to 0.94.")
    config_check['ewma_lambda'] = 0.94 # Default value
valid_var_loss_types = ["QLIKE", "Squared Error"]
if config_check.get('dm_test_variance_loss_type') not in valid_var_loss_types:
    log.warning(f"Invalid dm_test_variance_loss_type ({config_check.get('dm_test_variance_loss_type')}). Must be in {valid_var_loss_types}. Setting to QLIKE.")
    config_check['dm_test_variance_loss_type'] = "QLIKE"
if config_check.get('dm_test_variance_loss_type') == "QLIKE" and config_check.get('qlike_epsilon', 1e-8) <= 0:
    log.warning(f"Invalid qlike_epsilon ({config_check.get('qlike_epsilon')}). Must be > 0. Setting to 1e-8.")
    config_check['qlike_epsilon'] = 1e-8

# Use validated config from now on
CONFIG = config_check

############################################################################################################
# --- UTILITY FUNCTIONS ---
log.info("Defining helper functions...")

def compute_descriptive_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Computes descriptive statistics for price and log return columns."""
    stats_dict: Dict[str, float] = {}
    log.debug("Computing descriptive statistics...")
    def _add_stats(series: pd.Series, prefix: str, target_dict: Dict[str, float]):
        series_numeric = pd.to_numeric(series, errors='coerce').dropna()
        if series_numeric.empty:
            log.warning(f"'{prefix}' series empty or not numeric")
            target_dict[f'{prefix}_mean'] = np.nan
            target_dict[f'{prefix}_std'] = np.nan
            target_dict[f'{prefix}_min'] = np.nan
            target_dict[f'{prefix}_max'] = np.nan
            target_dict[f'{prefix}_skew'] = np.nan
            target_dict[f'{prefix}_kurtosis'] = np.nan
            return
        try:
            n = len(series_numeric)
            target_dict[f'{prefix}_mean'] = float(series_numeric.mean())
            target_dict[f'{prefix}_std'] = float(series_numeric.std()) if n > 1 else 0.0
            target_dict[f'{prefix}_min'] = float(series_numeric.min())
            target_dict[f'{prefix}_max'] = float(series_numeric.max())
            min_obs_sk = 4 # Min observations for robust skew/kurtosis
            if n >= min_obs_sk:
                vals_1d = series_numeric.values
                if hasattr(vals_1d, 'ndim') and vals_1d.ndim != 1: # Ensure 1D for scipy
                    vals_1d = vals_1d.flatten()
                if scipy_skew_kurt_available and skew is not None and scipy_kurtosis is not None:
                    try:
                        target_dict[f'{prefix}_skew'] = float(skew(vals_1d))
                        target_dict[f'{prefix}_kurtosis'] = float(scipy_kurtosis(vals_1d, fisher=True)) # Fisher's definition (normal=0)
                        
                        if jarque_bera is not None:
                            jb_result = jarque_bera(vals_1d)
                            target_dict[f'{prefix}_jb_stat'] = float(jb_result.statistic)
                            target_dict[f'{prefix}_jb_pvalue'] = float(jb_result.pvalue)
                        else:
                            target_dict[f'{prefix}_jb_stat'] = np.nan
                            target_dict[f'{prefix}_jb_pvalue'] = np.nan
                            
                    except Exception as sk_e:
                        log.error(f"Error calculating Scipy Skew/Kurtosis for {prefix}: {sk_e}")
                        target_dict[f'{prefix}_skew'] = np.nan
                        target_dict[f'{prefix}_kurtosis'] = np.nan
                else:
                    log.warning(f"Using Pandas Skew/Kurtosis for {prefix} (SciPy unavailable or failed).")
                    temp_series = pd.Series(vals_1d)
                    target_dict[f'{prefix}_skew'] = float(temp_series.skew())
                    target_dict[f'{prefix}_kurtosis'] = float(temp_series.kurtosis()) # Pandas uses Fisher=True by default
            else:
                log.debug(f"Need {min_obs_sk} numeric values for Skew/Kurtosis '{prefix}' (had {n})")
                target_dict[f'{prefix}_skew'] = np.nan
                target_dict[f'{prefix}_kurtosis'] = np.nan
        except Exception as e:
            log.error(f"Statistics calculation error for '{prefix}': {e}")
            # Ensure keys exist even on error
            target_dict.setdefault(f'{prefix}_mean', np.nan)
            target_dict.setdefault(f'{prefix}_std', np.nan)
            target_dict.setdefault(f'{prefix}_min', np.nan)
            target_dict.setdefault(f'{prefix}_max', np.nan)
            target_dict.setdefault(f'{prefix}_skew', np.nan)
            target_dict.setdefault(f'{prefix}_kurtosis', np.nan)

    if 'price' in df:
        _add_stats(df['price'], 'price', stats_dict)
    if 'log_return' in df:
        _add_stats(df['log_return'], 'logret', stats_dict)
    log.debug("Descriptive statistics calculated.")
    return stats_dict

def adf_test(series: pd.Series, significance: float = 0.05) -> Dict[str, Any]:
    """Performs the Augmented Dickey-Fuller test for stationarity."""
    log.debug("Performing ADF test...")
    series_clean = series.dropna()
    result_dict = {'p_value':np.nan,'is_stationary':False,'test_statistic':np.nan,'critical_values':{}, 'error': None}
    if series_clean.empty:
        log.warning("ADF skipped: Empty series")
        result_dict['error'] = "Empty series"
        return result_dict
    try:
        # Ensure finite values
        series_clean = series_clean[np.isfinite(series_clean)]
        if series_clean.empty:
            log.warning("ADF skipped: Empty after non-finite filtering")
            result_dict['error'] = "Empty after non-finite filtering"
            return result_dict
        # Ensure 1D
        if hasattr(series_clean, 'ndim') and series_clean.ndim != 1:
             log.warning(f"ADF Input NOT 1D: Shape={series_clean.shape}, attempting flatten.")
             if isinstance(series_clean, pd.DataFrame) and series_clean.shape[1] == 1:
                 series_clean = series_clean.iloc[:, 0]
             elif isinstance(series_clean, np.ndarray):
                 series_clean = series_clean.flatten()
             else: # Attempt generic conversion
                 series_clean = pd.Series(np.asarray(series_clean).flatten(), index=getattr(series_clean, 'index', None))
        # Ensure Series or ndarray
        if not isinstance(series_clean, (pd.Series, np.ndarray)):
             series_clean = pd.Series(series_clean)

        result = adfuller(series_clean, autolag='BIC')
        p_val = result[1]
        is_stat = bool(p_val < significance) # ADF null hypothesis is non-stationarity
        log.info(f"ADF Test: p-value={p_val:.4f}, Stationary={is_stat} (alpha={significance})")
        result_dict.update({'p_value':p_val,'is_stationary':is_stat, 'test_statistic': result[0],'critical_values': result[4]})
    except ValueError as ve:
        if "1-dimensional" in str(ve):
            log.critical(f"ADF failed: Dimension error: {ve}") # Critical as it indicates upstream data issues
        elif "maximum lag" in str(ve).lower():
            log.warning(f"ADF failed: likely due to small sample size: {ve}")
        else:
            log.error(f"ADF ValueError: {ve}")
        result_dict['error'] = str(ve)
    except Exception as e:
        log.error(f"ADF failed: unexpected error: {e}")
        result_dict['error'] = str(e)
    return result_dict

def kpss_test(series: pd.Series, significance: float = 0.05, regression: str = 'c') -> Dict[str, Any]:
    """Performs the KPSS test for stationarity."""
    log.debug(f"Performing KPSS test (reg='{regression}')...")
    series_clean = series.dropna()
    result_dict = {'p_value':np.nan,'is_stationary':False,'test_statistic':np.nan,'critical_values':{}, 'lags':None, 'error': None}
    if series_clean.empty:
        log.warning("KPSS skipped: Empty series")
        result_dict['error'] = "Empty series"
        return result_dict
    try:
        # Ensure finite values
        series_clean = series_clean[np.isfinite(series_clean)]
        if series_clean.empty:
            log.warning("KPSS skipped: Empty after non-finite filtering")
            result_dict['error'] = "Empty after non-finite filtering"
            return result_dict
        # Ensure 1D
        if hasattr(series_clean, 'ndim') and series_clean.ndim != 1:
             log.warning(f"KPSS Input NOT 1D: Shape={series_clean.shape}, attempting flatten.")
             if isinstance(series_clean, pd.DataFrame) and series_clean.shape[1] == 1:
                 series_clean = series_clean.iloc[:, 0]
             elif isinstance(series_clean, np.ndarray):
                 series_clean = series_clean.flatten()
             else: # Attempt generic conversion
                 series_clean = pd.Series(np.asarray(series_clean).flatten(), index=getattr(series_clean, 'index', None))
        # Ensure Series or ndarray
        if not isinstance(series_clean, (pd.Series, np.ndarray)):
             series_clean = pd.Series(series_clean)
        # Check length
        min_len_kpss = 5 # Heuristic minimum length for KPSS
        if len(series_clean) < min_len_kpss:
            log.warning(f"KPSS skipped: Series length {len(series_clean)} too short.")
            result_dict['error'] = f"Series length {len(series_clean)} too short"
            return result_dict

        # Run KPSS test, catch specific warning about p-value range
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The test statistic is outside of the range of p-values')
            try:
                # Use recommended lag calculation based on Schwert (1989) default in statsmodels
                n_lags = math.floor(12 * (len(series_clean) / 100)**(1/4))
                kpss_stat, p_val, lags, crit_vals = kpss(series_clean, regression=regression, nlags=n_lags)
            except ValueError as kpss_ve:
                 # Fallback if calculated nlags is too large for the sample
                 if "nlags is larger than" in str(kpss_ve).lower():
                      log.warning(f"KPSS nlags={n_lags} failed ({kpss_ve}), trying 'legacy'.")
                      kpss_stat, p_val, lags, crit_vals = kpss(series_clean, regression=regression, nlags='legacy')
                 else:
                     raise kpss_ve # Re-raise other ValueErrors

        is_stat = bool(p_val >= significance) # KPSS null hypothesis is stationarity
        log.info(f"KPSS Test (reg='{regression}', lags={lags}): p-value={p_val:.4f}, Stationary={is_stat} (alpha={significance})")
        result_dict.update({'p_value':p_val,'is_stationary':is_stat, 'test_statistic': kpss_stat,'critical_values': crit_vals, 'lags': lags})
    except ValueError as ve:
        if "1-dimensional" in str(ve):
            log.critical(f"KPSS failed: Dimension error: {ve}") # Critical
        else:
            log.error(f"KPSS ValueError: {ve}")
        result_dict['error'] = str(ve)
    except Exception as e:
        log.error(f"KPSS failed: unexpected error: {e}")
        result_dict['error'] = str(e)
    return result_dict

def difference_series(series: pd.Series, order: int = 1) -> pd.Series:
    """Applies differencing to a pandas Series."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series) # Attempt conversion
    if order <= 0:
        return series.copy()
    if order >= len(series):
        log.warning(f"Differencing order {order} >= series length {len(series)}. Returning empty series.")
        empty_index = series.index[:0] if hasattr(series, 'index') else pd.Index([])
        return pd.Series(dtype=series.dtype, index=empty_index)
    log.debug(f"Applying differencing d={order}")
    return series.diff(periods=order).dropna()

# --- Error Metrics ---
def mean_absolute_error(yt:Any,yp:Any)->float:
    """Calculates Mean Absolute Error, ignoring NaNs."""
    yta=np.asarray(yt)
    ypa=np.asarray(yp)
    mask=pd.notna(yta)&pd.notna(ypa)
    return np.nan if not np.any(mask) else float(np.mean(np.abs(yta[mask]-ypa[mask])))

def mean_squared_error(yt:Any,yp:Any)->float:
    """Calculates Mean Squared Error, ignoring NaNs."""
    yta=np.asarray(yt)
    ypa=np.asarray(yp)
    mask=pd.notna(yta)&pd.notna(ypa)
    return np.nan if not np.any(mask) else float(np.mean((yta[mask]-ypa[mask])**2))

def root_mean_squared_error(yt:Any,yp:Any)->float:
    """Calculates Root Mean Squared Error, ignoring NaNs."""
    mse=mean_squared_error(yt,yp)
    return np.nan if np.isnan(mse) else float(np.sqrt(mse))

def mean_squared_error_variance(yt_var:Any,yp_var:Any)->float:
    """Calculates Mean Squared Error for variance forecasts."""
    yta=np.asarray(yt_var)
    ypa=np.asarray(yp_var)
    # Ensure valid (non-NaN, finite, non-negative) pairs for variance MSE
    mask=pd.notna(yta)&pd.notna(ypa)&np.isfinite(yta)&np.isfinite(ypa)&(yta>=0)&(ypa>=0)
    return np.nan if not np.any(mask) else float(np.mean((yta[mask]-ypa[mask])**2))

def root_mean_squared_error_variance(yt_var:Any,yp_var:Any)->float:
    """Calculates Root Mean Squared Error for variance forecasts."""
    mse=mean_squared_error_variance(yt_var,yp_var)
    # Ensure non-negative MSE before sqrt
    return np.nan if (np.isnan(mse) or mse < 0) else float(np.sqrt(max(0,mse)))

def qlike_loss_calc(actual_var: float, forecast_var: float, epsilon: float = 1e-8) -> float:
    """Calculates QLIKE loss for a single point, ensuring positivity."""
    safe_actual = max(epsilon, actual_var)
    safe_forecast = max(epsilon, forecast_var)
    try:
        # QLIKE = actual/forecast - log(actual/forecast) - 1
        loss = safe_actual / safe_forecast - np.log(safe_actual / safe_forecast) - 1
        # QLIKE should theoretically be >= 0. Clamp small negative values due to precision.
        if loss < -epsilon:
            log.debug(f"QLIKE loss < 0 ({loss:.2e}) for actual={actual_var:.2e}, forecast={forecast_var:.2e}. Clamping to 0.")
            return 0.0
        return loss
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        log.warning(f"QLIKE calculation error: {e} for actual={actual_var}, forecast={forecast_var}. Returning NaN.")
        return np.nan

def qlike_loss(yt_var: Any, yp_var: Any, epsilon: float = 1e-8) -> float:
    """Calculates the average QLIKE loss for variance forecasts."""
    yta = np.asarray(yt_var)
    ypa = np.asarray(yp_var)
    mask = pd.notna(yta) & pd.notna(ypa) & np.isfinite(yta) & np.isfinite(ypa) & (yta >= 0) & (ypa >= 0)
    if not np.any(mask):
        return np.nan
    losses = [qlike_loss_calc(yta[i], ypa[i], epsilon) for i in np.where(mask)[0]]
    valid_losses = [l for l in losses if pd.notna(l) and np.isfinite(l)]
    return np.nan if not valid_losses else float(np.mean(valid_losses))

def mean_absolute_percentage_error(yt:Any,yp:Any)->float:
    """Calculates Mean Absolute Percentage Error, ignoring NaNs and zero actuals."""
    yta=np.asarray(yt)
    ypa=np.asarray(yp)
    eps=1e-8 # Threshold to avoid division by zero or near-zero
    mask_nan=pd.notna(yta)&pd.notna(ypa)
    mask_zero=np.abs(yta)>eps # Filter out actual values close to zero
    mask=mask_nan&mask_zero
    if not np.any(mask):
        log.warning("MAPE: No valid pairs found after filtering zeros/NaNs.")
        return np.nan
    ignored=np.sum(~mask_zero&mask_nan)
    if ignored > 0:
        log.debug(f"MAPE: Ignoring {ignored} zero/near-zero values in actuals.")
# Insert Risk Management Metrics code block immediately after MAPE function
    return float(np.mean(np.abs((yta[mask]-ypa[mask])/yta[mask]))*100.0)

# --- Risk Management Metrics ---
def value_at_risk(returns: Any, alpha: float = 0.05) -> float:
    """Calculates the empirical Value-at-Risk (VaR) at level alpha."""
    arr = np.asarray(returns)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        log.warning("VaR: empty returns series.")
        return np.nan
    return np.percentile(arr, 100 * alpha)

def calculate_parametric_var_es(mean_forecast: float,
                                vol_forecast: float,
                                distribution: str,
                                alpha: float = 0.05,
                                dist_params: Optional[Dict] = None) -> Tuple[float, float]:
    """
    Calculates parametric Value-at-Risk (VaR) and Expected Shortfall (ES).

    Args:
        mean_forecast: Forecasted mean (log return).
        vol_forecast: Forecasted volatility (standard deviation).
        distribution: Assumed distribution ('normal', 't', 'skewt').
        alpha: VaR/ES significance level (e.g., 0.05 for 95%).
        dist_params: Dictionary containing distribution parameters if needed
                     (e.g., {'nu': df} for t, {'nu': df, 'lambda': skew} for skewt).

    Returns:
        Tuple (VaR_threshold, ES_value). Returns (NaN, NaN) on error.
    """
    if pd.isna(mean_forecast) or pd.isna(vol_forecast) or vol_forecast < 0:
        # log.debug("NaN/Invalid input for parametric VaR/ES.")
        return np.nan, np.nan
    if vol_forecast < 1e-8: # Avoid issues with zero volatility
        return mean_forecast, mean_forecast # VaR/ES is just the mean if no volatility

    dist_lower = distribution.lower()
    var_thresh = np.nan
    es_val = np.nan

    try:
        if dist_lower == 'normal':
            if norm is None: raise ImportError("Scipy norm not available.")
            q = norm.ppf(alpha) # Quantile (negative for left tail)
            var_thresh = mean_forecast + q * vol_forecast
            # ES for Normal: mu - sigma * pdf(ppf(alpha)) / alpha
            es_val = mean_forecast - vol_forecast * norm.pdf(q) / alpha

        elif dist_lower == 't':
            if student_t is None: raise ImportError("Scipy student_t not available.")
            if dist_params is None or 'nu' not in dist_params or pd.isna(dist_params['nu']):
                log.warning("Student's t degrees of freedom ('nu') missing or invalid. Cannot calc VaR/ES.")
                return np.nan, np.nan
            df = dist_params['nu']
            if df <= 2: # ES is undefined for df<=2
                log.warning(f"Student's t ES undefined for df={df}<=2. Returning NaN for ES.")
                es_val = np.nan
            q = student_t.ppf(alpha, df)
            var_thresh = mean_forecast + q * vol_forecast * np.sqrt(max(0,(df-2)/df)) if df > 2 else mean_forecast + q * vol_forecast # Adjust vol for df>2 var
            # ES for Student's t: mu - sigma_adj * pdf(ppf(alpha,df),df)/alpha * (df + ppf(alpha,df)^2)/(df-1)
            if df > 2: # ES requires df>2 for adjusted sigma
                 sigma_adj = vol_forecast * np.sqrt((df-2)/df)
                 es_val = mean_forecast - sigma_adj * (student_t.pdf(q,df)/alpha) * ((df+q**2)/(df-1))
            elif df > 1: # VaR definable for df>1, ES for df>2
                 var_thresh = mean_forecast + q * vol_forecast # Use unadjusted vol if df<=2
                 # ES cannot be calculated as above for df<=2

        elif dist_lower == 'skewt':
            # Calculation for Skewed-t is more complex and depends on the specific parameterization.
            # Using 'skewnorm' as an approximation or placeholder if skewt not fully implemented.
            if skewnorm is None: raise ImportError("Scipy skewnorm not available.")
            if dist_params is None or 'lambda' not in dist_params or pd.isna(dist_params['lambda']):
                 log.warning("Skew-t skewness parameter ('lambda') missing/invalid. Using skewnorm approx.")
                 skew_param = 0 # Default to normal if skew param missing
            else:
                 skew_param = dist_params['lambda']
            # Note: skewnorm parameters are loc, scale, a (shape)
            # We map mean, vol -> loc, scale based on skew_param
            delta = skew_param / np.sqrt(1 + skew_param**2)
            adj_std = np.sqrt(1 - (2 * delta**2 / np.pi))
            if adj_std < 1e-6: adj_std = 1e-6 # Avoid division by zero
            scale_sn = vol_forecast / adj_std
            loc_sn = mean_forecast - scale_sn * delta * np.sqrt(2 / np.pi)

            q = skewnorm.ppf(alpha, a=skew_param, loc=loc_sn, scale=scale_sn)
            var_thresh = q
            # ES for skew-normal is complex. Placeholder: return VaR. Need specific formula.
            es_val = np.nan # Placeholder
            log.warning("ES calculation for Skew-t distribution not fully implemented. Returning NaN for ES.")

        else:
            log.warning(f"Unsupported distribution '{distribution}' for parametric VaR/ES. Returning NaN.")
            return np.nan, np.nan

        # Final check for NaN results
        if pd.isna(var_thresh): es_val = np.nan # If VaR is NaN, ES must be NaN too

    except ImportError as ie:
        log.error(f"Cannot calculate parametric VaR/ES: {ie}")
        return np.nan, np.nan
    except Exception as e:
        log.error(f"Error calculating parametric VaR/ES for dist='{distribution}': {e}", exc_info=True)
        return np.nan, np.nan

    # VaR is the threshold (a loss, so typically negative), ES is the expected value beyond VaR (more negative)
    return float(var_thresh), float(es_val)

def expected_shortfall(returns: Any, alpha: float = 0.05) -> float:
    """Calculates the Expected Shortfall (ES) at level alpha."""
    arr = np.asarray(returns)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        log.warning("ES: empty returns series.")
        return np.nan
    var = value_at_risk(arr, alpha)
    tail_losses = arr[arr <= var]
    if tail_losses.size == 0:
        # If no losses exceed VaR (e.g., alpha is very small or data has no tail), ES is just VaR
        # Alternative approaches exist, but this is common.
        return var # Or potentially np.nan if interpretation requires exceedances
    return tail_losses.mean()

def kupiec_test(violations: int, n: int, alpha: float = 0.05) -> Dict[str, float]:
    """Performs the Kupiec Proportion-of-Failures (POF) test."""
    # Ensure chi2 is available
    if not scipy_chi2_available or chi2 is None:
         log.error("Kupiec test requires scipy.stats.chi2.")
         return {'p_value': np.nan, 'LR_stat': np.nan, 'error': 'chi2 unavailable'}

    if n == 0:
        log.warning("Kupiec: zero observations.")
        return {'p_value': np.nan, 'LR_stat': np.nan, 'error': 'n=0'}
    if violations < 0 or violations > n:
        log.error(f"Kupiec: Invalid violations count {violations} for n={n}.")
        return {'p_value': np.nan, 'LR_stat': np.nan, 'error': 'Invalid violations count'}
    # Avoid log(0) issues with boundary cases
    if violations == 0:
        p_hat = 1e-9 # Use a tiny number instead of 0
    elif violations == n:
        p_hat = 1.0 - 1e-9 # Use a number very close to 1
    else:
        p_hat = violations / n

    # Likelihood ratio statistic (handle potential log(0) or division by zero if p_hat or alpha are boundaries)
    try:
        loglik_unrestricted = 0
        if violations > 0 and violations < n: # Only calculate if p_hat is not 0 or 1
            loglik_unrestricted = violations * np.log(p_hat) + (n - violations) * np.log(1 - p_hat)

        loglik_restricted = violations * np.log(alpha) + (n - violations) * np.log(1 - alpha)

        LR = -2 * (loglik_restricted - loglik_unrestricted)
        # LR statistic should be non-negative. Clamp numerical inaccuracies.
        if LR < 0: LR = 0

        p_value = 1 - chi2.cdf(LR, df=1)
        return {'LR_stat': float(LR), 'p_value': float(p_value)}
    except (ValueError, FloatingPointError) as e:
        log.warning(f"Kupiec numerical issue (p_hat={p_hat:.3f}, alpha={alpha:.3f}): {e}")
        return {'p_value': np.nan, 'LR_stat': np.nan, 'error': 'Numerical issue'}


def christoffersen_test(violation_series: Any, alpha: float = 0.05) -> Dict[str, float]:
    """Performs Christoffersen conditional coverage test on a series of 0/1 violations."""
    # Ensure chi2 is available
    if not scipy_chi2_available or chi2 is None:
         log.error("Christoffersen test requires scipy.stats.chi2.")
         return {'p_value': np.nan, 'LR_stat': np.nan, 'error': 'chi2 unavailable'}

    x = np.asarray(violation_series).astype(int)
    n = len(x)
    if n < 2:
        log.warning("Christoffersen: too few observations.")
        return {'p_value': np.nan, 'LR_stat': np.nan, 'error': 'n<2'}
    # Count transitions
    n00 = np.sum((x[:-1] == 0) & (x[1:] == 0))
    n01 = np.sum((x[:-1] == 0) & (x[1:] == 1))
    n10 = np.sum((x[:-1] == 1) & (x[1:] == 0))
    n11 = np.sum((x[:-1] == 1) & (x[1:] == 1))

    # Transition probabilities
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    # Overall probability of violation
    pi = (n01 + n11) / n if n > 0 else 0

    # Check for edge cases where probabilities are 0 or 1
    if pi == 0 or pi == 1:
        log.warning(f"Christoffersen: Unconditional exception rate is {pi*100}%. Independence test is not meaningful.")
        # Return Kupiec test result for conditional coverage in this case
        kupiec_res = kupiec_test(violations=int(np.sum(x)), n=n, alpha=alpha)
        kupiec_res['error'] = kupiec_res.get('error', f"Independence test skipped (pi={pi})")
        return kupiec_res # Return Kupiec result as conditional coverage is not testable

    # Likelihood under independence assumption (but observed unconditional probability pi)
    L0 = 1.0
    if pi > 0 and pi < 1:
        try: L0 = ((1 - pi)**(n00 + n10)) * (pi**(n01 + n11))
        except FloatingPointError: L0 = 1e-300 # Handle underflow

    # Likelihood under dependence assumption (Markov chain)
    term1 = (1 - pi0)**n00 if pi0 < 1 else 1.0 # Handle pi0=1 case
    term2 = pi0**n01 if pi0 > 0 else 1.0 # Handle pi0=0 case
    term3 = (1 - pi1)**n10 if pi1 < 1 else 1.0 # Handle pi1=1 case
    term4 = pi1**n11 if pi1 > 0 else 1.0 # Handle pi1=0 case
    L1 = term1 * term2 * term3 * term4

    # Likelihood ratio test for independence
    LR_ind = np.nan
    if L0 > 1e-300 and L1 > 1e-300: # Check for potential zero likelihoods
       try:
            LR_ind = -2 * np.log(L0 / L1)
            if LR_ind < 0: LR_ind = 0 # Clamp numerical errors
       except (ValueError, FloatingPointError) as log_e:
            log.warning(f"Christoffersen LR_ind calculation error (L0={L0:.2e}, L1={L1:.2e}): {log_e}")
            LR_ind = np.nan
    else:
         log.warning(f"Christoffersen likelihoods potentially too small (L0={L0:.2e}, L1={L1:.2e})")

    # Kupiec test for unconditional coverage
    kupiec_res = kupiec_test(violations=int(np.sum(x)), n=n, alpha=alpha)
    LR_uc = kupiec_res.get('LR_stat', np.nan)

    # Combined test statistic (Conditional Coverage)
    LR_cc = np.nan
    if pd.notna(LR_uc) and pd.notna(LR_ind):
        LR_cc = LR_uc + LR_ind
    else:
        log.warning("Cannot calculate combined Christoffersen LR_cc due to NaN components.")

    # P-value from Chi-squared distribution with 2 degrees of freedom
    p_value = 1 - chi2.cdf(LR_cc, df=2) if pd.notna(LR_cc) else np.nan

    return {'LR_stat': float(LR_cc), 'p_value': float(p_value), 'LR_ind': float(LR_ind), 'LR_uc': float(LR_uc)}


# --- Diagnostic Tests ---
def ljung_box_test(r:pd.Series, l:Union[int,List[int]]=20, sig_level:float=0.05) -> Dict[str,Any]:
    """Performs the Ljung-Box test for autocorrelation in residuals."""
    log.debug(f"Ljung-Box Test (Lags={l})...")
    r_c=r.dropna()
    n=len(r_c)
    lags_to_test:Optional[List[int]]=None
    # Max lag feasible: n/2 - 1, must be > 0
    max_lag_possible=max(1,n//2 - 1) if n > 3 else 0

    if isinstance(l,int):
        max_lag=min(l,max_lag_possible) if max_lag_possible > 0 else 0
        lags_to_test=[max_lag] if max_lag>0 else None
    elif isinstance(l,list):
        valid_lags=sorted([lg for lg in l if 0<lg<=max_lag_possible]) if max_lag_possible > 0 else []
        lags_to_test=valid_lags if valid_lags else None

    if lags_to_test is None or n<2:
        log.warning(f"L-B skipped: n={n}, Lags={l}, max_lag_possible={max_lag_possible}.")
        return{'lb_pvalue':np.nan,'is_white_noise':np.nan,'lb_stat':np.nan,'lb_lag_tested':0}

    try:
        max_lag_tested=max(lags_to_test)
        # Ensure lag is less than sample size
        if max_lag_tested >= n -1:
             log.warning(f"L-B skipped: max Lag {max_lag_tested} >= n-1 ({n-1})")
             return{'lb_pvalue':np.nan,'is_white_noise':np.nan,'lb_stat':np.nan,'lb_lag_tested':0}

        lb_result=acorr_ljungbox(r_c,lags=lags_to_test,return_df=True,boxpierce=False) # Use Ljung-Box, not Box-Pierce

        if lb_result.empty or max_lag_tested not in lb_result.index:
            raise ValueError("L-B result format error or lag not found.")

        lb_p_val=lb_result.loc[max_lag_tested,'lb_pvalue']
        lb_stat=lb_result.loc[max_lag_tested,'lb_stat']
        # Null hypothesis: No autocorrelation (residuals are white noise)
        is_wn=bool(lb_p_val>sig_level)
        log.info(f"Ljung-Box Test (Lag={max_lag_tested}): p={lb_p_val:.4f}, White Noise={is_wn}")
        return{'lb_pvalue':float(lb_p_val),'is_white_noise':is_wn,'lb_stat':float(lb_stat),'lb_lag_tested':max_lag_tested}
    except ValueError as ve:
         # Catch potential numerical issues from statsmodels
         if "invalid value encountered in" in str(ve).lower():
             log.error(f"L-B numerical error: {ve}")
         else:
             log.error(f"L-B Test failed: {ve}")
         return{'lb_pvalue':np.nan,'is_white_noise':np.nan,'lb_stat':np.nan,'lb_lag_tested':0}
    except Exception as e:
        log.error(f"L-B Test unexpectedly failed: {e}")
        return{'lb_pvalue':np.nan,'is_white_noise':np.nan,'lb_stat':np.nan,'lb_lag_tested':0}

def arch_test(r:pd.Series, l:Union[int,List[int]]=12, sig_level:float=0.05)->Dict[str,Any]:
    """Performs the ARCH LM test for heteroskedasticity."""
    log.debug(f"ARCH LM Test (Lags={l})...")
    r_c=r.dropna()
    n=len(r_c)
    nlags:int=0
    max_lag_possible=max(1,n//2 - 1) if n > 3 else 0

    if isinstance(l,int):
        nlags=min(l,max_lag_possible) if max_lag_possible > 0 else 0
    elif isinstance(l,list):
        valid_lags=[lg for lg in l if 0<lg<=max_lag_possible] if max_lag_possible > 0 else []
        nlags=max(valid_lags) if valid_lags else 0

    # Need at least nlags+1 observations
    if nlags==0 or n<nlags+1:
        log.warning(f"ARCH LM skipped: n={n}, nlags={nlags} (needs > Lags).")
        return{'arch_pvalue':np.nan,'heteroskedastic':np.nan,'arch_stat':np.nan,'arch_lag_tested':0}

    try:
        # Use het_arch from statsmodels.stats.diagnostic
        lm_stat,p_val,f_stat,fp_val=het_arch(r_c,nlags=nlags,store=False)
        if np.isnan(p_val): # Check for NaN p-value explicitly
            raise ValueError("NaN p-value returned by ARCH test.")
        # Null hypothesis: Homoskedasticity (No ARCH effects)
        is_het=bool(p_val<sig_level)
        log.info(f"ARCH LM Test (Lag={nlags}): p={p_val:.4f}, Heteroskedastic={is_het}")
        return{'arch_pvalue':float(p_val),'heteroskedastic':is_het,'arch_stat':float(lm_stat),'arch_lag_tested':nlags}
    except ValueError as ve:
        if "nlags must be smaller" in str(ve).lower():
             log.error(f"ARCH LM failed (likely small N or large lag): {ve}")
        else:
             log.error(f"ARCH LM ValueError: {ve}")
        return{'arch_pvalue':np.nan,'heteroskedastic':np.nan,'arch_stat':np.nan,'arch_lag_tested':0}
    except Exception as e:
        log.error(f"ARCH LM Test unexpectedly failed: {e}")
        return{'arch_pvalue':np.nan,'heteroskedastic':np.nan,'arch_stat':np.nan,'arch_lag_tested':0}

def diebold_mariano_test(actuals: Union[list, np.ndarray, pd.Series],
                         forecasts_model: Union[list, np.ndarray, pd.Series],
                         forecasts_naive: Union[list, np.ndarray, pd.Series],
                         h: int = 1,
                         loss_type: str = "Squared Error",
                         alternative: str = 'less', # 'less': model1 better, 'greater': model2 better, 'two_sided'
                         qlike_epsilon: float = 1e-8) -> Dict[str, Any]:
    """Performs the Diebold-Mariano test using the 'dieboldmariano' library."""
    log.debug(f"Starting Diebold-Mariano Test ('dieboldmariano' lib) for h={h}, loss='{loss_type}', alternative='{alternative}', var='bartlett'...")
    results = {'dm_stat': np.nan, 'p_value': np.nan, 'horizon': h, 'loss_type': loss_type, 'alternative': alternative, 'n_obs': 0, 'error': None}

    if not DIEBOLDMARIANO_LIB_AVAILABLE or lib_dm_test is None:
        results['error'] = "dieboldmariano library not available."
        log.error(results['error'])
        return results

    # Ensure consistent inputs and handle NaNs/Infs
    actuals_s = pd.Series(actuals).copy()
    fc1_s = pd.Series(forecasts_model).copy()
    fc2_s = pd.Series(forecasts_naive).copy()

    if not (len(actuals_s) == len(fc1_s) == len(fc2_s)):
        results['error'] = f"Input series must have the same length ({len(actuals_s)} vs {len(fc1_s)} vs {len(fc2_s)})"
        log.error(results['error'])
        return results

    valid_mask = actuals_s.notna() & fc1_s.notna() & fc2_s.notna() & np.isfinite(actuals_s) & np.isfinite(fc1_s) & np.isfinite(fc2_s)

    # Additional check for QLIKE: requires positive values
    if loss_type == "QLIKE":
        pos_mask = (actuals_s > 0) & (fc1_s > 0) & (fc2_s > 0)
        valid_mask = valid_mask & pos_mask
        log.debug(f"QLIKE DM Test: After positivity filtering, {valid_mask.sum()} / {len(actuals_s)} observations remain.")


    actuals_clean = actuals_s[valid_mask].tolist()
    fc1_clean = fc1_s[valid_mask].tolist()
    fc2_clean = fc2_s[valid_mask].tolist()

    n_obs = len(actuals_clean)
    results['n_obs'] = n_obs
    min_obs_dm = 10 # Heuristic minimum for DM test stability
    if n_obs < min_obs_dm:
        results['error'] = f"Not enough valid observation triplets ({n_obs}) for DM test (Minimum: {min_obs_dm})"
        log.warning(results['error'])
        return results

    if h < 1:
        results['error'] = f"Forecast horizon h must be >= 1 (given: {h})"
        log.error(results['error'])
        return results

    # Define loss function based on type
    loss_func: Optional[Callable[[float, float], float]] = None
    if loss_type == "Squared Error": loss_func = lambda act, pred: (act - pred) ** 2; log.debug("Using Squared Error loss for DM test.")
    elif loss_type == "Absolute Error": loss_func = lambda act, pred: abs(act - pred); log.debug("Using Absolute Error loss for DM test.")
    elif loss_type == "QLIKE": loss_func = lambda act, pred: qlike_loss_calc(act, pred, qlike_epsilon); log.debug(f"Using QLIKE loss for DM test (epsilon={qlike_epsilon}).")
    else: log.warning(f"Unknown loss_type '{loss_type}'. Using Squared Error."); loss_func = lambda act, pred: (act - pred) ** 2; results['loss_type'] = "Squared Error"

    if loss_func is None: results['error'] = "Could not define a valid loss function."; log.error(results['error']); return results

    # Handle alternative hypothesis for the library (which tests P1 vs P2)
    # We want to test if Model (fc1) is better than Naive (fc2) -> alternative='less'
    is_one_sided = False
    p1 = fc1_clean # Model forecast
    p2 = fc2_clean # Naive/Benchmark forecast
    if alternative == 'less': # H1: Loss(Model) < Loss(Naive)
        is_one_sided = True
    elif alternative == 'greater': # H1: Loss(Model) > Loss(Naive) -> test Loss(Naive) < Loss(Model)
        is_one_sided = True
        p1 = fc2_clean # Swap P1 and P2
        p2 = fc1_clean
    elif alternative == 'two_sided': # H1: Loss(Model) != Loss(Naive)
        is_one_sided = False
    else:
        log.warning(f"Invalid alternative '{alternative}', using 'less'.")
        alternative = 'less'
        results['alternative'] = alternative
        is_one_sided = True

    try:
        # Call the external library function
        # Using Bartlett kernel ('harvey_correction=True' implies Bartlett for h>1 in the lib)
        dm_stat, p_val = lib_dm_test(
            V=actuals_clean,         # Actual values
            P1=p1,                   # Forecasts from model 1 (potentially swapped based on alternative)
            P2=p2,                   # Forecasts from model 2 (potentially swapped based on alternative)
            h=h,                     # Forecast horizon
            loss=loss_func,          # Loss function callable
            one_sided=is_one_sided,  # Whether the test is one-sided
            harvey_correction=True,  # Apply Harvey et al. (1997) correction for small samples
            variance_estimator='bartlett' # Specify variance estimator explicitly
        )

        # Adjust DM stat sign if swapped P1/P2 for 'greater' alternative
        if alternative == 'greater':
            dm_stat = -dm_stat

        results['dm_stat'] = float(dm_stat)
        results['p_value'] = float(p_val)
        log.info(f"DM Test ('dieboldmariano' lib, h={h}, loss='{results['loss_type']}', var='bartlett'): Stat={results['dm_stat']:.4f}, p-value({alternative})={results['p_value']:.4f}, N={n_obs}")

    except ValueError as ve:
         error_msg = str(ve)
         if "math domain error" in error_msg.lower():
             results['error'] = f"DM Test (h={h}, loss='{results['loss_type']}', var='bartlett'): Math domain error (likely numerical instability in variance estimation)"
         elif "log" in error_msg.lower() and loss_type=="QLIKE":
             results['error'] = f"DM Test (h={h}, loss='{results['loss_type']}', var='bartlett'): ValueError, possibly QLIKE related? '{error_msg}'"
         else:
             results['error'] = f"DM Test ('dieboldmariano' lib) ValueError (h={h}, loss='{results['loss_type']}', var='bartlett'): {error_msg}"
         log.error(results['error'])
         results['dm_stat'] = np.nan
         results['p_value'] = np.nan
    except NegativeVarianceException as nve:
         # Specific exception from the library
         results['error'] = f"DM Test (h={h}, loss='{results['loss_type']}', var='bartlett'): NegativeVarianceException. Data might be unsuitable. Details: {nve}"
         log.error(results['error'])
         results['dm_stat'] = np.nan
         results['p_value'] = np.nan
    except Exception as e:
         error_type_name = type(e).__name__
         results['error'] = f"Unexpected error in DM Test ('dieboldmariano' lib, h={h}, loss='{results['loss_type']}', var='bartlett'): {error_type_name}: {e}"
         log.error(results['error'], exc_info=True)
         results['dm_stat'] = np.nan
         results['p_value'] = np.nan

    return results

def get_col_widths(data:List[List[Any]],sf:float=0.018,min_w:float=0.08,max_w:float=0.3)->List[float]:
    """Calculate appropriate column widths for matplotlib table based on content."""
    if not data or not data[0]: return []
    num_cols=len(data[0])
    max_len=[0]*num_cols
    for row in data:
        for j in range(num_cols): max_len[j]=max(max_len[j], math.ceil(len(str(row[j]).strip())))
    # Scale length by factor, apply min/max constraints
    return [min(max_w,max(min_w,l*sf)) for l in max_len]

def create_parameter_table_png(arima:Optional[Any], garch:Optional[Any], filename:str, title:str)->bool:
    """Creates a PNG image containing formatted parameter tables from ARIMA and GARCH results."""
    if not PLOT_AVAILABLE or plt is None or Table is None: log.warning("Plotting disabled, cannot create parameter table PNG."); return False
    if arima is None and garch is None: log.warning("No model results provided for parameter table."); return False

    tables=[]
    summaries=[]
    try: # Extract ARIMA parameters if available
        if arima and hasattr(arima,'summary'):
            s=arima.summary()
            if hasattr(s,'tables') and len(s.tables)>1: # Table 1 usually contains coefficients
                d=[[str(c).strip() for c in r] for r in s.tables[1].data] # Extract data, convert to string, strip whitespace
                if len(d)>1: # Ensure there's data beyond the header
                    tables.append({"title":"ARIMA Parameters","data":d})
                # Add summary stats
                summaries.append(f"ARIMA LLF:{getattr(arima,'llf',np.nan):.2f} AIC:{getattr(arima,'aic',np.nan):.2f} BIC:{getattr(arima,'bic',np.nan):.2f}")
    except Exception as e: log.error(f"Error extracting ARIMA table data: {e}")

    try: # Extract GARCH parameters if available
        if garch and hasattr(garch,'summary'):
            s=garch.summary()
            if hasattr(s,'tables') and len(s.tables)>1: # Table 1 for vol params
                gd=[[str(c).strip() for c in r] for r in s.tables[1].data]
                if len(s.tables)>2: # Table 2 often has distribution params
                    try:
                        gd.extend([[str(c).strip() for c in r] for r in s.tables[2].data])
                    except IndexError: log.debug("No GARCH distribution parameter table found in summary.")
                    except Exception as egd: log.warning(f"Error getting GARCH distribution params: {egd}")
                if len(gd)>1:
                    tables.append({"title":"GARCH Parameters (Vol+Dist)","data":gd})
                summaries.append(f"GARCH LLF:{getattr(garch,'loglikelihood',np.nan):.2f} AIC:{getattr(garch,'aic',np.nan):.2f} BIC:{getattr(garch,'bic',np.nan):.2f}")
    except Exception as e: log.error(f"Error extracting GARCH table data: {e}")

    if not tables: log.warning("No valid parameter tables extracted from model summaries."); return False

    num_rows_total = sum(len(t['data']) for t in tables) # Total rows across all tables
    fig_height = 1.5 + len(tables)*0.8 + num_rows_total*0.3 + len(summaries)*0.2 # Estimate figure height
    fig,ax=plt.subplots(figsize=(10,fig_height))
    ax.axis('off') # Hide axes

    current_y=0.98
    fig.text(0.5,current_y,title,ha='center',va='top',fontsize=14,weight='bold')
    current_y-=0.10

    # Add summary stats text
    if summaries:
        summary_text="\n".join(summaries)
        fig.text(0.05,current_y,summary_text,ha='left',va='top',fontsize=9,family='monospace')
        current_y-=(len(summaries)*0.02 + 0.05)

    # Add each table
    for table_info in tables:
        table_data=table_info['data']
        table_title=table_info['title']
        if not table_data or len(table_data)<=1: continue # Skip empty tables

        fig.text(0.5,current_y,table_title,ha='center',va='top',fontsize=11,weight='bold')
        current_y-=0.06

        rows,cols=len(table_data),len(table_data[0])
        try:
            col_widths=get_col_widths(table_data)
            # Estimate table height and width, position it
            table_height=min(0.9/max(1,len(tables)),rows*0.04) # Allocate vertical space
            bbox_width=min(0.95,sum(col_widths)+0.1) # Calculate required width
            bbox_x=max(0.02,0.5-bbox_width/2) # Center horizontally
            bbox=[bbox_x,max(0.01,current_y-table_height),bbox_width,table_height]

            tab=ax.table(cellText=table_data[1:],colLabels=table_data[0],colWidths=col_widths,
                           cellLoc='center',loc='upper center',bbox=bbox)
            tab.auto_set_font_size(False)
            tab.set_fontsize(7)

            # Style header
            for j in range(cols):
                cell=tab.get_celld().get((0,j)) # Header cell is row 0
                if cell: cell.set_text_props(weight='bold',color='white'); cell.set_facecolor('#000080') # Dark blue header

            current_y-=(table_height + 0.07) # Move down for next element
        except Exception as te: log.error(f"Error creating table '{table_title}' in PNG: {te}"); ax.text(0.5,current_y,f"Error rendering: {table_title}",ha='center',va='top',color='red'); current_y-=0.1

    try:
        fig.subplots_adjust(top=0.92,bottom=0.02,left=0.02,right=0.98) # Adjust layout
        fig.savefig(filename,dpi=150,bbox_inches='tight')
        log.info(f"Parameter table PNG saved: {filename}")
        plt.close(fig)
        return True
    except Exception as e: log.error(f"Error saving PNG {filename}: {e}"); plt.close(fig); return False

def print_model_summary_console(arima:Optional[Any], garch:Optional[Any], coin_id:str)->None:
    """Prints a formatted summary of ARIMA and GARCH model parameters to the console."""
    print(f"\n--- [{coin_id.upper()}] Model Parameter Summary ---")

    def _print(title:str, result_obj:Any):
        print(f"\n{title}:")
        if result_obj is None: print("  Model N/A."); return

        try:
            summary=result_obj.summary()
            # Check if summary has the expected table structure
            if not hasattr(summary,'tables') or len(summary.tables)<2:
                # Fallback: print basic info if summary structure is unexpected
                print(f"  Incomplete Summary. Basic Info:")
                print(f"   Class: {type(result_obj)}")
                llf = getattr(result_obj,'llf',getattr(result_obj,'loglikelihood',np.nan))
                aic = getattr(result_obj,'aic',np.nan)
                bic = getattr(result_obj,'bic',np.nan)
                nobs = getattr(result_obj,'nobs',np.nan)
                params = getattr(result_obj,'params', None)
                pvals = getattr(result_obj,'pvalues', None)
                print(f"   LLF: {llf:.3f}, AIC: {aic:.3f}, BIC: {bic:.3f}, Nobs: {nobs}")
                if params is not None:
                     print("   Parameters:")
                     params_df = pd.DataFrame({'coef': params})
                     if pvals is not None and len(pvals) == len(params):
                         params_df['P>|z|'] = pvals
                         # Format p-value nicely
                         params_df['P>|z|'] = pd.to_numeric(params_df['P>|z|'], errors='coerce').map('{:.3f}'.format)
                     print(params_df.to_string(float_format="{:.4f}".format, na_rep='N/A'))
                return

            param_table=summary.tables[1] # Usually parameters
            try:
                # Try parsing the HTML representation for better formatting
                html_content = param_table.as_html()
                df = None
                if not html_content or "</table>" not in html_content:
                     log.warning(f"Summary Table 1 ({title}) empty/invalid. Raw:")
                     print(param_table.as_text())
                else:
                    try:
                        # Requires lxml or html5lib
                        df = pd.read_html(html_content, header=0, index_col=0)[0]
                    except ImportError:
                        log.error("Pandas read_html requires 'lxml' or 'html5lib'. Install one.")
                        print("  ERROR: Could not read HTML table (missing library).")
                        print("  Raw:\n"+param_table.as_text())
                    except ValueError as ve_html:
                        log.error(f"Pandas read_html ValueError: {ve_html}")
                        print("  ERROR: Could not read HTML table (Pandas error).")
                        print("  Raw:\n"+param_table.as_text())

                if df is None:
                    print("  Could not parse Summary Table 1.")
                else:
                     # Select and format common columns
                     cols=['coef','std err','z','P>|z|']
                     existing_cols=[c for c in cols if c in df.columns]
                     if not existing_cols:
                         print(f"  Standard columns not found. Raw:\n{df.to_string(na_rep='N/A')}")
                     else:
                         if 'P>|z|' in df.columns:
                             # Add significance stars
                             df['p_num']=pd.to_numeric(df['P>|z|'],errors='coerce')
                             df['Sig.'] = df['p_num'].apply(lambda p: "***" if pd.notna(p) and p<0.01 else ("** " if pd.notna(p) and p<0.05 else ("*  " if pd.notna(p) and p<0.10 else "   ")))
                             df['P>|z|']=df['p_num'].map('{:.3f}'.format) # Format p-value
                             existing_cols.append('Sig.')
                             df=df.drop(columns=['p_num'])
                         # Print selected columns
                         print_cols = [c for c in existing_cols if c in df.columns]
                         print(df[print_cols].to_string(float_format="{:.4f}".format,na_rep='N/A'))

            except Exception as e_parse:
                 print(f"  Error parsing Table 1: {e_parse}. Raw:\n{param_table.as_text()}")

            # Print GARCH distribution parameters if available (usually Table 2)
            if title.startswith("GARCH") and len(summary.tables)>2:
                print("\n  Distribution Parameters:")
                dist_table=summary.tables[2]
                try:
                    html_dt = dist_table.as_html()
                    dist_df = None
                    if not html_dt or "</table>" not in html_dt:
                        log.warning(f"Distribution Params Table empty. Raw:")
                        print(dist_table.as_text())
                    else:
                        try:
                           dist_df = pd.read_html(html_dt,header=0,index_col=0)[0]
                        except ImportError:
                           log.error("Pandas read_html requires 'lxml' or 'html5lib'. Install one.")
                           print("  ERROR: Could not read HTML table (missing library).")
                           print("  Raw:\n"+dist_table.as_text())
                        except ValueError as ve_html_dist:
                           log.error(f"Pandas read_html ValueError (Dist): {ve_html_dist}")
                           print("  ERROR: Could not read HTML table (Pandas error).")
                           print("  Raw:\n"+dist_table.as_text())

                        if dist_df is not None and not dist_df.empty:
                           print(dist_df.to_string(float_format="{:.4f}".format,na_rep='N/A'))
                        elif dist_df is None:
                           pass # Error already printed
                        else: # Parsed but empty
                            print("  Could not parse Distribution Params table (empty).")

                except Exception as e_parse_dist:
                    print(f"  Error parsing Distribution Params: {e_parse_dist}. Raw:\n{dist_table.as_text()}")

            # Print overall model stats
            llf=getattr(result_obj,'llf',getattr(result_obj,'loglikelihood',np.nan))
            aic=getattr(result_obj,'aic',np.nan)
            bic=getattr(result_obj,'bic',np.nan)
            print(f"\n  LLF: {llf:.3f}, AIC: {aic:.3f}, BIC: {bic:.3f}")

        except Exception as e:
             print(f"  Unexpected error generating summary: {e}")
             print(f"  Object representation: {result_obj}")

    _print("ARIMA Parameters",arima)
    _print("GARCH Parameters",garch)
    print("--- End Parameter Summary ---")

def plot_parameter_stability(param_data:Dict[str,list], dates:list, filename:str, title:str)->bool:
    """Plots the evolution of selected model parameters over time (for backtesting)."""
    if not PLOT_AVAILABLE or plt is None: log.warning("Plotting disabled, cannot create stability plot."); return False
    if not param_data or not dates: log.warning("No data provided for stability plot."); return False

    try: plot_dates=pd.to_datetime(dates)
    except Exception as e: log.error(f"Date conversion failed for stability plot: {e}"); return False

    # Filter for parameters that have data and are not all NaN
    valid_params={k:v for k,v in param_data.items() if len(v)==len(plot_dates) and any(pd.notna(x) for x in v)}
    if not valid_params: log.warning("No valid parameters found for stability plot."); return False

    n_params=len(valid_params)
    fig_height = max(5.0, n_params * 2.0) # Adjust height based on number of parameters
    fig,axes=plt.subplots(n_params,1,figsize=(10, fig_height),sharex=True,squeeze=False)
    axes=axes.flatten() # Ensure axes is always a 1D array

    fig.suptitle(title,fontsize=14,y=0.99)
    param_names=list(valid_params.keys())

    for i,name in enumerate(param_names):
        ax=axes[i]
        values=valid_params[name]
        # Convert to numeric, create Series with dates, drop NaNs for plotting
        param_series=pd.Series(pd.to_numeric(values,errors='coerce'),index=plot_dates).dropna()

        if not param_series.empty:
            ax.plot(param_series.index,param_series.values,marker='.',linestyle='-',markersize=3,label=name,lw=1.2)
            mean_val=param_series.mean()
            std_dev = param_series.std()
            ax.axhline(mean_val,color='r',linestyle='--',linewidth=0.8,label=f'Mean: {mean_val:.4f}')
            # Add +/- 1 std dev shading if std dev is meaningful
            if pd.notna(std_dev) and std_dev > 1e-8:
                ax.fill_between(param_series.index, mean_val - std_dev, mean_val + std_dev, color='red', alpha=0.1, label=f'+/- 1 StdDev ({std_dev:.4f})')
            ax.legend(loc='best',fontsize='small')
            ax.grid(True,linestyle=':',alpha=0.6)
        else:
            # Handle case where parameter is all NaN after conversion
            ax.text(0.5,0.5,f'No valid data for {name}',ha='center',va='center',transform=ax.transAxes, color='grey')
            ax.grid(True,linestyle=':',alpha=0.6)

        # Clean up parameter name for display
        display_name = name.replace('[','').replace(']','') # Remove brackets often used in statsmodels names
        ax.set_ylabel(display_name)

    # Configure bottom axis
    axes[-1].set_xlabel("Date")
    plt.setp(axes[-1].xaxis.get_majorticklabels(),rotation=45,ha='right')
    fig.tight_layout(pad=0.5)
    try:
        fig.savefig(filename, dpi=100)
        log.info(f"Stability plot saved: {filename}")
        plt.close(fig)
        return True
    except Exception as e:
        log.error(f"Error saving stability plot {filename}: {e}")
        plt.close(fig)
        return False

# --- Data Handling ---
def fetch_data_yahoo(coin_id: str, start: Optional[str]=None, end: Optional[str]=None) -> pd.DataFrame:
    """Fetches historical price data from Yahoo Finance."""
    log.info(f"Fetching data for {coin_id} from Yahoo ({start} to {end})...")
    if coin_id not in CRYPTO_SYMBOLS: raise ValueError(f"Unknown Coin ID: {coin_id}")
    ticker=CRYPTO_SYMBOLS[coin_id]

    try:
        data=yf.download(ticker,start=start,end=end,progress=False,auto_adjust=False, timeout=30)
    except Exception as e:
        log.error(f"Download failed for {ticker}: {e}")
        raise ValueError(f"Data download failed: {e}")

    if data.empty: raise ValueError(f"No data received for {ticker}")

    # Find Date and Price columns robustly
    df=data.copy()
    date_source=None
    if isinstance(df.index, pd.DatetimeIndex):
        log.debug("Date found in index.")
        date_source = df.index
    else:
        # Check common column names, reset index if necessary
        potential_date_cols = ['Date', 'Datetime', 'index']
        df_reset = df.reset_index()
        for col in potential_date_cols:
             if col in df_reset.columns:
                 log.debug(f"Date found in column: '{col}'")
                 date_source = df_reset[col]
                 df = df_reset # Use reset df if date found in column
                 break
        # Check index name if not found in columns
        if date_source is None:
            if df.index.name in potential_date_cols:
                 log.debug("Date found in index name.")
                 df_reset = df.reset_index()
                 date_source = df_reset[df.index.name]
                 df = df_reset
            else:
                raise KeyError(f"No date source found. Columns: {list(df.columns)}, Index Name: {df.index.name}")

    price_source=None
    potential_price_cols = ['Close', 'Adj Close']
    col_names_list = list(df.columns) # Get column names (can be tuples if multi-index)
    for potential_col in potential_price_cols:
        found = False
        for col_name in col_names_list:
            # Handle simple and multi-index column names
            current_col_label = col_name[0] if isinstance(col_name, tuple) else col_name
            if current_col_label == potential_col:
                price_source = df[col_name]
                log.debug(f"Using price column: '{col_name}'.")
                found = True
                break
        if found: break

    # Try case-insensitive if not found
    if price_source is None:
        lower_mapping = {}
        original_cols = {}
        for i, col_name in enumerate(col_names_list):
             label = col_name[0] if isinstance(col_name, tuple) else col_name
             lower_label = label.lower()
             lower_mapping[lower_label] = col_name
             original_cols[i] = col_name # Store original name by position/label if needed
        for potential_col in potential_price_cols:
             potential_col_lower = potential_col.lower()
             if potential_col_lower in lower_mapping:
                 original_col_name = lower_mapping[potential_col_lower]
                 price_source = df[original_col_name]
                 log.debug(f"Using price column (case-insensitive): '{original_col_name}'.")
                 break

    if price_source is None: raise KeyError(f"Could not find closing price column ({potential_price_cols}). Columns found: {list(df.columns)}")

    # Ensure price_source is 1D (Series or flattened array)
    if isinstance(price_source, pd.DataFrame):
        if price_source.shape[1] == 1:
            log.warning("Price source is DataFrame, converting to Series.")
            price_source = price_source.iloc[:, 0]
        else:
            raise ValueError(f"Price source is DataFrame with unexpected shape: {price_source.shape}")
    elif isinstance(price_source, np.ndarray) and price_source.ndim > 1:
        if price_source.ndim == 2 and price_source.shape[1] == 1:
            log.warning(f"Price source is 2D NumPy ({price_source.shape}), flattening.")
            price_source = price_source.flatten()
        else:
            raise ValueError(f"Price source is NumPy array with unexpected shape: {price_source.shape}")

    # Create final DataFrame
    series_index = date_source if date_source is not None and len(date_source) == len(price_source) else None
    if series_index is None and date_source is not None:
        log.warning(f"Length mismatch between date ({len(date_source)}) and price ({len(price_source)}). Index will not be set initially.")

    # Ensure price_source is a Series before creating DataFrame
    if not isinstance(price_source, pd.Series):
        price_source = pd.Series(price_source, index=series_index, name='price')

    try:
        if date_source is None: raise ValueError("Date source is None.")
        if not isinstance(date_source, (pd.Series, pd.Index)):
            date_source = pd.Series(date_source, name='date')
        # Build DataFrame
        final_df = pd.DataFrame({'date': date_source, 'price': price_source})
    except ValueError as e:
        log.error(f"DataFrame creation failed. Price type: {type(price_source)}, Shape: {getattr(price_source, 'shape', 'N/A')}, Ndim: {getattr(price_source, 'ndim', 'N/A')}, Date type: {type(date_source)}, Len: {len(date_source) if date_source is not None else 'None'}")
        raise e

    # Standardize types
    try: final_df['date']=pd.to_datetime(final_df['date'])
    except Exception as e: raise ValueError(f"Error parsing date column: {e}")
    try: final_df['price']=pd.to_numeric(final_df['price'],errors='coerce')
    except Exception as e: log.error(f"Price conversion to numeric error: {e}"); raise TypeError(f"Price conversion failed: {e}")

    if final_df['price'].isnull().all(): raise ValueError("Price column is entirely NaN.")
    if final_df['price'].isnull().any(): log.warning(f"Found {final_df['price'].isnull().sum()} NaNs in price column.")

    # Sort, remove duplicates, reset index
    final_df=final_df.sort_values("date").drop_duplicates(subset='date', keep='first').reset_index(drop=True)
    log.info(f"{len(final_df)} rows fetched and prepared for {coin_id}.")
    return final_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the raw data: handles NaNs, duplicates, calculates log returns."""
    log.info("Preprocessing data...")
    if df.empty: raise ValueError("Input DataFrame is empty.")
    df_copy=df.copy()
    if 'date' not in df_copy.columns or 'price' not in df_copy.columns: raise KeyError("Requires 'date' and 'price' columns.")

    # Convert types, handle initial NaNs
    df_copy['date']=pd.to_datetime(df_copy['date'])
    df_copy['price']=pd.to_numeric(df_copy['price'],errors='coerce')
    df_copy.dropna(subset=['date','price'],inplace=True)
    if df_copy.empty: raise ValueError("DataFrame empty after dropping NaNs in date/price.")

    # Set index, sort, handle duplicates
    df_copy=df_copy.set_index('date').sort_index()
    if df_copy.index.has_duplicates:
        num_dupes = df_copy.index.duplicated().sum()
        log.warning(f"Found {num_dupes} duplicate dates. Keeping first occurrence.")
        df_copy = df_copy[~df_copy.index.duplicated(keep='first')]

    # Resample to daily frequency, filling potential gaps
    df_copy = df_copy.resample('D').first() # Use first price if multiple on same day (after duplicate removal)

    # Interpolate missing prices (e.g., weekends if original data had gaps)
    nan_count=df_copy['price'].isnull().sum()
    if nan_count>0:
        log.info(f"Interpolating {nan_count} missing daily prices (using time method)...")
        # Time interpolation is suitable for financial data
        df_copy["price"]=df_copy["price"].interpolate(method="time",limit_direction="both", limit_area=None)
        nan_count_after = df_copy['price'].isnull().sum()
        if nan_count_after > 0:
            log.warning(f"{nan_count_after} NaNs remain after interpolation. Removing rows.")
            df_copy.dropna(subset=["price"],inplace=True)

    if df_copy.empty: raise ValueError("DataFrame empty after interpolation.")

    # Calculate log returns
    if df_copy['price'].nunique() <= 1:
        log.warning("Price series is constant or near-constant. Log returns will be 0/NaN.")
        df_copy['log_return'] = 0.0
    else:
        # Use log1p for slightly more numerical stability with small percentage changes
        df_copy['log_return']=np.log1p(df_copy['price'].pct_change())
        # Replace potential infinities from division by zero (e.g., price going from 0 to non-zero)
        df_copy['log_return'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop first row (NaN log return) and any other rows with NaN returns
    df_copy=df_copy.dropna(subset=['log_return'])
    if df_copy.empty: raise ValueError("DataFrame empty after calculating log returns and dropping NaNs.")

    # Reset index and select final columns
    final_df=df_copy.reset_index()[["date","price","log_return"]]
    # Ensure final dtypes are correct
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df['price'] = pd.to_numeric(final_df['price'])
    final_df['log_return'] = pd.to_numeric(final_df['log_return'])

    log.info(f"Preprocessing finished. Final shape: {final_df.shape}")
    return final_df

def train_val_test_split(df: pd.DataFrame, ratios: Tuple[float, float, float]=(0.7,0.15,0.15), min_test_size: int=30) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the DataFrame into training, validation, and test sets based on ratios,
       prioritizing ratios unless the resulting test set is smaller than min_test_size."""
    n=len(df)
    log.info(f"Splitting {n} rows, Ratios={ratios}, min_test_size={min_test_size}...")
    tr_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(tr_ratio + val_ratio + test_ratio, 1.0): raise ValueError(f"Split ratios {ratios} must sum to 1.0")
    if n < min_test_size + 20: # Basic check for minimum data
        log.warning(f"Data size {n} potentially too small for split with min_test_size {min_test_size}. Proceeding cautiously.")
        # Fallback to simple min_test_size enforcement if n is very small
        if n < min_test_size: raise ValueError(f"Data size {n} is less than min_test_size {min_test_size}.")
        test_idx_start = max(0, n - min_test_size)
        val_idx_start = max(0, test_idx_start - 1) # Minimal validation if possible
        train_idx_end = val_idx_start
        log.warning(f"Fallback split due to small N: Train={train_idx_end}, Val=1, Test={n-test_idx_start}")
        return df.iloc[:train_idx_end].copy(), df.iloc[train_idx_end:test_idx_start].copy(), df.iloc[test_idx_start:].copy()


    # 1. Calculate split points based *only* on ratios
    train_idx_end_ratio = int(n * tr_ratio)
    test_idx_start_ratio = int(n * (tr_ratio + val_ratio))
    # Ensure indices are within bounds for ratio calculation
    train_idx_end_ratio = max(0, min(train_idx_end_ratio, n))
    test_idx_start_ratio = max(train_idx_end_ratio, min(test_idx_start_ratio, n))

    # 2. Calculate the resulting test set size based on ratios
    test_size_ratio = n - test_idx_start_ratio

    # 3. Check if ratio-based test set meets minimum size
    if test_size_ratio < min_test_size:
        log.warning(f"Ratio-based test set size ({test_size_ratio}) is smaller than min_test_size ({min_test_size}). Adjusting split to meet minimum.")
        # Enforce minimum test size by adjusting test_idx_start
        test_idx_start = max(0, n - min_test_size)
        # Adjust validation start to not overlap test, but try to respect original train ratio if possible
        val_idx_start = min(train_idx_end_ratio, test_idx_start)
        val_idx_start = max(0, val_idx_start) # Ensure validation start is not negative
        test_idx_start = max(val_idx_start, test_idx_start) # Ensure test starts after validation

        log.warning(f"Adjusted split: Train end index = {val_idx_start}, Test start index = {test_idx_start}")

    else:
        # Ratios result in a large enough test set, use ratio-based indices
        log.info("Ratio-based split meets minimum test size requirement.")
        val_idx_start = train_idx_end_ratio
        test_idx_start = test_idx_start_ratio

    # Perform the final split using the determined indices
    train = df.iloc[:val_idx_start].copy()
    val = df.iloc[val_idx_start:test_idx_start].copy()
    test = df.iloc[test_idx_start:].copy()

    log.info(f"Final split result: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    # Final confirmation (should always be >= min_test_size now unless n was too small initially)
    if len(test) < min_test_size and n >= min_test_size:
         log.error(f"Logic error: Final test size {len(test)} is still less than minimum {min_test_size}!")

    return train,val,test

# --- Modeling Functions ---
def fit_arima_garch(series:pd.Series,arima_order:Tuple[int,int,int],garch_order:Tuple[int,int],garch_type:str='GARCH',distribution:str='t',sf:float=100.0)->Tuple[Optional[Any],Optional[Any],float]:
    """Fits an ARIMA model followed by a GARCH model on the residuals."""
    # Note: arima_order d is ignored here, differencing is handled externally.
    # Fit ARIMA(p, 0, q) on the (potentially already differenced) series.
    p, _, q = arima_order # Unpack p, d, q but use only p, q for ARIMA fit
    gp, gq = garch_order
    log.debug(f"Fitting ARIMA({p},0,{q}) + {garch_type}({gp},{gq}) dist={distribution}... Input length: {len(series)}")

    actual_vol_model = garch_type.upper()
    # Determine if GARCH model requires asymmetry term 'o' (usually 1 for EGARCH, GJR etc.)
    o_asym = 1 if actual_vol_model in ['GJR','TARCH','EGARCH','APARCH'] else 0

    # Minimum length checks
    min_len_arima = max(p, q) + 5 # Heuristic
    min_len_garch = max(gp, gq, o_asym) + 5 # Heuristic
    min_len_practical = 15 # Absolute minimum
    min_len = max(min_len_arima, min_len_garch, min_len_practical)

    if len(series)<min_len:
        log.error(f"Series length {len(series)} < required minimum {min_len} for orders {(p,q), (gp,gq)}.")
        raise ValueError(f"Series length {len(series)} < required minimum {min_len}")

    # Scale the series (often helps convergence)
    s_sc=series*sf
    arima_result=None
    garch_result=None

    try:
        # Fit ARIMA(p, 0, q) model
        log.debug(f"Fitting ARIMA({p},0,{q}) with trend='c'...")
        # Ensure 1D input for ARIMA
        if hasattr(s_sc, 'ndim') and s_sc.ndim != 1:
             log.warning(f"ARIMA input not 1D (Shape {s_sc.shape}), flattening.")
             s_sc = s_sc.iloc[:, 0] if isinstance(s_sc, pd.DataFrame) and s_sc.shape[1] == 1 else np.asarray(s_sc).flatten()
             s_sc = pd.Series(s_sc, index=getattr(s_sc, 'index', None)) # Try to preserve index
        if not isinstance(s_sc, (pd.Series, np.ndarray)): s_sc = pd.Series(s_sc) # Ensure Series/ndarray

        # Instantiate ARIMA model
        # enforce_stationarity/invertibility=False allows fitting non-stationary models if needed,
        # though we aim for stationarity via external differencing. 'trend=c' includes a constant.
        arima_model=ARIMA(s_sc,order=(p,0,q),trend='c',enforce_stationarity=False,enforce_invertibility=False)
        try:
            # Fit the model with potentially tighter tolerances for convergence
            arima_result=arima_model.fit(method_kwargs={"xtol": 1e-6, "ftol": 1e-6, "maxiter": 1000})
            log.debug(f"ARIMA fit successful: LLF={arima_result.llf:.2f}")
        except (np.linalg.LinAlgError, ValueError, ConvergenceWarning, HessianInversionWarning) as arima_fit_e:
            log.warning(f"ARIMA fit failed: {type(arima_fit_e).__name__}: {arima_fit_e}")
            return None, None, sf # Return None if ARIMA fails
        except Exception as arima_fit_generic_e:
            log.error(f"Unexpected ARIMA fit error: {arima_fit_generic_e}")
            return None, None, sf

        # Fit GARCH model on ARIMA residuals
        residuals_scaled=arima_result.resid.copy()
        if residuals_scaled.isnull().any() or not np.all(np.isfinite(residuals_scaled)):
            log.error("ARIMA residuals contain NaN or Inf. Cannot fit GARCH.")
            return arima_result,None,sf # Return ARIMA result, but no GARCH

        # Check residual length for GARCH
        min_len_garch_resid = max(gp, gq, o_asym) + 5
        if len(residuals_scaled)<min_len_garch_resid:
            log.error(f"Residual length {len(residuals_scaled)} < GARCH required min {min_len_garch_resid}.")
            return arima_result,None,sf

        # Ensure 1D input for GARCH
        if hasattr(residuals_scaled, 'ndim') and residuals_scaled.ndim != 1:
             log.warning(f"GARCH input (residuals) not 1D (Shape {residuals_scaled.shape}), flattening.")
             residuals_scaled = residuals_scaled.iloc[:, 0] if isinstance(residuals_scaled, pd.DataFrame) and residuals_scaled.shape[1] == 1 else np.asarray(residuals_scaled).flatten()
             residuals_scaled = pd.Series(residuals_scaled, index=getattr(residuals_scaled, 'index', None))
        if not isinstance(residuals_scaled, (pd.Series, np.ndarray)): residuals_scaled = pd.Series(residuals_scaled)

        # Instantiate GARCH model using arch_model from 'arch' library
        # mean='Zero' because when modeling residuals from ARIMA which should have zero mean
        # rescale=False as scaled manually
        if actual_vol_model == 'EGARCH':
            log.debug(f"Fitting EGARCH(p={gp},o={o_asym},q={gq}) dist={distribution} mean=Zero...")
            garch_model=arch_model(residuals_scaled,p=gp,o=o_asym,q=gq, vol='EGARCH', dist=distribution,mean='Zero',rescale=False)
        elif actual_vol_model == 'FIGARCH':
            log.debug(f"Fitting FIGARCH(p={gp},q={gq}) dist={distribution} mean=Zero...")
            # FIGARCH has specific constraints (p<=1, q<=1 in arch implementation?) handled internally
            garch_model=arch_model(residuals_scaled,p=gp, q=gq, vol='FIGARCH', dist=distribution,mean='Zero',rescale=False)
        else: # Standard GARCH, GJR, TARCH etc.
             log.debug(f"Fitting {actual_vol_model}(p={gp},o={o_asym},q={gq}) dist={distribution} mean=Zero...")
             # arch_model uses vol='GARCH' for GJR/TARCH, distinguished by o>0
             vol_param_name = 'GARCH' if actual_vol_model in ['GJR', 'TARCH'] else actual_vol_model
             garch_model=arch_model(residuals_scaled,p=gp,o=o_asym,q=gq,vol=vol_param_name,dist=distribution,mean='Zero',rescale=False)

        try:
            # Fit GARCH model, suppress output, set max iterations
             garch_result=garch_model.fit(disp='off',show_warning=False,options={'maxiter':500})
             log.debug(f"GARCH fit successful: LLF={garch_result.loglikelihood:.2f}")
             # Check convergence flag (0 is success)
             if garch_result.convergence_flag!=0:
                 log.warning(f"GARCH did not converge (Flag:{garch_result.convergence_flag}). Results may be unreliable.")
                 # Keep the result but be aware it might be invalid
        except (ValueError, np.linalg.LinAlgError, ConvergenceWarning, HessianInversionWarning, RuntimeError) as garch_fit_e:
             # Handle common GARCH fit errors
             if "xt contain the same number" in str(garch_fit_e): log.error(f"GARCH fit failed: Input alignment issue: {garch_fit_e}")
             elif "Inputs have different indexes" in str(garch_fit_e): log.error(f"GARCH fit failed: Index mismatch: {garch_fit_e}")
             elif "optimization failed" in str(garch_fit_e).lower(): log.warning(f"GARCH fit failed: Explicit convergence failure: {garch_fit_e}")
             elif "p and q must be either 0 or 1" in str(garch_fit_e) and actual_vol_model == 'FIGARCH': log.warning(f"FIGARCH p/q constraint likely failed: {garch_fit_e}") # Specific FIGARCH warning
             else: log.warning(f"GARCH fit failed: {type(garch_fit_e).__name__}: {garch_fit_e}")
             garch_result = None # Set GARCH result to None on failure
        except Exception as garch_fit_generic_e:
            log.error(f"Unexpected GARCH fit error: {garch_fit_generic_e}")
            garch_result = None

    except (ValueError,TypeError,IndexError) as model_setup_e:
        # Catch errors during model instantiation (e.g., invalid orders, data types)
        if "1-dimensional" in str(model_setup_e):
            log.critical(f"Fit Setup failed: Dimension error: {model_setup_e}")
        else:
            log.warning(f"Model Setup failed: {type(model_setup_e).__name__}: {model_setup_e}")
        arima_result = None
        garch_result = None
    except Exception as e:
        log.error(f"Unexpected error during fitting process: {e}")
        arima_result=None
        garch_result=None

    return arima_result,garch_result,sf

def forecast_arima_garch(arima_res: Any, garch_res: Any, steps: int = 1, sf: float = 100.0) -> Tuple[pd.Series, pd.Series]: # <-- Return Tuple of Series
    """Generates forecasts from fitted ARIMA and GARCH models. Returns mean_return Series and variance Series."""
    MAX_VARIANCE_CLIP = 1.0 # Define the missing constant
    fc_index = pd.RangeIndex(1, steps + 1)
    # Initialize Series directly
    mean_ret_fc = pd.Series(np.nan, index=fc_index, name='mean_return')
    variance_fc = pd.Series(np.nan, index=fc_index, name='variance')

    if arima_res is None:
        log.warning("ARIMA result missing, cannot generate forecast.")
        return mean_ret_fc, variance_fc # Return tuple of NaNs
    if steps <= 0:
        log.error("Forecast steps must be > 0.")
        return mean_ret_fc, variance_fc # Return tuple of NaNs

    try:
        # 1. ARIMA Mean Forecast
        log.debug(f"Generating {steps}-step ARIMA mean forecast...")
        arima_forecast_obj = arima_res.get_forecast(steps=steps)
        mean_scaled = arima_forecast_obj.predicted_mean
        mean_forecast = mean_scaled / sf

        if len(mean_forecast) != steps:
             log.warning(f"ARIMA forecast length ({len(mean_forecast)}) != requested steps ({steps}). Adjusting.")
             mean_forecast_vals = mean_forecast.values[:steps]
             if len(mean_forecast_vals) < steps:
                 mean_forecast_vals = np.pad(mean_forecast_vals, (0, steps - len(mean_forecast_vals)), constant_values=np.nan)
        else:
            mean_forecast_vals = mean_forecast.values
        mean_ret_fc[:] = mean_forecast_vals # Assign values to the Series

        # 2. GARCH Variance Forecast
        variance_forecast_vals = np.full(steps, np.nan) # Use numpy array for calculation
        if garch_res is not None:
             log.debug(f"Generating {steps}-step GARCH variance forecast...")
             try:
                 fc_method = 'simulation' if steps > 1 else 'analytic'
                 garch_forecast_obj = garch_res.forecast(horizon=steps, reindex=False, method=fc_method)
                 var_data = None
                 # Correct way to access ARCH forecast variance: .variance attribute returns a DataFrame
                 if hasattr(garch_forecast_obj, 'variance') and not garch_forecast_obj.variance.empty:
                     var_data = garch_forecast_obj.variance.iloc[0].values # Get first row (forecasts)
                 # Fallback for older versions or different model types
                 elif hasattr(garch_forecast_obj, 'residual_variance') and not garch_forecast_obj.residual_variance.empty:
                     log.warning("Using 'residual_variance' from GARCH forecast - check if this is intended.")
                     var_data = garch_forecast_obj.residual_variance.iloc[0].values
                 else:
                     log.warning(f"No variance data found in GARCH forecast object (method='{fc_method}'). Attributes: {dir(garch_forecast_obj)}")


                 if var_data is not None:
                      variance_scaled = np.asarray(var_data).flatten()[:steps]
                      if len(variance_scaled) < steps: variance_scaled = np.pad(variance_scaled, (0, steps-len(variance_scaled)), constant_values=np.nan)
                      var_scaled_clean = np.where(np.isnan(variance_scaled), np.nan, np.maximum(0, variance_scaled))
                      variance_forecast_raw = var_scaled_clean / (sf**2)
                      variance_forecast_vals = np.clip(variance_forecast_raw, a_min=0, a_max=MAX_VARIANCE_CLIP) # Use defined constant
                      if np.any(variance_forecast_raw > MAX_VARIANCE_CLIP): log.warning(f"Variance clipped at steps {np.where(variance_forecast_raw > MAX_VARIANCE_CLIP)[0]+1}")
                 else: variance_forecast_vals = np.full(steps, np.nan)
             except Exception as ge: log.error(f"GARCH forecast error: {ge}")
        variance_fc[:] = variance_forecast_vals # Assign values to the Series

        m0 = f"{mean_ret_fc.iloc[0]:.6f}" if not mean_ret_fc.empty and pd.notna(mean_ret_fc.iloc[0]) else "NaN"
        v0 = f"{variance_fc.iloc[0]:.8f}" if not variance_fc.empty and pd.notna(variance_fc.iloc[0]) else "NaN"
        log.debug(f"{steps}-step fc generated. Mean[h=1]={m0}, Var[h=1]={v0}")

    except Exception as e:
        log.error(f"Overall forecast generation error: {e}")
        # Return Series of NaNs on error
        return mean_ret_fc.fillna(np.nan), variance_fc.fillna(np.nan)

    return mean_ret_fc, variance_fc # Return tuple of Series

# --- Model Selection / Tuning Helpers ---
def evaluate_candidate_on_val(val_series:pd.Series,arima_order_tune:Tuple[int,int,int],garch_order_tune:Tuple[int,int],garch_type_tune:str,garch_dist_tune:str,sf:float=100.0,tune_criterion:str='BIC')->Tuple[Optional[float],Optional[str]]:
    """Fits a candidate model on validation data and returns its information criterion."""
    p, _, q = arima_order_tune # d is handled externally
    gp, gq = garch_order_tune
    min_len_val = max(sum(arima_order_tune[:1]+arima_order_tune[2:])+10, max(garch_order_tune)+10, 25) # Heuristic min length

    if val_series.empty or len(val_series)< min_len_val:
        return None, f"Validation data too short ({len(val_series)}<{min_len_val})"

    description=f"ARIMA({p},0,{q})+{garch_type_tune}({gp},{gq})({garch_dist_tune})"
    try:
        # Fit the model on the validation series (which should be already differenced if needed)
        ar_val,gr_val,_ = fit_arima_garch(val_series,(p,0,q),(gp,gq),garch_type_tune,garch_dist_tune,sf)

        # Check fit results
        if ar_val is None: return None, "ARIMA Fit Failed on Validation"
        # If GARCH was expected but failed
        if gr_val is None and garch_type_tune.upper() != 'NONE': return None,"GARCH Fit Failed on Validation"
        # Check GARCH convergence if applicable
        if gr_val is not None and hasattr(gr_val,'convergence_flag') and gr_val.convergence_flag!=0:
            return None,f"GARCH Did Not Converge on Validation (Flag:{gr_val.convergence_flag})"

       # Calculate information criterion
        log_likelihood = np.nan
        num_params = 0
        num_obs = 0
        if gr_val is not None: # Use GARCH results if available (preferred for joint likelihood)
            log_likelihood=getattr(gr_val,'loglikelihood',np.nan)
            k_arima=len(getattr(ar_val,'params',[])) # ARIMA params (incl constant)
            k_garch=len(getattr(gr_val,'params',[])) # GARCH params (vol + dist)
            num_params=k_arima+k_garch
            num_obs=getattr(gr_val,'nobs',len(val_series))
        elif ar_val is not None: # Fallback to ARIMA-only stats if no GARCH
             log_likelihood=getattr(ar_val,'llf',np.nan)
             num_params=len(getattr(ar_val,'params',[]))
             num_obs=getattr(ar_val,'nobs',len(val_series))
        else:
             return None, "No model results available for criterion calculation"

        # Validate inputs for criterion calculation
        if not np.isfinite(log_likelihood) or num_params<=0 or num_obs<=num_params:
            reason = f"Invalid criterion inputs (LLF:{log_likelihood}, k:{num_params}, n:{num_obs})"
            log.warning(f"{description}: {reason}")
            return None, reason

        criterion_value=np.inf
        criterion_used=tune_criterion.upper()
        if criterion_used=='BIC':
            criterion_value=-2*log_likelihood+num_params*np.log(num_obs)
        elif criterion_used=='AIC':
            criterion_value=-2*log_likelihood+2*num_params
        else:
            log.warning(f"Unknown tuning criterion '{tune_criterion}', using BIC.")
            criterion_value=-2*log_likelihood+num_params*np.log(num_obs)

        if not np.isfinite(criterion_value):
            return None,"NaN/Inf criterion result"

        # Return criterion value and None for error message
        return float(criterion_value),None

    except ValueError as ve:
        log.warning(f"Evaluation ValueError for {description}: {ve}")
        return None, f"ValueError: {ve}"
    except Exception as e:
        log.error(f"Unexpected evaluation error for {description}: {e}")
        return None,f"Exception: {type(e).__name__}" 

def handle_differencing(series:pd.Series,adf_result:Dict,kpss_result:Dict,manual_d:Optional[int],auto_add_d:Optional[int],max_d:int=2,adf_alpha:float=0.05,kpss_alpha:float=0.05)->Tuple[pd.Series,int,str]:
    """Determines the differencing order 'd' based on tests, manual input, or tuning suggestion."""
    final_d=0
    reason="Initial stationarity tests"
    base_d=0 # d based on initial tests
    add_d=0 # Additional d from tuning

    if manual_d is not None and manual_d >= 0:
        # Manual override
        final_d=max(0,min(manual_d,max_d)) # Apply max_d constraint
        reason=f"Manual d={manual_d}"
        if final_d!=manual_d: reason += f" (limited to {final_d} by max_d={max_d})"
        # Check if manual choice contradicts tests
        adf_stat=adf_result.get('is_stationary', None)
        kpss_stat=kpss_result.get('is_stationary', None)
        if adf_stat is True and kpss_stat is True and final_d > 0:
            log.warning(f"{reason}, but initial tests suggest d=0.")
        elif adf_stat is False and kpss_stat is False and final_d == 0:
            log.warning(f"{reason}, but initial tests suggest d>0.")
        log.info(f"Applying: {reason}")
    else:
        # Automatic differencing based on tests and tuning
        adf_stat=adf_result.get('is_stationary',False)
        kpss_stat=kpss_result.get('is_stationary',False)
        adf_p=adf_result.get('p_value',np.nan)
        kpss_p=kpss_result.get('p_value',np.nan)

        adf_msg = f"ADF Stat={adf_stat}(p={adf_p:.3f})" if pd.notna(adf_p) else f"ADF Stat={adf_stat}(p=N/A)"
        kpss_msg= f"KPSS Stat={kpss_stat}(p={kpss_p:.3f})" if pd.notna(kpss_p) else f"KPSS Stat={kpss_stat}(p=N/A)"

        # Determine base differencing order from tests
        if adf_stat and kpss_stat: # Both suggest stationary
            base_d = 0
            reason=f"Base d=0 ({adf_msg}, {kpss_msg})"
        elif not adf_stat and not kpss_stat: # Both suggest non-stationary
            base_d = 1
            reason=f"Base d=1 ({adf_msg}, {kpss_msg})"
        else: # Tests conflict, often safer to difference once
            base_d = 1
            reason=f"Base d=1 (Conflict: {adf_msg}, {kpss_msg})"
            log.warning(f"ADF/KPSS tests conflict. Defaulting to base d=1.")

        # Add differencing order suggested by auto-tuning (if applicable)
        add_d = auto_add_d if auto_add_d is not None else 0
        if add_d > 0:
            reason += f" + AutoTune add_d={add_d}"

        recommended_d = base_d + add_d
        final_d = max(0, min(recommended_d, max_d)) # Apply max_d constraint

        if final_d != recommended_d:
            reason += f" = {recommended_d} (limited to {final_d} by max_d={max_d})"
        else:
            reason += f" = {final_d}"

        log.info(f"Final differencing order d={final_d} determined by: {reason}")

    diff_reason_final=f"Applied d={final_d} ({reason})"

    # Apply differencing
    if final_d > 0:
        try:
            differenced_series=difference_series(series,order=final_d)
            # Check if differencing resulted in empty series (can happen if order > length)
            if differenced_series.empty and not series.empty:
                 log.error(f"Differencing d={final_d} resulted in empty series. Falling back to d=0.")
                 return series.copy(),0,f"{diff_reason_final} - ERROR: Differencing failed, using d=0."
            # Optional: Check stationarity *after* differencing
            if not differenced_series.empty:
                 log.info(f"Checking stationarity after applying d={final_d}...")
                 adf_post=adf_test(differenced_series,significance=adf_alpha)
                 kpss_post=kpss_test(differenced_series,significance=kpss_alpha)
                 adf_s_post=adf_post.get('is_stationary',False)
                 kpss_s_post=kpss_post.get('is_stationary',False)
                 log.info(f"  ADF after d={final_d}: p={adf_post.get('p_value',np.nan):.4f}, Stationary={adf_s_post}")
                 log.info(f" KPSS after d={final_d}: p={kpss_post.get('p_value',np.nan):.4f}, Stationary={kpss_s_post}")
                 if not adf_s_post or not kpss_s_post:
                     log.warning(f"Series potentially still not stationary after d={final_d} based on tests.")
            return differenced_series, final_d, diff_reason_final
        except Exception as e:
            log.error(f"Error during differencing d={final_d}: {e}. Falling back to d=0.")
            return series.copy(),0,f"{diff_reason_final} - ERROR: Differencing exception, using d=0."
    else:
        # No differencing applied
        return series.copy(), 0, diff_reason_final

def auto_tune_arima_garch(
    train_val_data: pd.Series,
    n_cv_splits: int,
    min_p: int, max_p: int, min_q: int, max_q: int, min_d: int, max_d: int,
    min_gp: int, max_gp: int, min_gq: int, max_gq: int,
    garch_types_list: List[str],
    sf: float = 100.0, tune_criterion: str = 'BIC',
    param_sig_level: float = 0.10, verbose: bool = False
) -> Dict:
    """
    Tunes ARIMA-GARCH models using Time Series Cross-Validation.
    """
    min_len_train_fold = 30
    if len(train_val_data) < min_len_train_fold * (n_cv_splits + 1):
        raise ValueError(f"Data length ({len(train_val_data)}) is too short for {n_cv_splits} CV splits.")

    best_model = {
        'arima': (1, 0, 1), 'add_d_recommended': 0, 'garch_order': (1, 1),
        'garch_type': 'GARCH', 'garch_dist': 't', 'criterion_value': np.inf,
        'criterion_used': f"Avg OOS-{tune_criterion.upper()} ({n_cv_splits}-Fold CV)",
        'stable_and_significant': False, 'error': None
    }

    # --- (Combination generation logic remains the same) ---
    # ...
    combinations = valid_combinations # Assume this is populated correctly
    total_combinations = len(combinations)
    log.info(f"Auto-Tuning ~{total_combinations} models using {n_cv_splits}-Fold Time Series CV...")

    model_avg_scores = {}

    for count, (p, add_d, q, gp, gq, garch_type, g_dist) in enumerate(combinations, 1):
        model_desc = f"ARIMA({p},{add_d},{q})" + (f" + {garch_type}({gp},{gq}) ({g_dist})" if garch_type != 'None' else "")
        if verbose: log.info(f"Tuning {count}/{total_combinations}: {model_desc}")

        try:
            data_diff = difference_series(train_val_data, order=add_d) if add_d > 0 else train_val_data.copy()
            if data_diff.empty: continue
        except Exception: continue

        fold_scores = []
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(data_diff)):
            train_fold = data_diff.iloc[train_idx]
            val_fold = data_diff.iloc[val_idx]

            if len(train_fold) < min_len_train_fold or len(val_fold) < 1: continue

            try:
                ar_fit, gr_fit, sf_fit = fit_arima_garch(train_fold, (p, 0, q), (gp, gq), garch_type, g_dist, sf)
                fit_successful = (ar_fit is not None) and (garch_type == 'None' or (gr_fit is not None and gr_fit.convergence_flag == 0))
                if not fit_successful: continue

                # Parameter significance check on this fold's fit
                # (Full significance check logic from original script goes here)
                # ...
                # if not params_significant: continue

                horizon_val = len(val_fold)
                fc_mean, fc_var = forecast_arima_garch(ar_fit, gr_fit, steps=horizon_val, sf=sf_fit)

                min_len = min(len(val_fold), len(fc_mean), len(fc_var))
                if min_len < 1: continue

                actuals_val = val_fold.values[:min_len]
                means_val = fc_mean.values[:min_len]
                vars_val = fc_var.values[:min_len]

                dist_params = {'nu': gr_fit.params['nu']} if gr_fit and g_dist == 't' and 'nu' in gr_fit.params else {}
                oos_ll_fold = calculate_oos_log_likelihood(actuals_val, means_val, vars_val, g_dist if garch_type != 'None' else 'normal', dist_params)

                if np.isfinite(oos_ll_fold):
                    num_params = len(getattr(ar_fit, 'params', [])) + len(getattr(gr_fit, 'params', []))
                    num_obs_fold = len(actuals_val)
                    if num_obs_fold > num_params:
                        criterion_val = -2 * oos_ll_fold + num_params * np.log(num_obs_fold) if tune_criterion == 'BIC' else -2 * oos_ll_fold + 2 * num_params
                        if np.isfinite(criterion_val):
                            fold_scores.append(criterion_val)

            except Exception:
                continue

        if fold_scores:
            avg_score = np.mean(fold_scores)
            model_avg_scores[(p, add_d, q, gp, gq, garch_type, g_dist)] = avg_score

    if not model_avg_scores:
        log.warning("Auto-Tuning with CV failed: No model could be successfully evaluated.")
        best_model['error'] = "No model succeeded in CV."
        return best_model

    best_params_tuple = min(model_avg_scores, key=model_avg_scores.get)
    p, add_d, q, gp, gq, garch_type, g_dist = best_params_tuple
    
    best_model.update({
        'arima': (p, 0, q), 'add_d_recommended': add_d, 'garch_order': (gp, gq),
        'garch_type': garch_type, 'garch_dist': g_dist,
        'criterion_value': model_avg_scores[best_params_tuple],
        'stable_and_significant': True, 'error': None
    })

    log.info(f"--- Best Model Found via Time Series CV ({best_model['criterion_used']}={best_model['criterion_value']:.2f}) ---")
    log.info(f"  ARIMA (p,q): ({best_model['arima'][0]},{best_model['arima'][2]}), Recommended additional d: {best_model['add_d_recommended']}")
    if best_model['garch_type'] != 'None':
         log.info(f"  GARCH Type: {best_model['garch_type']}, Order (gp,gq): {best_model['garch_order']}, Dist: {best_model['garch_dist']}")

    return best_model

    log.info(f"Auto-Tuning ~{total_combinations} valid combinations (Criterion: {best_model['criterion_used']}, Sig Level: <{param_sig_level})...")

    # Counters for tracking progress/issues
    count = 0; fits_ok = 0; skips_data = 0; skips_fit = 0; skips_conv = 0; skips_sig = 0; skips_crit = 0

    for p, add_d, q, gp, gq, garch_type, g_dist in combinations:
        count += 1
        arima_order_current = (p, 0, q) # Use 0 for d within fit function
        garch_order_current = (gp, gq)
        is_pure_arima = (garch_type == 'None') # Check our marker
        model_desc = f"ARIMA({p},{add_d},{q})" + (f" + {garch_type}({gp},{gq}) ({g_dist})" if not is_pure_arima else " (Pure ARIMA)")

        if verbose or (count % max(1, total_combinations // 1) == 0): # Log progress periodically
            prog = (count / total_combinations * 100) if total_combinations > 0 else 0
            best_v_str = f"{best_model['criterion_value']:.2f}" if best_model['stable_and_significant'] else "inf"
            log.info(f" Tuning: {count}/{total_combinations} ({prog:.1f}%) | Testing: {model_desc} | Best ({best_model['criterion_used']}): {best_v_str}")

        # 1. Apply additional differencing for this candidate
        try:
            series_diff = difference_series(data, order=add_d) if add_d > 0 else data.copy()
            # Check length after differencing
            min_len_req_arima = p + q + 10 if p>0 or q>0 else 0
            min_len_req_garch = max(1, gp, gq) + 10 if not is_pure_arima else 0 # Need >0 for max()
            min_len_req = max(min_len_req_arima, min_len_req_garch, min_len_tune)

            if series_diff.empty or len(series_diff) < min_len_req:
                if verbose: log.debug(f" Skip {model_desc}: Data too short ({len(series_diff)}<{min_len_req}) after add_d={add_d}")
                skips_data += 1; continue
        except Exception as e:
            log.warning(f" Skip {model_desc}: Differencing d={add_d} failed ({e})")
            skips_data += 1; continue

        # 2. Fit the model (ARIMA or ARIMA-GARCH)
        ar = None; gr = None; fit_ok = False
        try:
            if is_pure_arima:
                 # Fit ARIMA only, handle p=0, q=0 case
                 am = ARIMA(series_diff,order=(0,0,0),trend='c') if p==0 and q==0 else ARIMA(series_diff,order=(p,0,q),trend='c')
                 ar=am.fit(); fit_ok = (ar is not None); gr = None
                 if fit_ok and verbose: log.debug(f"Pure ARIMA fit OK: LLF={ar.llf:.2f}")
            else:
                 # Fit combined ARIMA-GARCH
                 ar, gr, _ = fit_arima_garch(series_diff, arima_order_current, garch_order_current, garch_type, g_dist, sf)
                 fit_ok = (ar is not None and gr is not None)
                 # Check GARCH convergence explicitly
                 if fit_ok and hasattr(gr, 'convergence_flag') and gr.convergence_flag != 0:
                      skips_conv += 1
                      if verbose: log.debug(f" Skip {model_desc}: GARCH Did Not Converge (Flag: {gr.convergence_flag})")
                      fit_ok = False # Treat non-convergence as failure for selection

            if not fit_ok:
                skips_fit += 1
                if verbose: log.debug(f" Skip {model_desc}: Fit returned None or GARCH did not converge")
                continue # Skip to next combination if fit failed
            fits_ok += 1 # Count successful fits (incl. converged GARCH)

        except Exception as e:
            skips_fit += 1
            if verbose: log.debug(f" Skip {model_desc}: Fit raised exception ({type(e).__name__})")
            continue

        # 3. Check Parameter Significance (only for successfully fitted models)
        try:
            params_significant = True
            insignificant_list = []
            # Get p-values safely using getattr
            ar_pvals = getattr(ar, 'pvalues', pd.Series(dtype=float))
            gr_pvals = getattr(gr, 'pvalues', pd.Series(dtype=float)) if gr is not None else pd.Series(dtype=float)

            # Check highest lag AR parameter
            if p > 0:
                ar_p_name = f"ar.L{p}"
                # Check if param exists and p-value >= threshold (or is NaN)
                if ar_p_name not in ar_pvals or pd.isna(ar_pvals[ar_p_name]) or ar_pvals[ar_p_name] >= param_sig_level:
                    params_significant = False
                    insignificant_list.append(f"{ar_p_name}(p={ar_pvals.get(ar_p_name, np.nan):.3f})")
            # Check highest lag MA parameter
            if q > 0:
                ma_q_name = f"ma.L{q}"
                if ma_q_name not in ar_pvals or pd.isna(ar_pvals[ma_q_name]) or ar_pvals[ma_q_name] >= param_sig_level:
                    params_significant = False
                    insignificant_list.append(f"{ma_q_name}(p={ar_pvals.get(ma_q_name, np.nan):.3f})")

            # Check sigma2 for pure ARIMA
            if is_pure_arima and 'sigma2' in ar_pvals and ar_pvals['sigma2'] >= param_sig_level:
                 params_significant = False; insignificant_list.append(f"sigma2(p={ar_pvals['sigma2']:.3f})")

            # Check GARCH parameters if not pure ARIMA
            if not is_pure_arima and gr is not None:
                current_garch_type_upper = garch_type.upper()
                # Check highest lag ARCH (alpha) parameter
                if gp > 0:
                    alpha_gp_name = f"alpha[{gp}]"
                    if alpha_gp_name in gr_pvals:
                         if pd.isna(gr_pvals[alpha_gp_name]) or gr_pvals[alpha_gp_name] >= param_sig_level:
                             params_significant = False; insignificant_list.append(f"{alpha_gp_name}(p={gr_pvals[alpha_gp_name]:.3f})")
                    # FIGARCH doesn't always have standard alpha/beta names? Need careful checking based on arch lib.
                    # Assume standard naming for now, except for FIGARCH.
                    elif current_garch_type_upper != 'FIGARCH':
                         params_significant = False; insignificant_list.append(f"{alpha_gp_name}(missing)")

                # Check highest lag GARCH (beta) parameter
                if gq > 0:
                    beta_gq_name = f"beta[{gq}]"
                    if beta_gq_name in gr_pvals:
                         if pd.isna(gr_pvals[beta_gq_name]) or gr_pvals[beta_gq_name] >= param_sig_level:
                             params_significant = False; insignificant_list.append(f"{beta_gq_name}(p={gr_pvals[beta_gq_name]:.3f})")
                    elif current_garch_type_upper != 'FIGARCH':
                         params_significant = False; insignificant_list.append(f"{beta_gq_name}(missing)")

                # Check asymmetry term (o) if expected (gamma, eta, delta...)
                o_asym_tune = 1 if current_garch_type_upper in ['GJR','TARCH','EGARCH','APARCH'] else 0
                if o_asym_tune > 0:
                     asym_param_found = False; asym_param_sig = False
                     for pname in gr_pvals.index:
                         # Common asymmetry parameter names
                         if pname.startswith(('gamma[', 'eta[', 'delta[')): # Check prefixes
                             asym_param_found = True
                             if pd.notna(gr_pvals[pname]) and gr_pvals[pname] < param_sig_level:
                                 asym_param_sig = True; break # Found one significant asym term
                             else:
                                 insignificant_list.append(f"{pname}(p={gr_pvals[pname]:.3f})")
                     # If asymmetry term was expected and found, but none were significant
                     if asym_param_found and not asym_param_sig:
                         params_significant = False

                # Check FIGARCH 'd' parameter specifically
                elif current_garch_type_upper == 'FIGARCH':
                    d_name = 'd'
                    if d_name in gr_pvals:
                         if pd.isna(gr_pvals[d_name]) or gr_pvals[d_name] >= param_sig_level:
                             params_significant = False; insignificant_list.append(f"{d_name}(p={gr_pvals[d_name]:.3f})")
                    else: # 'd' parameter should exist for FIGARCH
                         params_significant = False; insignificant_list.append(f"{d_name}(missing)")

            # If any required parameter is insignificant, skip this model
            if not params_significant:
                skips_sig += 1
                if verbose: log.debug(f" Skip {model_desc}: Insignificant params (p>={param_sig_level}): {', '.join(insignificant_list)}")
                continue

        except Exception as e:
            # Error during significance check
            skips_sig += 1
            if verbose: log.debug(f" Skip {model_desc}: Significance check failed ({type(e).__name__})")
            continue

        # 4. Calculate Information Criterion (only for fitted models with significant params)
        try:
            log_likelihood = np.nan; num_params = 0; num_obs = 0
            if is_pure_arima:
                num_obs = getattr(ar, 'nobs', 0)
                num_params = len(getattr(ar,'params',[]))
                log_likelihood = getattr(ar, 'llf', np.nan)
            else: # Combined model
                num_obs = getattr(gr, 'nobs', 0) # Use GARCH nobs if available
                k_arima = len(getattr(ar,'params',[]))
                k_garch = len(getattr(gr,'params',[]))
                num_params = k_arima + k_garch
                log_likelihood = getattr(gr, 'loglikelihood', np.nan) # Use GARCH LLF

            # Validate inputs for criterion calculation
            if not np.isfinite(log_likelihood) or num_params <= 0 or num_obs <= num_params:
                skips_crit += 1
                if verbose: log.debug(f" Skip {model_desc}: Invalid criterion inputs (LLF:{log_likelihood},k:{num_params},n:{num_obs})")
                continue

            criterion_value = np.inf
            crit = best_model['criterion_used']
            if crit == 'BIC': criterion_value = -2 * log_likelihood + num_params * np.log(num_obs)
            elif crit == 'AIC': criterion_value = -2 * log_likelihood + 2 * num_params
            else: # Should not happen based on config validation, but fallback
                  log.warning(f"Unknown criterion {crit} during tuning, using BIC.")
                  criterion_value = -2 * log_likelihood + num_params * np.log(num_obs)
                  crit = 'BIC'

            if not np.isfinite(criterion_value):
                skips_crit += 1
                if verbose: log.debug(f" Skip {model_desc}: Criterion calculation resulted in NaN/Inf")
                continue

            # 5. Update best model if current one is better
            if criterion_value < best_model['criterion_value']:
                best_model.update({
                    'arima': arima_order_current,
                    'add_d_recommended': add_d,
                    'garch_order': garch_order_current,
                    'garch_type': garch_type if not is_pure_arima else 'None',
                    'garch_dist': g_dist if not is_pure_arima else 'None',
                    'criterion_value': criterion_value,
                    'criterion_used': crit,
                    'stable_and_significant': True, # Mark that we found a valid model
                    'error': None
                })
                log_msg = f"  >> New best significant model found: {model_desc} -> {crit}={criterion_value:.2f}"
                if verbose: log.info(log_msg)
                else:
                     # Log less frequently if not verbose
                     if count % max(1, total_combinations // 5) == 0:
                         log.info(f"  New best significant model found ({crit}={criterion_value:.2f})")

        except Exception as e:
            skips_crit += 1
            if verbose: log.debug(f" Skip {model_desc}: Criterion calculation error ({type(e).__name__})")
            continue

    # --- Tuning finished ---
    log.info(f"Auto-Tuning finished. Evaluated {count}/{total_combinations} combinations. Successful Fits: {fits_ok}")
    log.info(f"  Skips - Data:{skips_data}, FitErr:{skips_fit}, Convergence:{skips_conv}, Insignificant:{skips_sig}, CriterionErr:{skips_crit}")

    if not best_model['stable_and_significant']:
        log.warning(f"Auto-Tuning failed: No model converged with all parameters significant (p<{param_sig_level}). Check ranges or data.")
        best_model['error'] = "No model satisfied criteria"
    else:
        log.info(f"--- Best Model Found (Criterion: {best_model['criterion_used']}={best_model['criterion_value']:.2f}, Sig p<{param_sig_level}) ---")
        log.info(f"  ARIMA (p,q): ({best_model['arima'][0]},{best_model['arima'][2]}), Recommended additional d: {best_model['add_d_recommended']}")
        if best_model['garch_type'] != 'None':
             log.info(f"  GARCH Type: {best_model['garch_type']}, Order (gp,gq): {best_model['garch_order']}, Dist: {best_model['garch_dist']}")
        else:
             log.info("  Pure ARIMA model selected (no GARCH component).")

    # Clean up internal flag before returning
    best_model.pop('stable', None) # Remove deprecated key if exists
    return best_model

def invert_price_forecast(last_actual_price: float, log_return_forecasts: np.ndarray, diff_order: int, recent_actual_log_returns: Optional[List[float]] = None) -> List[Optional[float]]:
    """Converts differenced log return forecasts back to price level forecasts."""
    horizon = len(log_return_forecasts)
    price_forecasts: List[Optional[float]] = [None] * horizon # Initialize with None

    if pd.isna(last_actual_price):
        log.error("Cannot invert price forecast: Last actual price is NaN.")
        return price_forecasts

    log_returns_undifferenced = np.full(horizon, np.nan)

    # Step 1: Undifference the log return forecasts
    if diff_order == 0:
        log_returns_undifferenced = log_return_forecasts.copy()
    elif diff_order > 0:
        if recent_actual_log_returns is None or len(recent_actual_log_returns) < diff_order:
            log.error(f"Cannot undifference d={diff_order}: Need {diff_order} recent actual log returns, only have {len(recent_actual_log_returns) if recent_actual_log_returns else 0}.")
            return price_forecasts # Cannot proceed

        # Use the provided recent actual log returns as history
        history = list(recent_actual_log_returns[-diff_order:])

        for i in range(horizon):
            forecast_diff = log_return_forecasts[i]
            if pd.isna(forecast_diff):
                log.warning(f"NaN found in differenced log return forecast at step {i+1}. Stopping undifferencing.")
                break # Stop if forecast is NaN

            try:
                undiff_val = np.nan
                if diff_order == 1:
                    # X_t = X_{t-1} + diff_forecast_t
                    undiff_val = history[-1] + forecast_diff
                elif diff_order == 2:
                    # diff2_forecast_t = diff1_t - diff1_{t-1}
                    # diff1_t = diff1_{t-1} + diff2_forecast_t
                    # X_t = X_{t-1} + diff1_t
                    if len(history) < 2: # Should not happen if initial check passed, but safety first
                        log.error(f"Undifferencing d=2 failed: Not enough history at step {i+1}")
                        break
                    diff_1_lag1 = history[-1] - history[-2] # Reconstruct diff1_{t-1}
                    undiff_level_1 = diff_1_lag1 + forecast_diff # Calculate diff1_t
                    undiff_val = history[-1] + undiff_level_1 # Calculate X_t
                else:
                    # Extend this logic for higher orders if needed
                    log.error(f"Undifferencing order d={diff_order} not implemented.")
                    break

                # Check for NaN/Inf during calculation
                if not pd.notna(undiff_val) or not np.isfinite(undiff_val):
                    log.warning(f"NaN or Inf result during undifferencing calculation at step {i+1}.")
                    break

                log_returns_undifferenced[i] = undiff_val
                # Update history for next step
                history.append(undiff_val)
                if len(history) > diff_order:
                    history.pop(0) # Keep history length equal to diff_order

            except Exception as e:
                log.error(f"Error during undifferencing calculation at step {i+1}: {e}")
                break # Stop on error

    # Step 2: Convert undifferenced log returns to prices
    last_price = last_actual_price
    for i in range(horizon):
        log_ret = log_returns_undifferenced[i]
        if pd.notna(log_ret) and pd.notna(last_price):
            # Clip log return forecast to prevent extreme price explosion from exp()
            clipped_log_ret = np.clip(log_ret, -5.0, 5.0) # +/- 5 log return is huge
            if clipped_log_ret != log_ret:
                 log.debug(f"Clipped Log-Return forecast {log_ret:.4f} to {clipped_log_ret:.4f} at step {i+1} for price inversion.")

            try:
                 # P_t = P_{t-1} * exp(log_return_t)
                 current_price_forecast = last_price * np.exp(clipped_log_ret)
            except OverflowError:
                 log.warning(f"OverflowError during exp({clipped_log_ret:.2f}) at step {i+1}. Setting price forecast to NaN.")
                 current_price_forecast = np.nan

            # Check if forecast is valid (non-NaN, finite, non-negative)
            if pd.notna(current_price_forecast) and np.isfinite(current_price_forecast) and current_price_forecast >= 0:
                 price_forecasts[i] = current_price_forecast
                 last_price = current_price_forecast # Update last price for next step
            else:
                 log.warning(f"Invalid price forecast ({current_price_forecast}) calculated at step {i+1}. Stopping price inversion.")
                 break # Stop if price becomes invalid
        else:
            log.warning(f"Cannot calculate price at step {i+1} due to missing LogRet ({log_ret}) or previous Price ({last_price}).")
            break # Stop if inputs are missing

    return price_forecasts

# ==============================================================================
# --- MAIN ANALYSIS FUNCTION FOR ONE COIN ---
# ==============================================================================
def run_analysis_for_coin(coin_id: str, config: Dict) -> Dict:
    """Runs the complete ARIMA-GARCH analysis workflow for a single coin."""
    start_time_coin = time.time()
    results = {
        "coin_id": coin_id,
        "status": "OK",
        "error_message": None,
        "benchmarks": {"ewma_var": {}}, # For EWMA results
        "variance_dm": {"backtest": {}, "horizon": {}}, # DM test results for variance
        "multi_horizon_eval": {"metrics":{}, "price_dm_tests":{}, "variance_dm_tests":{}}, # Horizon eval specific
        "roll_fc":{} # Backtest specific
    }
    current_coin_filter.coin_id = coin_id # Set coin ID for logging context
    log.info(f"--- Starting Analysis for {coin_id.upper()} ---")

    # --- Setup Output Directory ---
    coin_plot_dir = os.path.join(config['plot_dir_base'], coin_id)
    is_plot_available = config.get('plot_available', PLOT_AVAILABLE) # Use config flag if present
    ewma_lambda = config['ewma_lambda']
    dm_var_loss = config['dm_test_variance_loss_type']
    qlike_eps = config['qlike_epsilon']

    if is_plot_available and not os.path.exists(coin_plot_dir):
        try:
            os.makedirs(coin_plot_dir, exist_ok=True)
            log.info(f"Output directory created: {coin_plot_dir}")
        except Exception as e:
            log.error(f"Could not create plot directory {coin_plot_dir}: {e}")
            is_plot_available = False # Disable plotting for this coin if dir fails
            log.warning(f"Plotting disabled for {coin_id} due to directory creation error.")
    results["plot_directory"] = coin_plot_dir if is_plot_available else "Directory Creation Failed"

    # Config flags for optional outputs
    gen_tabs = config.get("generate_parameter_tables", True) and is_plot_available
    gen_console = config.get("generate_console_parameter_output", True)
    gen_stab = config.get("generate_stability_plots", True) and is_plot_available and config['forecast_mode'] == 'backtest'
    # Parameters to track for stability plot
    stab_params = ['const','ar.L1','ma.L1','sigma2', # Common ARIMA params
                   'mu','omega', 'rho', 'phi', 'd', # Common GARCH core params (incl FIGARCH 'd')
                   'alpha[1]','q[1]', # Arch lib specific alpha/beta if needed (q[1] for FIGARCH)
                   'beta[1]', # Common GARCH beta
                   'gamma[1]','eta[1]','delta[1]', # Common asymmetry params
                   'nu','lambda'] # Common distribution params (t-dist, skew-t)

    # --- Data Loading and Preprocessing ---
    try:
        log.info(f"[{coin_id}] Fetching data ({config['start_date']} to {config['end_date'] or 'Latest'})...")
        raw_data=fetch_data_yahoo(coin_id, config['start_date'], config['end_date'])
        processed_data=preprocess_data(raw_data)
        if len(processed_data)<config['min_data_length']:
            raise ValueError(f"Data too short after preprocessing ({len(processed_data)} < {config['min_data_length']}).")
        results['total_rows_preprocessed']=len(processed_data)
        log.info(f"[{coin_id}] Data fetched/preprocessed: {len(processed_data)} rows.")

        # Calculate and print descriptive stats for the whole period
        desc_stats=compute_descriptive_stats(processed_data)
        results['descriptive_stats_full']=desc_stats
        print(f"\n[{coin_id.upper()}] --- Descriptive Statistics (Full Period) ---")
        for k,v in desc_stats.items():
             print(f"  {k:<20}: {v:>15.4f}" if isinstance(v,float) else f"  {k:<20}: {str(v):>15}")
        print("-"*40)

    except Exception as e:
        log.critical(f"[{coin_id}] Data Fetching/Preprocessing failed: {type(e).__name__}: {e}", exc_info=True)
        results["status"]="ERROR"
        results["error_message"]=f"Data Fetching/Preprocessing: {type(e).__name__}: {e}"
        current_coin_filter.coin_id="N/A" # Reset logger context
        return results

    # --- Data Splitting ---
    try:
        train_df,val_df,test_df = train_val_test_split(processed_data, config['split_ratios'], config['min_test_set_size'])
        results['train_size']=len(train_df)
        results['val_size']=len(val_df)
        results['test_size']=len(test_df)
        log.info(f"[{coin_id}] Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # Sanity checks for split sizes relative to mode
        if config['forecast_mode']=='backtest' and len(test_df) < config.get('refit_interval', 1):
            log.warning(f"Test set size {len(test_df)} < refit interval {config.get('refit_interval', 1)}. Backtest might end early or behave unexpectedly.")
        valid_horizons_check = [h for h in config.get('evaluation_horizons', []) if isinstance(h, int) and h > 0]
        max_h_check = max(valid_horizons_check) if valid_horizons_check else 0
        if config['forecast_mode']=='horizon_evaluation' and len(test_df) <= (max_h_check + 10): # 10 is buffer for DM test
            log.warning(f"Test set size {len(test_df)} might be small for max horizon {max_h_check} + DM test buffer (10).")
        if config['forecast_mode']=='future' and (len(train_df) + len(val_df)) == 0:
            raise ValueError("Train+Validation sets are empty, cannot fit model for future forecast.")

    except Exception as e:
        log.critical(f"[{coin_id}] Data Splitting failed: {e}", exc_info=True)
        results["status"]="ERROR"
        results["error_message"]=f"Data Splitting: {e}"
        current_coin_filter.coin_id="N/A"
        return results

    # --- Prepare Data for Initial Fit/Tuning (Train + Validation) ---
    # Concatenate train and validation for model selection/initial fit
    fit_base_df = pd.concat([train_df, val_df], ignore_index=True).sort_values('date').reset_index(drop=True)
    if fit_base_df.empty:
        log.critical(f"[{coin_id}] Combined Train+Validation data is empty.")
        results["status"]="ERROR"; results["error_message"]="Train+Validation data empty."
        current_coin_filter.coin_id="N/A"; return results

    # --- Initial Stationarity Check (on Train+Val data) ---
    try:
        log.info(f"[{coin_id}] Performing initial stationarity check (on Train+Val log returns)...")
        adf_init_result={}; kpss_init_result={}
        base_logret = fit_base_df['log_return'].dropna()
        if not base_logret.empty:
            adf_init_result=adf_test(base_logret, config['adf_significance'])
            kpss_init_result=kpss_test(base_logret, config['kpss_significance'], 'c') # 'c' for level stationarity
        else:
            log.warning("Train+Val log returns are empty for stationarity check.")
            adf_init_result={'error':'Empty log returns'}; kpss_init_result={'error':'Empty log returns'}

        if adf_init_result.get('error'): log.error(f"ADF Test failed on initial data: {adf_init_result['error']}")
        if kpss_init_result.get('error'): log.error(f"KPSS Test failed on initial data: {kpss_init_result['error']}")

        results['initial_adf_p_value']=adf_init_result.get('p_value',np.nan)
        results['initial_adf_is_stationary']=adf_init_result.get('is_stationary',False)
        results['initial_kpss_p_value']=kpss_init_result.get('p_value',np.nan)
        results['initial_kpss_is_stationary']=kpss_init_result.get('is_stationary',False)
        log.info(f"Initial Stationarity (Train+Val): ADF Stationary={results['initial_adf_is_stationary']} (p={results['initial_adf_p_value']:.4f}), KPSS Stationary={results['initial_kpss_is_stationary']} (p={results['initial_kpss_p_value']:.4f})")
    except Exception as e:
        log.critical(f"[{coin_id}] Stationarity Check failed: {e}", exc_info=True)
        results["status"]="ERROR"; results["error_message"]=f"Stationarity Check: {e}"
        current_coin_filter.coin_id="N/A"; return results

    # --- Determine Model Structure (Manual or Auto-Tune) ---
    p = 0; q = 0; gp = 0; gq = 0; d_final = 0 # Final orders
    final_dist = config['garch_distribution_manual_choice']
    final_garch_type = config['garch_vol_model_manual']
    tuning_criterion = config['tune_criterion'].upper()
    param_sig_level = config.get('param_significance_level', 0.10)
    final_model_desc = "N/A"
    recommended_add_d = None # Additional d from tuning
    manual_d_input = None # Manual d input from config
    initial_arima_fit = None # Store initial fit on train+val
    initial_garch_fit = None
    results['selection_method'] = 'Manual' # Default
    auto_tune_successful = False
    differenced_series = None # Initialize variable

    try:
        # Use Train+Val log returns for model selection
        selection_data = fit_base_df['log_return'].copy()
        log.info(f"[{coin_id}] Determining model structure using Train+Val data ({len(selection_data)} points)...")

         if config['use_auto_tune']:
            results['selection_method'] = f"Auto-Tune ({config['n_cv_splits']}-Fold TimeSeries CV)"
            log.info(f"[{coin_id}] Starting Auto-Tuning with Time Series Cross-Validation...")
            
            # Kombiniere Train und Val für die CV
            train_val_data = pd.concat([train_df['log_return'], val_df['log_return']]).dropna()
            
            tuning_result = auto_tune_arima_garch(
                train_val_data,
                n_cv_splits=config['n_cv_splits'],
                min_p=config['tune_min_p'], max_p=config['tune_max_p'],
                min_q=config['tune_min_q'], max_q=config['tune_max_q'],
                min_d=config['tune_min_d'], max_d=config['tune_max_d'],
                min_gp=config['tune_min_gp'], max_gp=config['tune_max_gp'],
                min_gq=config['tune_min_gq'], max_gq=config['tune_max_gq'],
                garch_types_list=config.get('garch_types_to_tune', ['GARCH']),
                sf=config['default_scale_factor'],
                tune_criterion=config['tune_criterion'],
                param_sig_level=config['param_significance_level'],
                verbose=config['verbose_tuning']
            )

            if tuning_result.get('error') or not tuning_result.get('stable_and_significant'):
                log.warning(f"[{coin_id}] Auto-Tune failed or found no significant model: {tuning_result.get('error','No suitable model found')}. Falling back to manual configuration.")
                results['selection_method']='Manual (Auto-Tune Failed/Insignificant)'
                p,manual_d_cfg,q = config['manual_arima_order'] # Unpack manual orders
                manual_d_input = manual_d_cfg # Set manual d
                gp,gq = config['manual_garch_order']
                final_garch_type=config['garch_vol_model_manual']
                final_dist=config['garch_distribution_manual_choice']
                recommended_add_d=None
                results['autotune_criterion_value']=np.nan
            else:
                # Use results from tuning
                auto_tune_successful = True
                p,_,q = tuning_result['arima'] # Get tuned p, q (d is separate)
                gp,gq = tuning_result['garch_order']
                final_garch_type = tuning_result['garch_type']
                final_dist = tuning_result['garch_dist']
                recommended_add_d = tuning_result['add_d_recommended'] # Get recommended additional d
                manual_d_input = None # Not using manual d
                tuning_criterion = tuning_result.get('criterion_used', tuning_criterion) # Update criterion if changed by tuning
                results['autotune_criterion_value'] = tuning_result.get('criterion_value')
                log.info(f"Auto-Tune recommended additional differencing d={recommended_add_d}")

        else: # Manual Mode
            p,manual_d_cfg,q = config['manual_arima_order']
            manual_d_input = manual_d_cfg # Set manual d
            gp,gq = config['manual_garch_order']
            final_garch_type=config['garch_vol_model_manual']
            final_dist=config['garch_distribution_manual_choice']
            recommended_add_d=None
            log.info(f"Using Manual Configuration: ARIMA({p},d={manual_d_input},{q}) + {final_garch_type}({gp},{gq}) Dist={final_dist}")

        # --- Determine and Apply Final Differencing Order ---
        # Use the handle_differencing function which considers tests, manual input, and tuning recommendation
        series_to_difference = selection_data # Use train+val data for this decision
        differenced_series, d_final, diff_reason = handle_differencing(
            series_to_difference, adf_init_result, kpss_init_result,
            manual_d_input, recommended_add_d,
            config['max_differencing_order'], config['adf_significance'], config['kpss_significance']
        )
        results["applied_differencing_order"]=d_final
        results["recommended_add_d"]=recommended_add_d if config['use_auto_tune'] and auto_tune_successful else 'N/A'
        results["differencing_reason"]=diff_reason

        min_len_after_diff = 30 # Need sufficient data after differencing to fit models
        if differenced_series.empty or len(differenced_series)<min_len_after_diff:
            raise ValueError(f"Differenced series (d={d_final}) is unusable ({len(differenced_series)}<{min_len_after_diff}). Check data or differencing order.")
        log.info(f"Final differencing order set to d={d_final}. Reason: {diff_reason}")


        # --- Compare GARCH Distributions (if not auto-tuned and GARCH model is used) ---
        skip_distribution_comparison = (config['use_auto_tune'] and auto_tune_successful) or (final_garch_type == 'None')
        if config['compare_garch_dists'] and scipy_stats_available and not skip_distribution_comparison:
            crit_dist_comp = tuning_criterion if tuning_criterion in ['AIC','BIC'] else 'AIC' # Use AIC/BIC for comparison
            log.info(f"[{coin_id}] Comparing GARCH Distributions for {final_garch_type}({gp},{gq}) using {crit_dist_comp} (on Train+Val differenced d={d_final} data)...")
            best_criterion_dist=np.inf
            best_dist_found=final_dist # Start with the current choice
            dist_options_comp=['normal','t']
            dist_results_values={}

            try:
                # Need ARIMA residuals first. Fit ARIMA part once.
                temp_arima_order_comp = (p, 0, q)
                arima_temp_comp_res, _, _ = fit_arima_garch(differenced_series, temp_arima_order_comp, (1,1), 'GARCH', 'normal', sf=config['default_scale_factor']) # GARCH part is dummy here
                if arima_temp_comp_res is None: raise ValueError("Temporary ARIMA fit failed for distribution comparison.")

                temp_residuals = arima_temp_comp_res.resid.copy().dropna()
                if temp_residuals.empty or not np.all(np.isfinite(temp_residuals)): raise ValueError("Invalid residuals obtained for distribution comparison.")
                # Ensure 1D
                if hasattr(temp_residuals, 'ndim') and temp_residuals.ndim != 1: temp_residuals = temp_residuals.iloc[:, 0] if isinstance(temp_residuals, pd.DataFrame) else temp_residuals.flatten()
                if not isinstance(temp_residuals, (pd.Series, np.ndarray)): temp_residuals = pd.Series(temp_residuals)

                # Now fit GARCH with different distributions on these residuals
                temp_garch_order_comp = (gp, gq)
                temp_garch_type_comp = final_garch_type
                o_asym_temp_comp = 1 if temp_garch_type_comp.upper() in ['GJR','TARCH','EGARCH','APARCH'] else 0

                # Determine vol model name for arch_model call
                if temp_garch_type_comp.upper() == 'FIGARCH': vol_model_temp_comp = 'FIGARCH'; extra_args_comp = {}
                elif temp_garch_type_comp.upper() == 'EGARCH': vol_model_temp_comp = 'EGARCH'; extra_args_comp = {}
                elif temp_garch_type_comp.upper() in ['GJR', 'TARCH']: vol_model_temp_comp = 'GARCH'; extra_args_comp = {} # Handled by 'o' param
                else: vol_model_temp_comp = temp_garch_type_comp.upper(); extra_args_comp = {}

                for dist_option in dist_options_comp:
                     try:
                         gm_comp=arch_model(temp_residuals, p=gp, o=o_asym_temp_comp, q=gq, vol=vol_model_temp_comp, dist=dist_option, mean='Zero', rescale=False, **extra_args_comp)
                         gr_comp=gm_comp.fit(disp='off',show_warning=False,options={'maxiter':500})
                         if gr_comp.convergence_flag!=0: raise RuntimeError(f"GARCH dist='{dist_option}' did not converge.")
                         # Get the criterion value
                         criterion_val=getattr(gr_comp,crit_dist_comp.lower(),np.inf)
                         if not np.isfinite(criterion_val): raise ValueError(f"NaN criterion value for dist='{dist_option}'.")

                         dist_results_values[dist_option]=criterion_val
                         log.info(f"  Distribution '{dist_option}' -> {crit_dist_comp}: {criterion_val:.2f}")
                         if criterion_val<best_criterion_dist:
                             best_criterion_dist=criterion_val
                             best_dist_found=dist_option
                     except Exception as de:
                         log.warning(f" Fitting/Evaluation failed for dist='{dist_option}': {de}")
                         dist_results_values[dist_option]=np.inf # Assign inf on failure

                # Update final distribution if comparison found a better one
                if best_criterion_dist != np.inf and final_dist != best_dist_found:
                    log.info(f"Distribution Comparison: Best distribution changed from '{final_dist}' to '{best_dist_found}' ({crit_dist_comp}={best_criterion_dist:.2f}).")
                    final_dist = best_dist_found
                elif best_criterion_dist != np.inf:
                     log.info(f"Distribution Comparison: Best distribution remains '{final_dist}' ({crit_dist_comp}={best_criterion_dist:.2f}).")
                else:
                     log.warning(f"Distribution Comparison failed to find a valid best distribution. Keeping '{final_dist}'.")
                results["distribution_comparison"]={"results":dist_results_values,"best_dist":final_dist,"criterion_used":crit_dist_comp,"skipped":False}

            except Exception as comp_e:
                log.error(f"Distribution comparison process failed: {comp_e}. Keeping original distribution '{final_dist}'.")
                results["distribution_comparison"]={"error":str(comp_e),"best_dist":final_dist,"skipped":False}

        elif skip_distribution_comparison:
            reason_skip = "Auto-Tune OK" if (config['use_auto_tune'] and auto_tune_successful) else "Pure ARIMA model"
            log.info(f"Skipping Distribution Comparison: {reason_skip}. Using distribution '{final_dist}'.")
            results["distribution_comparison"]={"skipped":True,"reason":reason_skip}
        elif not scipy_stats_available:
             log.warning("Skipping Distribution Comparison: SciPy not available for extended distributions.")
             results["distribution_comparison"]={"skipped":True,"reason":"Scipy not available"}
        else: # compare_garch_dists is False
             log.info(f"Skipping Distribution Comparison as per config. Using distribution '{final_dist}'.")
             results["distribution_comparison"]={"skipped":True,"reason":"Config disabled"}

        # --- Final Model Specification ---
        chosen_garch_dist_final = final_dist if final_garch_type != 'None' else 'N/A'
        chosen_garch_type_final = final_garch_type if final_garch_type != 'None' else 'N/A'
        chosen_gp_final = gp if final_garch_type != 'None' else 0
        chosen_gq_final = gq if final_garch_type != 'None' else 0

        # Store final orders including the applied differencing order
        arima_order_final = (p, d_final, q)
        garch_order_fit_final = (chosen_gp_final, chosen_gq_final) # (gp, gq) used for fitting

        # Create description string
        final_model_desc = f"ARIMA({p},{d_final},{q})"
        if chosen_garch_type_final != 'N/A':
            final_model_desc += f" - {chosen_garch_type_final}({chosen_gp_final},{chosen_gq_final}) - Dist='{chosen_garch_dist_final}'"

        # Store final model parameters in results
        results["final_model_description"]=final_model_desc
        results["arima_order_p"]=p; results["arima_order_q"]=q; results["applied_differencing_order"]=d_final
        results["garch_order_p"]=chosen_gp_final; results["garch_order_q"]=chosen_gq_final
        results['garch_type']=chosen_garch_type_final
        results['garch_distribution']=chosen_garch_dist_final
        results['tuning_criterion']=tuning_criterion
        results['param_significance_level']=param_sig_level
        log.info(f"--- Final Model Structure ({results['selection_method']}): {final_model_desc} ---")

        # --- Initial Fit of Final Model on Train+Val (for reporting/diagnostics) ---
        log.info(f"Performing initial fit of final model ({final_model_desc}) on Train+Val data (using differenced d={d_final})...")
        try:
             if not differenced_series.empty:
                 if chosen_garch_type_final == 'N/A': # Pure ARIMA
                      log.debug("Fitting pure ARIMA model for initial report...")
                      am_init=ARIMA(differenced_series,order=(p,0,q),trend='c')
                      initial_arima_fit = am_init.fit()
                      initial_garch_fit = None
                 else: # ARIMA-GARCH
                      initial_arima_fit, initial_garch_fit, _ = fit_arima_garch(
                           differenced_series,
                           (p,0,q), # Use p,0,q for internal ARIMA fit
                           garch_order_fit_final,
                           chosen_garch_type_final,
                           chosen_garch_dist_final,
                           sf=config['default_scale_factor']
                      )
             else:
                  log.warning("Skipping initial fit: differenced series is empty.")
                  initial_arima_fit=None; initial_garch_fit=None

             if initial_arima_fit is None and chosen_garch_type_final != 'N/A':
                 log.warning("Initial fit of combined ARIMA-GARCH model failed.")
             elif initial_arima_fit is None and chosen_garch_type_final == 'N/A':
                 log.warning("Initial fit of pure ARIMA model failed.")
             else:
                 log.info("Initial fit successful.")
                 
                 if initial_arima_fit:
                     results['arima_initial_aic'] = getattr(initial_arima_fit, 'aic', np.nan)
                     results['arima_initial_bic'] = getattr(initial_arima_fit, 'bic', np.nan)
                     results['arima_initial_llf'] = getattr(initial_arima_fit, 'llf', np.nan)
                 if initial_garch_fit:
                     results['garch_initial_aic'] = getattr(initial_garch_fit, 'aic', np.nan)
                     results['garch_initial_bic'] = getattr(initial_garch_fit, 'bic', np.nan)
                     results['garch_initial_llf'] = getattr(initial_garch_fit, 'loglikelihood', np.nan)

        except Exception as init_fit_e:
            log.error(f"Initial fit of the final model failed: {init_fit_e}")
            initial_arima_fit=None; initial_garch_fit=None

    except Exception as model_select_e:
        log.critical(f"Model selection/setup phase failed: {model_select_e}",exc_info=True)
        results["status"]="ERROR"
        results["error_message"]=f"Model Selection/Setup: {model_select_e}"
        current_coin_filter.coin_id="N/A"; return results

    # --- Generate Initial Fit Outputs (Tables, Console, Plots) ---
    if gen_tabs and (initial_arima_fit is not None or initial_garch_fit is not None):
        table_filename=os.path.join(coin_plot_dir, f"{coin_id}_params_initial_fit_d{d_final}.png")
        table_title=f"{coin_id.upper()} Initial Parameters (Train+Val Fit)\n({final_model_desc})"
        try:
            table_created=create_parameter_table_png(initial_arima_fit, initial_garch_fit, table_filename, table_title)
            results["initial_parameter_table_path"]=table_filename if table_created else "Failed"
        except Exception as table_e:
            log.error(f"Error generating initial parameter PNG: {table_e}")
            results["initial_parameter_table_path"]=f"Error: {table_e}"
    elif gen_tabs:
        log.warning("Skipping initial parameter PNG: Initial fit failed.")
        results["initial_parameter_table_path"]="Skipped (Fit Failed)"

    if gen_console:
        try:
            print_model_summary_console(initial_arima_fit, initial_garch_fit, f"{coin_id} (Initial Fit on Train+Val)")
        except Exception as e:
            log.error(f"Console output for initial fit failed: {e}")

    # Initial Diagnostic Plots (ACF/PACF on differenced Train+Val, QQ on original Train+Val LogRet)
    if is_plot_available and plt is not None and sm is not None and differenced_series is not None:
        log.info(f"Generating initial diagnostic plots using Train+Val data (d={d_final})...")
        # Use the differenced series obtained during model selection for ACF/PACF
        plot_series_input=differenced_series
        if plot_series_input is not None and not plot_series_input.empty:
             plot_description=f"Train+Val (LogRet, Differenced d={d_final})"
             acf_filename=os.path.join(coin_plot_dir,f"{coin_id}_acf_init_d{d_final}.png")
             pacf_filename=os.path.join(coin_plot_dir,f"{coin_id}_pacf_init_d{d_final}.png")
             # Determine appropriate lags (e.g., 40 or N/2)
             lags_plot=min(40,max(1,len(plot_series_input)//2-1)) if len(plot_series_input)>3 else 0

             try: # ACF Plot
                 fig,ax=plt.subplots(figsize=(10,4))
                 if lags_plot>0:
                     plot_acf(plot_series_input,lags=lags_plot,ax=ax,title=f'ACF ({coin_id}) - {plot_description}',zero=False,color="black") # Use dynamic lags
                 else: ax.text(0.5,0.5,'Not enough data for ACF',ha='center',va='center')
                 fig.tight_layout(); fig.savefig(acf_filename); plt.close(fig)
                 log.info(f"ACF plot saved: {acf_filename}")
             except Exception as e: log.error(f"ACF plot generation failed: {e}"); plt.close('all') # Close any open figures

             try: # PACF Plot
                 fig,ax=plt.subplots(figsize=(10,4))
                 if lags_plot>0:
                      # method='ywm' is often preferred for PACF
                     plot_pacf(plot_series_input,lags=lags_plot,method='ywm',ax=ax,title=f'PACF ({coin_id}) - {plot_description}',zero=False,color="black") # Use dynamic lags
                 else: ax.text(0.5,0.5,'Not enough data for PACF',ha='center',va='center')
                 fig.tight_layout(); fig.savefig(pacf_filename); plt.close(fig)
                 log.info(f"PACF plot saved: {pacf_filename}")
             except Exception as e: log.error(f"PACF plot generation failed: {e}"); plt.close('all')
        else:
             log.warning("Skipping ACF/PACF plots: Input series (differenced Train+Val) is empty.")

        # Q-Q Plot of original log returns (Train+Val) vs Normal distribution
        qq_series=fit_base_df['log_return'].dropna()
        qq_filename=os.path.join(coin_plot_dir,f"{coin_id}_qq_init_logret.png")
        if not qq_series.empty:
             try:
                 fig,ax=plt.subplots(figsize=(6,5))
                 sm.qqplot(qq_series,line='s',fit=True,ax=ax,markerfacecolor='black', markeredgecolor='black', markersize=4) # 's' standardized line, 'fit=True' uses sample moments
                 ax.set_title(f'Q-Q Plot (Train+Val Original LogRet vs Normal)\n({coin_id})')
                 fig.tight_layout(); fig.savefig(qq_filename); plt.close(fig)
                 log.info(f"Q-Q plot saved: {qq_filename}")
             except Exception as e: log.error(f"Q-Q plot generation failed: {e}"); plt.close('all')
        else:
            log.warning("Skipping Q-Q plot: Original Train+Val Log Returns are empty.")
    elif is_plot_available and differenced_series is None:
         log.warning("Skipping initial diagnostic plots as differenced series was not generated.")
         
        
# ==========================================================================
    # --- FORECASTING & EVALUATION BLOCK ---
    # ==========================================================================
    forecast_mode = config['forecast_mode']

    # --- Mode 1: Rolling 1-Step Forecast ('backtest') ---
    if forecast_mode == 'backtest':
        # BACKTEST LOGIC 
        log.info(f"--- [{coin_id}] Starting Rolling 1-Step Forecast (Backtest) ---")
        log.info(f" Fitting Window: {config['fitting_window_size']} days, Refit Interval: {config['refit_interval']} days")
        log.info(f" EWMA Lambda (Benchmark): {ewma_lambda}")
        log.info(f" Variance DM Test Loss: {dm_var_loss}")
        alpha = config.get("dm_test_alpha", 0.05) # Use same alpha for VaR/ES/Tests
        log.info(f" VaR/ES/Backtest Alpha: {alpha:.2%}")
        t_roll_start=time.time()
        results["roll_fc"]={} # Initialize dict for backtest results

        # Use full processed data, indexed by date
        full_data_indexed = processed_data.set_index('date').sort_index()
        test_start_date = test_df['date'].iloc[0] if not test_df.empty else None
        if test_start_date is None:
            raise ValueError("Test Set is empty, cannot perform backtest.")

        # Find the integer index corresponding to the start of the test set
        try:
            test_start_loc = full_data_indexed.index.get_loc(test_start_date)
            test_start_idx = test_start_loc.start if isinstance(test_start_loc, slice) else test_start_loc
        except KeyError:
            # Fallback if exact date not found (e.g., due to resampling) - find next available date
            try:
                test_start_idx = full_data_indexed.index.get_indexer([test_start_date], method='bfill')[0]
            except IndexError:
                raise ValueError(f"Test start date {test_start_date} could not be located in the indexed data.")
        if test_start_idx < 0: # get_indexer returns -1 if no date found
            raise ValueError("Invalid test start index found.")

        n_total=len(full_data_indexed)
        n_rolling_steps=n_total-test_start_idx # Number of steps to forecast in the test set

        if n_rolling_steps<=0:
            raise ValueError("No steps available in the test set for rolling forecast.")

        log.info(f"Rolling forecast over {n_rolling_steps} steps in the test set, starting from {full_data_indexed.index[test_start_idx].date()}")
        log.info(f" Using model: {final_model_desc}")

        # Lists to store results for each step
        forecast_dates_list=[]
        actual_prices_list=[]
        forecast_prices_ag_list=[] # AG = ARIMA-GARCH
        forecast_prices_naive_list=[]
        actual_log_returns_list=[]
        forecast_log_returns_ag_list=[]
        forecast_log_returns_naive_list = []
        forecast_volatility_ag_list=[]
        forecast_variance_ag_list=[]
        forecast_variance_ewma_list=[]
        actual_variance_proxies_list=[] # Using squared log return as proxy
        # --- Lists for VaR/ES/Violations ---
        var_thresh_ag_list = []
        es_ag_list = []
        violation_ag_list = []
        var_thresh_ewma_list = []
        es_ewma_list = []
        violation_ewma_list = []

        # For parameter stability plot
        stability_param_data = {par:[] for par in stab_params if gen_stab}
        stability_dates_list = []

        # Cache last fitted models to avoid refitting every step if interval > 1
        last_fit_arima_cached = initial_arima_fit # **FIX:** Starte mit dem initialen Fit als erstem Cache
        last_fit_garch_cached = initial_garch_fit
        last_fit_step_idx = -1 # Zeigt an, dass der letzte Fit der initiale war

        # State for EWMA calculation
        last_ewma_variance_state = np.nan

        # Counters
        skipped_steps_count = 0
        fit_errors_count = 0
        # --- Define min length for diagnostics/backtests ---
        min_resid_len_diag = 25 # Minimum length for reliable Ljung-Box/ARCH/VaR tests

        # --- Rolling Forecast Loop ---
        for i in range(n_rolling_steps):
            current_step_test_idx = test_start_idx + i # Index in the full dataset
            current_date = full_data_indexed.index[current_step_test_idx]
            step_number = i + 1

            # Log progress periodically
            log_freq=max(1,n_rolling_steps//10)
            if step_number==1 or step_number%log_freq==0 or step_number==n_rolling_steps:
                log.info(f"Rolling Backtest: Step {step_number}/{n_rolling_steps} (Forecasting for {current_date.date()})...")

            # Determine if model needs refitting
            should_refit = (i % config['refit_interval'] == 0)

            # Define the window for fitting the model
            # Fit window ends *before* the current step we want to forecast
            fit_window_end_idx = current_step_test_idx # Data up to t-1
            fit_window_start_idx = max(0, fit_window_end_idx - config['fitting_window_size'])
            fitting_window_data_df = full_data_indexed.iloc[fit_window_start_idx:fit_window_end_idx]

            # Get actual values for the current step (t)
            actual_price_t = full_data_indexed.iloc[current_step_test_idx]['price']
            actual_log_return_t = full_data_indexed.iloc[current_step_test_idx]['log_return']
            actual_variance_proxy_t = actual_log_return_t**2 if pd.notna(actual_log_return_t) else np.nan

            # Get naive forecasts (price = last actual price, log return = 0)
            last_actual_price_t_minus_1 = full_data_indexed.iloc[current_step_test_idx - 1]['price'] if current_step_test_idx > 0 else np.nan
            forecast_price_naive_t = last_actual_price_t_minus_1
            forecast_log_return_naive_t = 0.0

            # Initialize forecasts for this step
            fc_price_ag_t = np.nan
            fc_log_return_ag_t = np.nan # This will hold the *undifferenced* log return forecast
            fc_volatility_ag_t = np.nan
            fc_variance_ag_t = np.nan
            fc_variance_ewma_t = np.nan
            fc_volatility_ewma_t = np.nan # Initialize EWMA volatility
            arima_params_current_step = {}
            garch_params_current_step = {}
            # --- Initialize VaR/ES/Violation for this step ---
            var_thresh_ag_t = np.nan
            es_ag_t = np.nan
            violation_ag_t = np.nan # Use np.nan to indicate not calculable vs. 0 (no violation)
            var_thresh_ewma_t = np.nan
            es_ewma_t = np.nan
            violation_ewma_t = np.nan
            dist_params_model_t = None # Store distribution parameters for this step if refit

            append_nan_ag = False # Flag to append NaN if AG model fails
            append_nan_ewma = False # Flag to append NaN if EWMA fails

            try:
                # --- Fit/Load ARIMA-GARCH Model ---
                if should_refit:
                    log.debug(f" Step {step_number}: Refitting model...")
                    # Check if fitting window is large enough
                    if len(fitting_window_data_df) < config['min_fitting_window_size']:
                         log.warning(f" Step {step_number}: Skipping AG Fit/Forecast. Window size ({len(fitting_window_data_df)}) < Minimum ({config['min_fitting_window_size']}).")
                         skipped_steps_count += 1; append_nan_ag = True
                         last_fit_arima_cached = None; last_fit_garch_cached = None # Invalidate cache
                    else:
                        fit_log_returns_series = fitting_window_data_df['log_return'].copy()
                        # Apply differencing to the window data
                        fit_series_differenced = difference_series(fit_log_returns_series, order=d_final) if d_final > 0 else fit_log_returns_series.copy()
                        fit_series_differenced = fit_series_differenced.dropna() # Drop NaNs from differencing

                        # Check length *after* differencing
                        min_len_fit_diff_arima = p + q + 5 if p > 0 or q > 0 else 0
                        min_len_fit_diff_garch = chosen_gp_final + chosen_gq_final + 5 if chosen_garch_type_final != 'N/A' else 0
                        min_len_fit_diff = max(min_len_fit_diff_arima, min_len_fit_diff_garch, 15) # Min length required for fit

                        if fit_series_differenced.empty or len(fit_series_differenced) < min_len_fit_diff:
                            log.warning(f" Step {step_number}: Skipping AG Fit/Forecast. Differenced series too short ({len(fit_series_differenced)}<{min_len_fit_diff}) for d={d_final}.")
                            skipped_steps_count += 1; append_nan_ag = True
                            last_fit_arima_cached = None; last_fit_garch_cached = None # Invalidate cache
                        else:
                            # Fit the final chosen model structure
                            arima_order_for_fit = (p, 0, q) # d=0 as series is already differenced
                            try:
                                if chosen_garch_type_final == 'N/A': # Pure ARIMA
                                    log.debug(f"Step {step_number}: Fitting pure ARIMA({p},{d_final},{q})...")
                                    arima_model_step = ARIMA(fit_series_differenced, order=arima_order_for_fit, trend='c')
                                    arima_result_step = arima_model_step.fit(method_kwargs={"xtol": 1e-6, "ftol": 1e-6, "maxiter": 1000})
                                    garch_result_step = None
                                else: # Combined ARIMA-GARCH
                                    arima_result_step, garch_result_step, scale_factor_step = fit_arima_garch(
                                         fit_series_differenced,
                                         arima_order_for_fit,
                                         garch_order_fit_final,
                                         chosen_garch_type_final,
                                         chosen_garch_dist_final,
                                         sf=config['default_scale_factor']
                                    )

                                # Check if fit was successful
                                fit_successful_flag = False
                                if chosen_garch_type_final == 'N/A' and arima_result_step is not None:
                                    fit_successful_flag = True
                                elif chosen_garch_type_final != 'N/A' and arima_result_step is not None and garch_result_step is not None:
                                     # Check GARCH convergence
                                     if hasattr(garch_result_step,'convergence_flag') and garch_result_step.convergence_flag != 0:
                                         log.warning(f" Step {step_number}: GARCH did not converge during refit (Flag: {garch_result_step.convergence_flag}). Treating as fit failure.")
                                         # Fit failed even if objects exist
                                     else:
                                         fit_successful_flag = True

                                if fit_successful_flag:
                                     log.debug(f" Step {step_number}: Refit successful.")
                                     last_fit_arima_cached = arima_result_step
                                     last_fit_garch_cached = garch_result_step
                                     last_fit_step_idx = i # Update index of last successful fit
                                     # --- Extract distribution parameters if GARCH fit ---
                                     if garch_result_step is not None:
                                          params_garch_t = getattr(garch_result_step, 'params', {})
                                          if chosen_garch_dist_final == 't' and 'nu' in params_garch_t:
                                               dist_params_model_t = {'nu': params_garch_t['nu']}
                                          elif chosen_garch_dist_final == 'skewt' and 'nu' in params_garch_t and 'lambda' in params_garch_t:
                                               dist_params_model_t = {'nu': params_garch_t['nu'], 'lambda': params_garch_t['lambda']}
                                          else: dist_params_model_t = None
                                     else: dist_params_model_t = None
                                     
                                else:
                                     log.warning(f" Step {step_number}: Model refit failed.")
                                     fit_errors_count += 1; append_nan_ag = True
                                     last_fit_arima_cached = None; last_fit_garch_cached = None # Invalidate cache
                                     dist_params_model_t = None # No params if fit failed

                            except Exception as fit_exception:
                                 log.error(f" Step {step_number}: Error during model refitting: {fit_exception}")
                                 fit_errors_count += 1; append_nan_ag = True
                                 last_fit_arima_cached = None; last_fit_garch_cached = None # Invalidate cache
                                 dist_params_model_t = None
                else:
                     # Use cached model if not refitting
                     log.debug(f" Step {step_number}: Using cached model from step {last_fit_step_idx+1}")
                     arima_result_step = last_fit_arima_cached
                     garch_result_step = last_fit_garch_cached
                     scale_factor_step = config['default_scale_factor'] # Assume default scale if using cache
                     # --- Get dist params from cached GARCH fit ---
                     if last_fit_garch_cached is not None:
                          params_garch_t = getattr(last_fit_garch_cached, 'params', {})
                          if chosen_garch_dist_final == 't' and 'nu' in params_garch_t:
                               dist_params_model_t = {'nu': params_garch_t['nu']}
                          elif chosen_garch_dist_final == 'skewt' and 'nu' in params_garch_t and 'lambda' in params_garch_t:
                               dist_params_model_t = {'nu': params_garch_t['nu'], 'lambda': params_garch_t['lambda']}
                          else: dist_params_model_t = None
                     else: dist_params_model_t = None
                     

                # --- Generate 1-step Forecast using fitted/cached model ---
                if not append_nan_ag and arima_result_step is not None:
                     # Get recent actual log returns needed for undifferencing
                     recent_log_returns_history = None
                     if d_final > 0:
                          history_end_idx = current_step_test_idx # Use data up to t-1
                          history_start_idx = max(0, history_end_idx - d_final)
                          if history_end_idx - history_start_idx == d_final:
                              recent_log_returns_history = full_data_indexed['log_return'].iloc[history_start_idx : history_end_idx].tolist()
                          else:
                              log.warning(f"Step {step_number}: Not enough history ({current_step_test_idx} < {d_final}) for undifferencing d={d_final}. Log return forecast will be NaN.")

                     # Generate 1-step forecast (returns mean and variance)
                     fc_mean_series_ag_t, fc_variance_series_ag_t = forecast_arima_garch(
                         arima_result_step, garch_result_step, steps=1, sf=scale_factor_step
                     )
                     fc_log_return_diff_1 = fc_mean_series_ag_t.iloc[0] # Forecast of differenced log return
                     fc_variance_1 = fc_variance_series_ag_t.iloc[0]
                     fc_volatility_1 = np.sqrt(max(0, fc_variance_1)) if pd.notna(fc_variance_1) else np.nan

                     # Validate variance forecast
                     if not np.isfinite(fc_variance_1):
                          log.error(f"Step {step_number}: Non-finite AG variance forecast ({fc_variance_1}) received!")
                          append_nan_ag = True # Treat as failure
                     elif fc_variance_1 < 0:
                          log.warning(f"Step {step_number}: Negative AG variance forecast ({fc_variance_1:.6f}), clamping to 0.")
                          fc_variance_1 = 0.0
                          fc_volatility_1 = 0.0
                     elif fc_variance_1 > 0.5: # Arbitrary threshold for very high daily variance
                          log.warning(f"Step {step_number}: Very high AG variance forecast ({fc_variance_1:.6f}).")

                     if not append_nan_ag:
                         # Undifference log return forecast to get LEVEL log return forecast
                         if pd.notna(fc_log_return_diff_1):
                             if d_final == 0:
                                 fc_log_return_ag_t = fc_log_return_diff_1 # Already level forecast
                             elif d_final == 1 and recent_log_returns_history is not None:
                                 fc_log_return_ag_t = recent_log_returns_history[-1] + fc_log_return_diff_1
                             elif d_final == 2 and recent_log_returns_history is not None and len(recent_log_returns_history)==2:
                                 diff_1_lag1 = recent_log_returns_history[-1] - recent_log_returns_history[-2]
                                 undiff_level_1 = diff_1_lag1 + fc_log_return_diff_1
                                 fc_log_return_ag_t = recent_log_returns_history[-1] + undiff_level_1
                             else:
                                 # Handles missing history or unimplemented d order
                                 fc_log_return_ag_t = np.nan # Set undifferenced forecast to NaN
                                 log.warning(f"Step {step_number}: Could not undifference AG log return forecast for d={d_final}.")
                         else:
                              fc_log_return_ag_t = np.nan # If differenced forecast is NaN, level is NaN

                         # Convert log return forecast to price forecast
                         if pd.notna(fc_log_return_ag_t) and pd.notna(last_actual_price_t_minus_1):
                              try:
                                   fc_price_ag_t = last_actual_price_t_minus_1 * np.exp(np.clip(fc_log_return_ag_t, -5, 5)) # Clip logret before exp
                              except OverflowError:
                                   log.warning(f"OverflowError during exp() for AG price forecast step {step_number}.")
                                   fc_price_ag_t = np.nan
                         else:
                              fc_price_ag_t = np.nan

                         # Assign variance/volatility
                         fc_variance_ag_t = fc_variance_1
                         fc_volatility_ag_t = fc_volatility_1 # Already calculated from potentially clamped variance

                         # Store parameters if refit and stability plot enabled
                         if gen_stab and should_refit:
                             if arima_result_step is not None: arima_params_current_step = getattr(arima_result_step,'params',pd.Series(dtype=float)).to_dict()
                             if garch_result_step is not None: garch_params_current_step = getattr(garch_result_step,'params',pd.Series(dtype=float)).to_dict()

                else:
                     # Case where AG model fit failed or was skipped
                     if not append_nan_ag: # Only log if not already flagged
                        log.warning(f" Step {step_number}: No valid AG model available for forecast. Appending NaNs.")
                        append_nan_ag = True

                # --- Calculate EWMA Variance Forecast ---
                try:
                    if i == 0: # Initialize EWMA state on the first step
                        # Use variance of log returns in the initial fitting window
                        initial_fit_log_rets = fitting_window_data_df['log_return'].dropna()
                        if len(initial_fit_log_rets) > 1:
                            initial_variance = initial_fit_log_rets.var()
                            # Ensure initial variance is non-negative and not NaN
                            last_ewma_variance_state = max(0.0, initial_variance) if pd.notna(initial_variance) else np.nan
                            log.debug(f"Step {step_number}: Initialized EWMA variance state with {last_ewma_variance_state:.8f} (from initial window N={len(initial_fit_log_rets)})")
                        else:
                            log.warning(f"Step {step_number}: Not enough data in initial window ({len(initial_fit_log_rets)}) to initialize EWMA variance.")
                            last_ewma_variance_state = np.nan
                        # EWMA forecast for t+1 is the state at time t
                        fc_variance_ewma_t = last_ewma_variance_state
                    else:
                        # Update EWMA state using previous step's actual log return
                        previous_actual_log_return = full_data_indexed.iloc[current_step_test_idx - 1]['log_return']
                        previous_actual_var_proxy = previous_actual_log_return**2 if pd.notna(previous_actual_log_return) else np.nan

                        if pd.notna(last_ewma_variance_state) and pd.notna(previous_actual_var_proxy):
                            # EWMA update: var_t = lambda * var_{t-1} + (1-lambda) * actual_proxy_{t-1}
                            # Forecast for t+1 is the state calculated using info up to t
                            current_ewma_variance_state = ewma_lambda * last_ewma_variance_state + (1 - ewma_lambda) * previous_actual_var_proxy
                            fc_variance_ewma_t = current_ewma_variance_state # Forecast for t+1 is state at t
                            last_ewma_variance_state = current_ewma_variance_state # Update state for next iteration
                        else:
                            fc_variance_ewma_t = np.nan
                            last_ewma_variance_state = np.nan # Reset state if calculation failed
                            log.warning(f"Step {step_number}: EWMA calculation failed due to NaN (last state: {last_ewma_variance_state}, prev proxy: {previous_actual_var_proxy}).")

                    # Ensure EWMA forecast is non-negative
                    if pd.notna(fc_variance_ewma_t) and fc_variance_ewma_t < 0:
                         log.warning(f"Step {step_number}: Negative EWMA variance forecast ({fc_variance_ewma_t:.6f}), clamping to 0.")
                         fc_variance_ewma_t = 0.0
                         last_ewma_variance_state = 0.0 # Also clamp the state
                    elif pd.notna(fc_variance_ewma_t):
                         # Calculate EWMA volatility forecast
                         fc_volatility_ewma_t = np.sqrt(fc_variance_ewma_t)
                    else:
                         fc_volatility_ewma_t = np.nan


                except Exception as ewma_exception:
                    log.error(f" Step {step_number}: Error during EWMA calculation: {ewma_exception}")
                    fc_variance_ewma_t = np.nan
                    fc_volatility_ewma_t = np.nan
                    last_ewma_variance_state = np.nan
                    append_nan_ewma = True

                if pd.isna(fc_variance_ewma_t): append_nan_ewma = True

                # --- Calculate VaR/ES and Violations for step t ---
                if not append_nan_ag:
                     # Use the UNDIFFERENCED log return forecast (fc_log_return_ag_t)
                     # and the calculated volatility (fc_volatility_ag_t)
                     var_thresh_ag_t, es_ag_t = calculate_parametric_var_es(
                         fc_log_return_ag_t, fc_volatility_ag_t, chosen_garch_dist_final, alpha, dist_params_model_t
                     )
                     # Determine violation if actual return and VaR threshold are valid
                     if pd.notna(actual_log_return_t) and pd.notna(var_thresh_ag_t):
                         violation_ag_t = 1 if actual_log_return_t < var_thresh_ag_t else 0

                if not append_nan_ewma:
                     # Use 0 mean for EWMA benchmark VaR
                     var_thresh_ewma_t, es_ewma_t = calculate_parametric_var_es(
                         0.0, fc_volatility_ewma_t, 'normal', alpha # Assume normal dist for EWMA benchmark
                     )
                     if pd.notna(actual_log_return_t) and pd.notna(var_thresh_ewma_t):
                         violation_ewma_t = 1 if actual_log_return_t < var_thresh_ewma_t else 0
                

            except Exception as step_exception:
                 log.error(f" Step {step_number}: Unexpected error in rolling step: {step_exception}", exc_info=True)
                 append_nan_ag = True; append_nan_ewma = True; skipped_steps_count += 1 # Count as skipped

            finally:
                # Append results for this step
                forecast_dates_list.append(current_date)
                actual_prices_list.append(actual_price_t)
                actual_log_returns_list.append(actual_log_return_t)
                actual_variance_proxies_list.append(actual_variance_proxy_t)
                # Append naive forecasts
                forecast_prices_naive_list.append(forecast_price_naive_t)
                forecast_log_returns_naive_list.append(forecast_log_return_naive_t)
                # Append AG forecasts (or NaN if failed)
                forecast_prices_ag_list.append(fc_price_ag_t if not append_nan_ag else np.nan)
                forecast_log_returns_ag_list.append(fc_log_return_ag_t if not append_nan_ag else np.nan) # Append undifferenced forecast
                forecast_volatility_ag_list.append(fc_volatility_ag_t if not append_nan_ag else np.nan)
                forecast_variance_ag_list.append(fc_variance_ag_t if not append_nan_ag else np.nan)
                # Append EWMA forecast (or NaN if failed)
                forecast_variance_ewma_list.append(fc_variance_ewma_t if not append_nan_ewma else np.nan)
                # Append VaR/ES/Violation results ---
                var_thresh_ag_list.append(var_thresh_ag_t)
                es_ag_list.append(es_ag_t)
                violation_ag_list.append(violation_ag_t)
                var_thresh_ewma_list.append(var_thresh_ewma_t)
                es_ewma_list.append(es_ewma_t)
                violation_ewma_list.append(violation_ewma_t)
                

                # Append parameters for stability plot if refit occurred successfully
                if gen_stab and should_refit and not append_nan_ag and (last_fit_arima_cached is not None or last_fit_garch_cached is not None):
                     stability_dates_list.append(current_date)
                     # Combine parameters from both models
                     combined_params_step = {}
                     if last_fit_arima_cached is not None: combined_params_step.update(getattr(last_fit_arima_cached,'params',pd.Series(dtype=float)).to_dict())
                     if last_fit_garch_cached is not None: combined_params_step.update(getattr(last_fit_garch_cached,'params',pd.Series(dtype=float)).to_dict())
                     # Append each tracked parameter, using NaN if not present
                     for p_name in stab_params:
                         stability_param_data[p_name].append(combined_params_step.get(p_name, np.nan))
                elif gen_stab and should_refit: # Handle case where refit was attempted but failed
                     stability_dates_list.append(current_date)
                     for p_name in stab_params:
                          stability_param_data[p_name].append(np.nan)

        # --- End of Rolling Forecast Loop ---

        results['rolling_forecast_duration_min']=(time.time()-t_roll_start)/60
        log.info(f"--- Rolling 1-Step Forecast Finished: {results['rolling_forecast_duration_min']:.2f} min ---")
        if fit_errors_count>0: log.warning(f"{fit_errors_count} ARIMA/GARCH fit errors encountered during rolling forecast.")
        if skipped_steps_count>0: log.warning(f"{skipped_steps_count}/{n_rolling_steps} steps skipped/failed during rolling forecast.")
        results['rolling_fit_errors']=fit_errors_count
        results['rolling_skipped_steps']=skipped_steps_count

        # --- Evaluate Rolling Forecast Results ---
        evaluation_df_dict = {
            'date':forecast_dates_list,
            'actual_price':actual_prices_list,
            'forecast_price_ag':forecast_prices_ag_list,
            'forecast_price_naive':forecast_prices_naive_list,
            'actual_log_return':actual_log_returns_list,
            'forecast_log_return_ag':forecast_log_returns_ag_list,
            'forecast_log_return_naive': forecast_log_returns_naive_list,
            'forecast_volatility_ag':forecast_volatility_ag_list,
            'forecast_variance_ag':forecast_variance_ag_list,
            'forecast_variance_ewma':forecast_variance_ewma_list,
            'actual_variance_proxy':actual_variance_proxies_list,
            # --- Add VaR/ES/Violation columns ---
            'var_thresh_ag': var_thresh_ag_list,
            'es_ag': es_ag_list,
            'violation_ag': violation_ag_list,
            'var_thresh_ewma': var_thresh_ewma_list,
            'es_ewma': es_ewma_list,
            'violation_ewma': violation_ewma_list
            
        }
        # Check if all lists have the same length before creating DataFrame
        list_lengths = {k: len(v) for k, v in evaluation_df_dict.items()}
        if len(set(list_lengths.values())) > 1:
            log.error(f"List length mismatch for evaluation DataFrame: {list_lengths}")
            # Handle error: maybe return partial results or raise exception
            raise ValueError("Inconsistent list lengths for evaluation DataFrame creation.")

        evaluation_df = pd.DataFrame(evaluation_df_dict).set_index('date')


        # --- Evaluate Price Forecasts ---
        required_cols_price=['actual_price','forecast_price_ag','forecast_price_naive']
        eval_df_price=evaluation_df.dropna(subset=required_cols_price)
        valid_price_steps=len(eval_df_price)
        results['rolling_valid_steps_price']=valid_price_steps
        print(f"\n--- Rolling 1-Step Performance (Test Set, d={d_final}) ---")
        log.info(f"Evaluating 1-step price forecasts over {valid_price_steps} valid steps.")
        if valid_price_steps > 0:
             
            yt_p=eval_df_price['actual_price'].values
            yp_p_ag=eval_df_price['forecast_price_ag'].values
            yp_p_n=eval_df_price['forecast_price_naive'].values
            # Calculate metrics
            results['price_mae_ag']=mean_absolute_error(yt_p,yp_p_ag)
            results['price_rmse_ag']=root_mean_squared_error(yt_p,yp_p_ag)
            results['price_mape_ag']=mean_absolute_percentage_error(yt_p,yp_p_ag)
            results['price_mae_naive']=mean_absolute_error(yt_p,yp_p_n)
            results['price_rmse_naive']=root_mean_squared_error(yt_p,yp_p_n)
            results['price_mape_naive']=mean_absolute_percentage_error(yt_p,yp_p_n)
            # Print metrics
            print("  1-Step Price Accuracy:")
            print(f"    AG    MAE:{results['price_mae_ag']:,.4f} RMSE:{results['price_rmse_ag']:,.4f} MAPE:{results['price_mape_ag']:.2f}%")
            print(f"    Naive MAE:{results['price_mae_naive']:,.4f} RMSE:{results['price_rmse_naive']:,.4f} MAPE:{results['price_mape_naive']:.2f}%")

            # Perform Diebold-Mariano Test for Price Forecasts (AG vs Naive)
            print(f"\n--- Diebold-Mariano Test for Prices (Backtest, h=1, Compare AG vs. Naive) ---")
            dm_results_price_bt = {}
            dm_loss_type_price = config.get("dm_test_loss_type", "Squared Error") # Define loss type here
            if not DIEBOLDMARIANO_LIB_AVAILABLE:
                 dm_results_price_bt = {'error': "Library not found"}
                 print("  DM Test skipped: 'dieboldmariano' library not installed.")
            else:
                 actuals_dm = eval_df_price['actual_price']
                 forecasts_ag_dm = eval_df_price['forecast_price_ag']
                 forecasts_naive_dm = eval_df_price['forecast_price_naive']
                 # H1: AG model has lower loss than Naive model (alternative='less')
                 dm_test_output_p = diebold_mariano_test(actuals_dm, forecasts_ag_dm, forecasts_naive_dm, h=1, loss_type=dm_loss_type_price, alternative='less')
                 dm_results_price_bt = dm_test_output_p.copy()
                 dm_results_price_bt['is_significant'] = False # Initialize
                 if dm_test_output_p['error'] is not None:
                     dm_results_price_bt['interpretation'] = f"Error: {dm_test_output_p['error']}"
                 elif pd.notna(dm_test_output_p['p_value']):
                     p_val, dm_stat = dm_test_output_p['p_value'], dm_test_output_p['dm_stat']
                     if p_val < alpha: # Use defined alpha
                         if dm_stat < 0:
                             dm_results_price_bt['interpretation'] = f"AG significantly better (p={p_val:.4f} < {alpha})"
                             dm_results_price_bt['is_significant'] = True
                         else:
                             dm_results_price_bt['interpretation'] = f"Significant (p={p_val:.4f}), but DM Stat >= 0 ({dm_stat:.2f}) - Contradiction?"
                     else:
                         dm_results_price_bt['interpretation'] = f"No significant difference (p={p_val:.4f} >= {alpha})"
                 else:
                     dm_results_price_bt['interpretation'] = "Test result invalid (NaN p-value)"
                 print(f"  Loss='{dm_loss_type_price}' | N={dm_results_price_bt['n_obs']:<5} | DM Stat={dm_results_price_bt['dm_stat']:>8.3f} | p-value={dm_results_price_bt['p_value']:>8.4f} | {dm_results_price_bt['interpretation']}")
            results['roll_fc_price_dm_test'] = dm_results_price_bt
        else:
             log.warning("No valid steps for 1-step price evaluation (including DM test).")
             results.update({k:np.nan for k in ['price_mae_ag','price_rmse_ag','price_mape_ag','price_mae_naive','price_rmse_naive','price_mape_naive']})
             results['roll_fc_price_dm_test'] = {'error': 'No valid steps for price evaluation'}

        # --- Evaluate Log Return Forecasts --- 
        required_cols_lr = ['actual_log_return', 'forecast_log_return_ag', 'forecast_log_return_naive']
        eval_df_lr = evaluation_df.dropna(subset=required_cols_lr)
        valid_lr_steps = len(eval_df_lr)
        results['rolling_valid_steps_lr'] = valid_lr_steps
        print("\n  1-Step Log Return Accuracy:")
        if valid_lr_steps > 0:
             
            yt_lr = eval_df_lr['actual_log_return'].values
            yp_lr_ag = eval_df_lr['forecast_log_return_ag'].values
            yp_lr_n = eval_df_lr['forecast_log_return_naive'].values # Should be all zeros
            results['logret_mae_ag'] = mean_absolute_error(yt_lr, yp_lr_ag)
            results['logret_rmse_ag'] = root_mean_squared_error(yt_lr, yp_lr_ag)
            results['logret_mae_naive'] = mean_absolute_error(yt_lr, yp_lr_n)
            results['logret_rmse_naive'] = root_mean_squared_error(yt_lr, yp_lr_n)
            print(f"    AG    MAE:{results['logret_mae_ag']:.6f} RMSE:{results['logret_rmse_ag']:.6f}")
            print(f"    Naive MAE:{results['logret_mae_naive']:.6f} RMSE:{results['logret_rmse_naive']:.6f} (vs. 0 Forecast)")

            # Perform Diebold-Mariano Test for Log Return Forecasts (AG vs Naive=0)
            print(f"\n--- Diebold-Mariano Test for Log Returns (Backtest, h=1, Compare AG vs. Naive=0) ---")
            dm_results_lr_bt = {}
            if not DIEBOLDMARIANO_LIB_AVAILABLE:
                 dm_results_lr_bt = {'error': "Library not found"}
                 print("  DM Test for Log Returns skipped: 'dieboldmariano' library not installed.")
            else:
                 actuals_lr_dm = eval_df_lr['actual_log_return']
                 forecasts_ag_lr_dm = eval_df_lr['forecast_log_return_ag']
                 forecasts_naive_lr_dm = eval_df_lr['forecast_log_return_naive'] # Series of zeros
                 # H1: AG model has lower loss than Naive=0 model (alternative='less')
                 dm_test_lr_output = diebold_mariano_test(actuals_lr_dm, forecasts_ag_lr_dm, forecasts_naive_lr_dm, h=1, loss_type=dm_loss_type_price, alternative='less')
                 dm_results_lr_bt = dm_test_lr_output.copy()
                 dm_results_lr_bt['is_significant'] = False # Initialize
                 if dm_test_lr_output['error'] is not None:
                     dm_results_lr_bt['interpretation'] = f"Error: {dm_test_lr_output['error']}"
                 elif pd.notna(dm_test_lr_output['p_value']):
                     p_val, dm_stat = dm_test_lr_output['p_value'], dm_test_lr_output['dm_stat']
                     if p_val < alpha: # Use defined alpha
                         if dm_stat < 0:
                             dm_results_lr_bt['interpretation'] = f"AG significantly better (p={p_val:.4f} < {alpha})"
                             dm_results_lr_bt['is_significant'] = True
                         else:
                              dm_results_lr_bt['interpretation'] = f"Significant (p={p_val:.4f}), but DM Stat >= 0 ({dm_stat:.2f}) - Contradiction?"
                     else:
                         dm_results_lr_bt['interpretation'] = f"No significant difference (p={p_val:.4f} >= {alpha})"
                 else:
                     dm_results_lr_bt['interpretation'] = "Test result invalid (NaN p-value)"
                 print(f"  Loss='{dm_loss_type_price}' | N={dm_results_lr_bt['n_obs']:<5} | DM Stat={dm_results_lr_bt['dm_stat']:>8.3f} | p-value={dm_results_lr_bt['p_value']:>8.4f} | {dm_results_lr_bt['interpretation']}")
            results['roll_fc_logret_dm_test'] = dm_results_lr_bt
        else:
            log.warning("No valid steps for 1-step log return evaluation (including DM test).")
            results.update({k: np.nan for k in ['logret_mae_ag', 'logret_rmse_ag', 'logret_mae_naive', 'logret_rmse_naive']})
            results['roll_fc_logret_dm_test'] = {'error': 'No valid steps for LogRet evaluation'}

        # --- Evaluate Variance Forecasts --- 
        required_cols_var = ['actual_variance_proxy','forecast_variance_ag', 'forecast_variance_ewma']
        eval_df_var = evaluation_df.dropna(subset=required_cols_var)
        # Further filter for non-negative values required for variance metrics
        eval_df_var = eval_df_var[(eval_df_var['actual_variance_proxy'] >= 0) & (eval_df_var['forecast_variance_ag'] >= 0) & (eval_df_var['forecast_variance_ewma'] >= 0)]
        valid_var_steps = len(eval_df_var)
        results['rolling_valid_steps_variance'] = valid_var_steps
        print("\n  1-Step Variance Accuracy (vs Proxy=LogRet^2):")
        if valid_var_steps > 0:
             
            yt_v = eval_df_var['actual_variance_proxy'].values
            yp_v_ag = eval_df_var['forecast_variance_ag'].values
            yp_v_ewma = eval_df_var['forecast_variance_ewma'].values
            # Calculate metrics for AG
            results['vol_rmse_ag'] = root_mean_squared_error_variance(yt_v, yp_v_ag)
            results['vol_qlike_ag'] = qlike_loss(yt_v, yp_v_ag, epsilon=qlike_eps)
            print(f"    AG    RMSE:{results['vol_rmse_ag']:.8f} QLIKE:{results['vol_qlike_ag']:.6f} ({valid_var_steps} steps)")
            # Calculate metrics for EWMA benchmark
            results['benchmarks']['ewma_var']['rmse'] = root_mean_squared_error_variance(yt_v, yp_v_ewma)
            results['benchmarks']['ewma_var']['qlike'] = qlike_loss(yt_v, yp_v_ewma, epsilon=qlike_eps)
            print(f"    EWMA  RMSE:{results['benchmarks']['ewma_var']['rmse']:.8f} QLIKE:{results['benchmarks']['ewma_var']['qlike']:.6f} ({valid_var_steps} steps)")

            # Perform Diebold-Mariano Test for Variance Forecasts (AG vs EWMA)
            print(f"\n--- Diebold-Mariano Test for Variance (Backtest, h=1, Compare AG vs. EWMA) ---")
            dm_results_var_bt = {}
            if not DIEBOLDMARIANO_LIB_AVAILABLE:
                 dm_results_var_bt = {'error': "Library not found"}
                 print("  Variance DM Test skipped: 'dieboldmariano' library not installed.")
            else:
                 actuals_v_dm = eval_df_var['actual_variance_proxy']
                 forecasts_ag_v_dm = eval_df_var['forecast_variance_ag']
                 forecasts_ewma_v_dm = eval_df_var['forecast_variance_ewma']
                 # Pass epsilon if using QLIKE
                 eps_arg_dm = {'qlike_epsilon': qlike_eps} if dm_var_loss == "QLIKE" else {} # Use defined dm_var_loss
                 # H1: AG model has lower loss than EWMA model (alternative='less')
                 dm_test_var_output = diebold_mariano_test(
                     actuals_v_dm, forecasts_ag_v_dm, forecasts_ewma_v_dm,
                     h=1, loss_type=dm_var_loss, alternative='less', **eps_arg_dm # Use defined dm_var_loss
                 )
                 dm_results_var_bt = dm_test_var_output.copy()
                 dm_results_var_bt['is_significant'] = False # Initialize
                 if dm_test_var_output['error'] is not None:
                     dm_results_var_bt['interpretation'] = f"Error: {dm_test_var_output['error']}"
                 elif pd.notna(dm_test_var_output['p_value']):
                     p_val, dm_stat = dm_test_var_output['p_value'], dm_test_var_output['dm_stat']
                     if p_val < alpha: # Use defined alpha
                         if dm_stat < 0:
                             dm_results_var_bt['interpretation'] = f"AG significantly better (p={p_val:.4f} < {alpha})"
                             dm_results_var_bt['is_significant'] = True
                         else:
                             dm_results_var_bt['interpretation'] = f"Significant (p={p_val:.4f}), but DM Stat >= 0 ({dm_stat:.2f}) - Contradiction?"
                     else:
                         dm_results_var_bt['interpretation'] = f"No significant difference (p={p_val:.4f} >= {alpha})"
                 else:
                     dm_results_var_bt['interpretation'] = "Test result invalid (NaN p-value)"
                 print(f"  Loss='{dm_var_loss}' | N={dm_results_var_bt['n_obs']:<5} | DM Stat={dm_results_var_bt['dm_stat']:>8.3f} | p-value={dm_results_var_bt['p_value']:>8.4f} | {dm_results_var_bt['interpretation']}")
            results['variance_dm']['backtest'] = dm_results_var_bt
        else:
            log.warning("No valid steps for 1-step variance evaluation (including DM test).")
            results.update({k:np.nan for k in ['vol_rmse_ag','vol_qlike_ag']})
            results['benchmarks']['ewma_var'] = {'rmse': np.nan, 'qlike': np.nan}
            results['variance_dm']['backtest'] = {'error': 'No valid steps for variance evaluation'}

        # --- Evaluate VaR/ES Backtest ---
        print(f"\n--- VaR Backtesting (Backtest, alpha={alpha:.2%}) ---")
        # Extract violation series (dropping NaNs which indicate calculation failure)
        violation_ag_series = evaluation_df['violation_ag'].dropna().astype(int).values
        violation_ewma_series = evaluation_df['violation_ewma'].dropna().astype(int).values
        n_backtest_ag = len(violation_ag_series)
        n_backtest_ewma = len(violation_ewma_series)

        results['backtest_n_obs_var'] = n_backtest_ag # Store number of observations for backtest

    

        if n_backtest_ag < min_resid_len_diag: # Use same min length as residual diagnostics
            log.warning(f"Skipping VaR backtests: Not enough valid observations for AG model ({n_backtest_ag} < {min_resid_len_diag}).")
            results.update({k: np.nan for k in ['backtest_violations_ag', 'backtest_kupiec_p_ag', 'backtest_christ_p_ag']})
        else:
            n_violations_ag = int(np.sum(violation_ag_series))
            results['backtest_violations_ag'] = n_violations_ag
            print(f"  AG Model: Violations={n_violations_ag}/{n_backtest_ag} (Expected: {n_backtest_ag*alpha:.1f})")
            kupiec_ag_res = kupiec_test(n_violations_ag, n_backtest_ag, alpha)
            christ_ag_res = christoffersen_test(violation_ag_series, alpha)
            results['backtest_kupiec_p_ag'] = kupiec_ag_res.get('p_value')
            results['backtest_christ_p_ag'] = christ_ag_res.get('p_value')
            results['backtest_kupiec_stat_ag'] = kupiec_ag_res.get('LR_stat')
            results['backtest_christ_stat_ag'] = christ_ag_res.get('LR_stat')
            print(f"    Kupiec p={results['backtest_kupiec_p_ag']:.4f}, Christoffersen p={results['backtest_christ_p_ag']:.4f}")

        if n_backtest_ewma < min_resid_len_diag:
            log.warning(f"Skipping VaR backtests: Not enough valid observations for EWMA model ({n_backtest_ewma} < {min_resid_len_diag}).")
            results.update({k: np.nan for k in ['backtest_violations_ewma', 'backtest_kupiec_p_ewma', 'backtest_christ_p_ewma']})
        else:
            n_violations_ewma = int(np.sum(violation_ewma_series))
            results['backtest_violations_ewma'] = n_violations_ewma
            print(f"  EWMA Benchmark: Violations={n_violations_ewma}/{n_backtest_ewma} (Expected: {n_backtest_ewma*alpha:.1f})")
            kupiec_ewma_res = kupiec_test(n_violations_ewma, n_backtest_ewma, alpha)
            christ_ewma_res = christoffersen_test(violation_ewma_series, alpha)
            results['backtest_kupiec_p_ewma'] = kupiec_ewma_res.get('p_value')
            results['backtest_christ_p_ewma'] = christ_ewma_res.get('p_value')
            results['backtest_kupiec_stat_ewma'] = kupiec_ewma_res.get('LR_stat')
            results['backtest_christ_stat_ewma'] = christ_ewma_res.get('LR_stat')
            print(f"    Kupiec p={results['backtest_kupiec_p_ewma']:.4f}, Christoffersen p={results['backtest_christ_p_ewma']:.4f}")

        # Calculate Average VaR/ES over the backtest period
        results['backtest_avg_var_ag'] = evaluation_df['var_thresh_ag'].mean()
        results['backtest_avg_es_ag'] = evaluation_df['es_ag'].mean()
        results['backtest_avg_var_ewma'] = evaluation_df['var_thresh_ewma'].mean()
        results['backtest_avg_es_ewma'] = evaluation_df['es_ewma'].mean()
        print(f"  Avg VaR AG={results['backtest_avg_var_ag']:.4f}, Avg ES AG={results['backtest_avg_es_ag']:.4f}")
        print(f"  Avg VaR EWMA={results['backtest_avg_var_ewma']:.4f}, Avg ES EWMA={results['backtest_avg_es_ewma']:.4f}")
        
                # --- Store Backtest VaR/ES/Backtest metrics directly in results or roll_fc ---
        if n_backtest_ag >= min_resid_len_diag:
            results['roll_fc']['backtest_violations_ag'] = n_violations_ag # Store in sub-dict
            results['backtest_kupiec_p_ag'] = kupiec_ag_res.get('p_value') # Store directly
            results['backtest_christ_p_ag'] = christ_ag_res.get('p_value')
        else:
             results['roll_fc']['backtest_violations_ag'] = np.nan
             results['backtest_kupiec_p_ag'] = np.nan
             results['backtest_christ_p_ag'] = np.nan
        # Do the same for EWMA results
        if n_backtest_ewma >= min_resid_len_diag:
             results['roll_fc']['backtest_violations_ewma'] = n_violations_ewma
             results['backtest_kupiec_p_ewma'] = kupiec_ewma_res.get('p_value')
             results['backtest_christ_p_ewma'] = christ_ewma_res.get('p_value')
        else:
             results['roll_fc']['backtest_violations_ewma'] = np.nan
             results['backtest_kupiec_p_ewma'] = np.nan
             results['backtest_christ_p_ewma'] = np.nan

        results['roll_fc']['backtest_avg_var_ag'] = evaluation_df['var_thresh_ag'].mean()
        results['roll_fc']['backtest_avg_es_ag'] = evaluation_df['es_ag'].mean()
        results['roll_fc']['backtest_avg_var_ewma'] = evaluation_df['var_thresh_ewma'].mean()
        results['roll_fc']['backtest_avg_es_ewma'] = evaluation_df['es_ewma'].mean()
        results['roll_fc']['backtest_n_obs_var'] = n_backtest_ag 
        
          
                # --- Generate and Store Thesis Recommendations for Backtest Mode ---
        alpha_bt = config.get("dm_test_alpha", 0.05)
        dm_loss_price_bt = config.get("dm_test_loss_type", "Squared Error")
        dm_loss_var_bt = config.get("dm_test_variance_loss_type", "QLIKE")

        # --- Price DM Recommendation ---
        reco_price_text_bt = f"Diebold-Mariano Test (Price, AG vs Naive, h=1, alpha={alpha_bt:.2%}, H1: AG better, Loss={dm_loss_price_bt}):\n"
        dm_p_res_bt = results.get('roll_fc_price_dm_test', {}) 
        if dm_p_res_bt.get('error'):
            reco_price_text_bt += f"- Test Error: {dm_p_res_bt['error']}.\n"
        elif pd.notna(dm_p_res_bt.get('p_value')):
            p_val = dm_p_res_bt['p_value']
            dm_stat = dm_p_res_bt.get('dm_stat', 0)
            if p_val < alpha_bt and dm_stat < 0:
                 reco_price_text_bt += f"- AG significantly better (p={p_val:.3f}).\n"
            else:
                 reco_price_text_bt += f"- No significant difference (p={p_val:.3f}).\n"
        else: reco_price_text_bt += "- Test Skipped or Failed.\n"
        
        results["thesis_reco_price_dm"] = reco_price_text_bt 

        # --- Variance DM Recommendation ---
        reco_var_text_bt = f"Diebold-Mariano Test (Variance, AG vs EWMA, h=1, alpha={alpha_bt:.2%}, H1: AG better, Loss={dm_loss_var_bt}):\n"
        dm_v_res_bt = results.get('variance_dm', {}).get('backtest', {}) 
        if dm_v_res_bt.get('error'):
             reco_var_text_bt += f"- Test Error: {dm_v_res_bt['error']}.\n"
        elif pd.notna(dm_v_res_bt.get('p_value')):
             p_val = dm_v_res_bt['p_value']
             dm_stat = dm_v_res_bt.get('dm_stat', 0)
             if p_val < alpha_bt and dm_stat < 0:
                  reco_var_text_bt += f"- AG significantly better (p={p_val:.3f}).\n"
             else:
                  reco_var_text_bt += f"- No significant difference (p={p_val:.3f}).\n"
        else: reco_var_text_bt += "- Test Skipped or Failed.\n"
        
        results["thesis_reco_var_dm"] = reco_var_text_bt 
    

        # --- VaR Backtesting Recommendation ---
        reco_var_backtest_text_bt = f"VaR Backtesting (AG vs EWMA, h=1, alpha={alpha_bt:.2%}):\n"
       
        kup_p_ag = results.get('roll_fc', {}).get('bt_kupiec_p_ag', results.get('bt_kupiec_p_ag')) 
        chr_p_ag = results.get('roll_fc', {}).get('bt_christ_p_ag', results.get('bt_christ_p_ag')) 
        kup_p_ewma = results.get('roll_fc', {}).get('bt_kupiec_p_ewma', results.get('bt_kupiec_p_ewma')) 
        kup_p_ag_str = f"{kup_p_ag:.3f}" if pd.notna(kup_p_ag) else "N/A"
        chr_p_ag_str = f"{chr_p_ag:.3f}" if pd.notna(chr_p_ag) else "N/A"
        kup_p_ewma_str = f"{kup_p_ewma:.3f}" if pd.notna(kup_p_ewma) else "N/A"
        reco_var_backtest_text_bt += f"- AG Model: Kupiec {'OK' if pd.isna(kup_p_ag) or kup_p_ag >= alpha_bt else 'REJECTED'} (p={kup_p_ag_str}), Christoffersen {'OK' if pd.isna(chr_p_ag) or chr_p_ag >= alpha_bt else 'REJECTED'} (p={chr_p_ag_str}).\n"
        reco_var_backtest_text_bt += f"- EWMA Model: Kupiec {'OK' if pd.isna(kup_p_ewma) or kup_p_ewma >= alpha_bt else 'REJECTED'} (p={kup_p_ewma_str}).\n"
       
        results["thesis_reco_var_backtest"] = reco_var_backtest_text_bt 
         
        # --- END Recommendation Block ---
        

        # --- Residual Diagnostics on Rolling Forecast Errors --- 
        required_cols_resid=['actual_log_return','forecast_log_return_ag','forecast_volatility_ag']
        eval_df_resid=evaluation_df.dropna(subset=required_cols_resid)
        valid_resid_steps = len(eval_df_resid)
        

        if valid_resid_steps > min_resid_len_diag:
            log.info(f"Diagnosing {valid_resid_steps} standardized 1-step AG residuals...")
            # Calculate standardized residuals: (actual - forecast) / forecast_volatility
            forecast_error=eval_df_resid['actual_log_return']-eval_df_resid['forecast_log_return_ag']
            forecast_vol=eval_df_resid['forecast_volatility_ag'].replace(0,np.nan).clip(lower=1e-8) # Avoid division by zero/small numbers
            standardized_residuals=(forecast_error/forecast_vol).dropna()

            if len(standardized_residuals)>min_resid_len_diag:
                 print("\n  Rolling 1-Step Residual Diagnostics (Standardized, AG Model):")
                 # Determine lags for tests (e.g., min(20, N/2 - 1))
                 lb_lags_diag = min(20, len(standardized_residuals)//2 - 1) if len(standardized_residuals) > 5 else 0
                 arch_lags_diag = min(12, len(standardized_residuals)//2 - 1) if len(standardized_residuals) > 5 else 0

                 # Ljung-Box test on standardized residuals (check for remaining autocorrelation)
                 lb_result=ljung_box_test(standardized_residuals,l=lb_lags_diag if lb_lags_diag>0 else 1)
                 # ARCH-LM test on standardized residuals (not squared here, test for ARCH in std resid)
                 arch_result=arch_test(standardized_residuals,l=arch_lags_diag if arch_lags_diag>0 else 1)

                 results['resid_lb_pvalue']=lb_result.get('lb_pvalue')
                 results['resid_lb_wn']=lb_result.get('is_white_noise') # True if no autocorrelation
                 results['resid_arch_pvalue']=arch_result.get('arch_pvalue')
                 results['resid_arch_het']=arch_result.get('heteroskedastic') # True if ARCH effects remain

                 print(f"    Ljung-Box p={results['resid_lb_pvalue']:.4f} (Lag={lb_result.get('lb_lag_tested',0)}, White Noise={results['resid_lb_wn']})")
                 print(f"    ARCH LM   p={results['resid_arch_pvalue']:.4f} (Lag={arch_result.get('arch_lag_tested',0)}, Heteroskedastic={results['resid_arch_het']})")

                 # Plot standardized residuals if plotting enabled
                 if is_plot_available and plt is not None:
                     try:
                         fig,ax=plt.subplots(figsize=(12,5))
                         ax.plot(standardized_residuals.index,standardized_residuals.values,lw=1.0,marker='.',ms=3,alpha=0.7,label='Standardized Residuals (AG)',color='black')
                         ax.set_title(f"Standardized 1-Step Residuals ({coin_id}, d={d_final})\n({final_model_desc})")
                         ax.axhline(0,color='red',linestyle='--')
                         ax.axhline(1.96,color='grey',linestyle=':', linewidth=0.8) # 95% bounds for approx normal
                         ax.axhline(-1.96,color='grey',linestyle=':', linewidth=0.8)
                         ax.grid(True); ax.legend(); fig.tight_layout()
                         resid_filename=os.path.join(coin_plot_dir,f"{coin_id}_residuals_roll_std_d{d_final}.png")
                         fig.savefig(resid_filename); plt.close(fig)
                         log.info(f"Rolling residuals plot saved: {resid_filename}")
                     except Exception as e: log.error(f"Plotting rolling residuals failed: {e}"); plt.close('all')

            else:
                 log.warning(f"Not enough standardized residuals ({len(standardized_residuals)}) for diagnostic tests (require > {min_resid_len_diag}).")
                 results.update({k:np.nan for k in ['resid_lb_pvalue','resid_lb_wn','resid_arch_pvalue','resid_arch_het']})
        else:
            log.warning(f"Skipping residual diagnostics: Not enough valid forecast steps ({valid_resid_steps}) with required data.")
            results.update({k:np.nan for k in ['resid_lb_pvalue','resid_lb_wn','resid_arch_pvalue','resid_arch_het']})

        # --- Generate Parameter Stability Plot --- 
        if gen_stab and stability_dates_list and any(len(v)>1 for v in stability_param_data.values()):
             log.info("Generating parameter stability plot...")
             stab_filename=os.path.join(coin_plot_dir,f"{coin_id}_param_stab_d{d_final}.png")
             stab_title=f"{coin_id} Parameter Stability (Backtest, d={d_final})\n({final_model_desc})"
             # Filter out parameters that were never estimated or always NaN
             valid_stab_data={k:v for k,v in stability_param_data.items() if any(pd.notna(x) for x in v)}
             if valid_stab_data:
                 try:
                     plot_ok=plot_parameter_stability(valid_stab_data, stability_dates_list, stab_filename, stab_title)
                     results["parameter_stability_plot_path"]=stab_filename if plot_ok else "Failed"
                 except Exception as e:
                     log.error(f"Generating parameter stability plot failed: {e}")
                     results["parameter_stability_plot_path"]=f"Error: {e}"
             else:
                 log.warning("Skipping stability plot: No valid parameter data collected during backtest.")
                 results["parameter_stability_plot_path"]="Skipped (no valid data)"
        elif gen_stab:
             log.warning("Skipping stability plot: Stability tracking disabled or no data collected.")
             results["parameter_stability_plot_path"]="Skipped (no data/disabled)"

        # --- Save Rolling Forecast Results --- 
        try:
            eval_filename=os.path.join(coin_plot_dir,f"{coin_id}_rolling_1step_results_d{d_final}.csv")
            evaluation_df.to_csv(eval_filename,index=True,float_format="%.8f")
            log.info(f"Rolling 1-step results CSV saved: {eval_filename}")
            results["rolling_results_csv_path"]=eval_filename
        except Exception as e:
            log.error(f"Saving rolling 1-step results CSV failed: {e}")
            results["rolling_results_csv_path"]=f"Failed: {e}"

        # --- Generate Plots for Rolling Forecasts --- 
        if is_plot_available and plt is not None:
             # Plot Price Forecasts vs Actuals
             if valid_price_steps>0:
                 try:
                     fig,ax=plt.subplots(figsize=(14,7))
                     ax.plot(eval_df_price.index,eval_df_price['actual_price'],label='Actual Price',color='black',lw=1.5)
                     ax.plot(eval_df_price.index,eval_df_price['forecast_price_ag'],label='AG Forecast (t+1)',color='red',ls='--',lw=1.0,alpha=0.9)
                     ax.plot(eval_df_price.index,eval_df_price['forecast_price_naive'],label='Naive Forecast (y_t)',color='blue',ls=':',lw=1.0,alpha=0.7)
                     ax.set_title(f'{coin_id} Rolling 1-Step Price Forecast (d={d_final})\n({final_model_desc})')
                     ax.set_ylabel("Price")
                     ax.legend(); ax.grid(True,ls=':')
                     plt.setp(ax.xaxis.get_majorticklabels(),rotation=45,ha='right')
                     fig.tight_layout()
                     price_plot_filename=os.path.join(coin_plot_dir,f"{coin_id}_roll_1step_price_fc_d{d_final}.png")
                     fig.savefig(price_plot_filename); plt.close(fig)
                     log.info(f"Rolling 1-step price plot saved: {price_plot_filename}")
                 except Exception as e: log.error(f"Rolling 1-step price plot failed: {e}"); plt.close('all')
             else: log.warning("Skipping rolling 1-step price plot: No valid steps.")

             # Plot Volatility Forecasts vs Actual Proxy
             volatility_index=evaluation_df.index
             if not volatility_index.empty:
                 try:
                     fig,ax=plt.subplots(figsize=(14,7))
                     # Use absolute log return as a proxy for actual volatility
                     actual_vol_proxy=np.sqrt(evaluation_df.loc[volatility_index,'actual_variance_proxy'].clip(0)) # sqrt(proxy) = |logret|
                     ax.plot(volatility_index, actual_vol_proxy, label='Actual Proxy |LogRet|',color='purple',lw=1.2,alpha=0.5)
                     # Plot AG volatility forecast if available
                     eval_vol_ag_valid = evaluation_df['forecast_volatility_ag'].dropna()
                     if not eval_vol_ag_valid.empty:
                         ax.plot(eval_vol_ag_valid.index, eval_vol_ag_valid, label='AG Forecast Vol (t+1)',color='orange',ls='-',lw=1.5)
                     # Plot EWMA volatility forecast if available
                     ewma_vol_valid = np.sqrt(evaluation_df['forecast_variance_ewma'].dropna().clip(0))
                     if not ewma_vol_valid.empty:
                         ax.plot(ewma_vol_valid.index, ewma_vol_valid, label='EWMA Forecast Vol (t+1)',color='black',ls=':',lw=1.2,alpha=0.8)

                     ax.set_title(f'{coin_id} Rolling 1-Step Volatility Forecast (AG vs EWMA, d={d_final})\n({final_model_desc})')
                     ax.set_ylabel("Volatility (Daily)")
                     ax.legend(); ax.grid(True,ls=':')
                     plt.setp(ax.xaxis.get_majorticklabels(),rotation=45,ha='right')
                     fig.tight_layout()
                     vol_plot_filename=os.path.join(coin_plot_dir,f"{coin_id}_roll_1step_vol_fc_ag_vs_ewma_d{d_final}.png")
                     fig.savefig(vol_plot_filename); plt.close(fig)
                     log.info(f"Rolling 1-step volatility plot (AG vs EWMA) saved: {vol_plot_filename}")
                 except Exception as e: log.error(f"Rolling 1-step volatility plot failed: {e}"); plt.close('all')
             else: log.warning("Skipping rolling 1-step volatility plot: No forecast data.")

             # --- Plot VaR Thresholds ---
             if is_plot_available and plt is not None and 'actual_log_return' in evaluation_df.columns:
                  eval_df_var_plot = evaluation_df.dropna(subset=['actual_log_return', 'var_thresh_ag', 'var_thresh_ewma'])
                  if not eval_df_var_plot.empty:
                       try:
                           fig, ax = plt.subplots(figsize=(14, 7))
                           ax.plot(eval_df_var_plot.index, eval_df_var_plot['actual_log_return'], label='Actual Log Return', color='black', alpha=0.6, lw=1.0)
                           ax.plot(eval_df_var_plot.index, eval_df_var_plot['var_thresh_ag'], label=f'AG VaR ({alpha:.0%})', color='red', ls='--', lw=1.2)
                           ax.plot(eval_df_var_plot.index, eval_df_var_plot['var_thresh_ewma'], label=f'EWMA VaR ({alpha:.0%})', color='blue', ls=':', lw=1.2)
                           # Highlight violations for AG model
                           violations_idx_ag = eval_df_var_plot[eval_df_var_plot['violation_ag'] == 1].index
                           if not violations_idx_ag.empty:
                               ax.scatter(violations_idx_ag, eval_df_var_plot.loc[violations_idx_ag, 'actual_log_return'], color='red', marker='o', s=20, label='AG Violation')

                           ax.set_title(f'{coin_id} Rolling 1-Step VaR ({alpha:.0%}) Thresholds vs Actual Log Return (d={d_final})')
                           ax.set_ylabel("Log Return / VaR Threshold")
                           ax.legend(); ax.grid(True, ls=':')
                           plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                           fig.tight_layout()
                           var_plot_filename = os.path.join(coin_plot_dir, f"{coin_id}_roll_1step_var_thresh_d{d_final}.png")
                           fig.savefig(var_plot_filename); plt.close(fig)
                           log.info(f"Rolling 1-step VaR plot saved: {var_plot_filename}")
                       except Exception as e:
                           log.error(f"Rolling 1-step VaR plot failed: {e}"); plt.close('all')
                  else:
                      log.warning("Skipping rolling 1-step VaR plot: No valid data points.")
             


        # --- Mode 2: Multi-Horizon Evaluation ('horizon_evaluation') ---
    elif forecast_mode == 'horizon_evaluation':

        dm_loss_type_price = config.get("dm_test_loss_type", "Squared Error")
        dm_loss_type_var = config.get("dm_test_variance_loss_type", "QLIKE")
        alpha = config.get("dm_test_alpha", 0.05)
        min_resid_len_diag = 25 # Minimum length for reliable Backtests/Residual Diag


        valid_horizons_eval = sorted(list(set(h for h in config.get('evaluation_horizons', []) if isinstance(h, int) and h > 0)))
        if not valid_horizons_eval: raise ValueError("No valid evaluation_horizons provided for 'horizon_evaluation' mode.")
        horizons = valid_horizons_eval
        max_horizon = max(horizons)

        log.info(f"--- [{coin_id}] Starting Multi-Horizon Evaluation ---")
        log.info(f" VaR/ES/Backtest Alpha: {alpha:.2%}")

        # --- Risk Metrics on Test Set (Empirical) ---
        test_returns = test_df['log_return'].dropna()
        if not test_returns.empty:
            var_95_empirical = value_at_risk(test_returns, alpha)
            es_95_empirical = expected_shortfall(test_returns, alpha)
            log.info(f"Empirical Test Set (N={len(test_returns)}): VaR({alpha:.0%})={var_95_empirical:.4f}, ES({alpha:.0%})={es_95_empirical:.4f}")
            results[f'empirical_var{int(alpha*100)}'] = var_95_empirical
            results[f'empirical_es{int(alpha*100)}'] = es_95_empirical
        else:
            log.warning("Test set log returns empty, cannot calculate empirical VaR/ES.")
            results[f'empirical_var{int(alpha*100)}'] = np.nan
            results[f'empirical_es{int(alpha*100)}'] = np.nan

        log.info(f" Evaluation Horizons: {horizons} days ahead. Model: {final_model_desc}")
        log.info(f" EWMA Lambda (Benchmark): {ewma_lambda}")
        log.info(f" Variance DM Test Loss: {dm_var_loss}")
        t_multi_start = time.time()

        # Use the initial fit on Train+Val for static forecasts
        if initial_arima_fit is None: raise ValueError("Initial static ARIMA/GARCH fit failed, cannot perform horizon evaluation.")
        arima_static_fit = initial_arima_fit
        garch_static_fit = initial_garch_fit
        scale_factor_static = config['default_scale_factor']

        # Setup indices for evaluation period
        full_data_indexed = processed_data.set_index('date').sort_index()
        n_total = len(full_data_indexed)
        test_start_date = test_df['date'].iloc[0] if not test_df.empty else None
        if test_start_date is None: raise ValueError("Test Set is empty, cannot perform horizon evaluation.")
        try:
            test_start_loc = full_data_indexed.index.get_loc(test_start_date)
            test_start_idx = test_start_loc.start if isinstance(test_start_loc, slice) else test_start_loc
        except Exception as e: raise ValueError(f"Cannot find test start index: {e}")

        # Determine the end point for forecast origins to ensure actuals exist for max_horizon
        eval_end_idx = n_total - max_horizon # Last index from which we can forecast max_horizon steps
        n_eval_points = eval_end_idx - test_start_idx # Number of forecast origins in the test set

        if n_eval_points <= 0:
            raise ValueError(f"Test set too short ({len(test_df)}) for max horizon ({max_horizon}). Need at least {max_horizon} points after test start ({test_start_date.date()}).")
        log.info(f"Evaluating forecasts from {n_eval_points} points in the test set (indices {test_start_idx} to {eval_end_idx-1}).")

        # Dictionary to store evaluation tuples for each horizon
        horizon_eval_data = {h: [] for h in horizons}
        fc_generation_errors = 0
        fc_inversion_errors = 0
        ewma_fc_errors = 0

        # --- Loop over forecast origins in the test set ---
        for i in range(n_eval_points):
            current_eval_origin_idx = test_start_idx + i
            current_date = full_data_indexed.index[current_eval_origin_idx]
            step_number = i + 1

            # Log progress
            log_freq=max(1,n_eval_points//10)
            if step_number==1 or step_number%log_freq==0 or step_number==n_eval_points:
                log.info(f"Multi-Horizon Eval: Step {step_number}/{n_eval_points} (Forecasting from {current_date.date()})...")

            # Get actual price at origin (t) needed for naive forecast and price inversion
            actual_price_t = full_data_indexed.iloc[current_eval_origin_idx]['price']

            if pd.isna(actual_price_t):
                 log.warning(f" Step {step_number}: Skipping forecast origin {current_date.date()} due to NaN price.")
                 fc_generation_errors += 1; continue # Skip if origin price is NaN

            # Get recent history for undifferencing if needed
            recent_log_returns_history_h = None
            if d_final > 0:
                 hist_end_idx = current_eval_origin_idx # History up to t-1
                 hist_start_idx = max(0, hist_end_idx - d_final)
                 if hist_end_idx - hist_start_idx == d_final:
                     recent_log_returns_history_h = full_data_indexed['log_return'].iloc[hist_start_idx:hist_end_idx].tolist()
                 else:
                      log.warning(f" Step {step_number}: Not enough history available at {current_date.date()} for undifferencing d={d_final}.")

            # --- Generate AG multi-step forecast from this origin ---
            try:
                 fc_mean_series_ag_h, fc_variance_series_ag_h = forecast_arima_garch(
                      arima_static_fit, garch_static_fit, steps=max_horizon, sf=scale_factor_static
                      )
                 log_return_forecasts_diff_ag_h = fc_mean_series_ag_h.values
                 variance_forecasts_ag_h = fc_variance_series_ag_h.values.copy()

                 # Invert log return forecasts to price level forecasts
                 price_forecasts_ag_h = invert_price_forecast(actual_price_t, log_return_forecasts_diff_ag_h, d_final, recent_log_returns_history_h)


            except Exception as fc_e:
                 log.error(f" Step {step_number}: Error during AG forecast generation/inversion from {current_date.date()}: {type(fc_e).__name__}: {fc_e}", exc_info=False) # Log error type
                 fc_generation_errors += 1
                 continue # Skip to next origin

            # --- Generate EWMA multi-step variance forecast from this origin ---
            try:
                # Use log return history up to time t (inclusive) to calculate EWMA state at t
                history_up_to_t_log_ret = full_data_indexed['log_return'].iloc[:current_eval_origin_idx+1].dropna()
                ewma_var_fc_h = np.full(max_horizon, np.nan) # Initialize forecast array
                ewma_var_state_t = np.nan # State at time t (forecast for all h > 0)

                if len(history_up_to_t_log_ret) >= 2:
                    ewma_series_approx = (history_up_to_t_log_ret**2).ewm(alpha=1-ewma_lambda, adjust=False).mean()
                    if not ewma_series_approx.empty:
                        ewma_var_state_t = ewma_series_approx.iloc[-1] # State at time t
                        ewma_var_state_t = max(0.0, ewma_var_state_t) if pd.notna(ewma_var_state_t) else np.nan
                else:
                     log.debug(f"Step {step_number}: Not enough history ({len(history_up_to_t_log_ret)}) for EWMA forecast from {current_date.date()}.")

                if pd.notna(ewma_var_state_t):
                    ewma_var_fc_h.fill(ewma_var_state_t)

            except Exception as ewma_e:
                 log.error(f" Step {step_number}: Error during EWMA variance forecast calculation from {current_date.date()}: {ewma_e}")
                 ewma_fc_errors += 1
                 ewma_var_fc_h = np.full(max_horizon, np.nan) # Ensure it's NaN on error

            # --- Store forecasts and actuals for each required horizon ---
            for h in horizons:
                 forecast_index_h = h - 1 # 0-based index for forecast arrays
                 actual_index_h = current_eval_origin_idx + h # Index of the actual value at t+h

                 if actual_index_h < n_total: # Check if actual value exists within data bounds
                     actual_price_t_plus_h = full_data_indexed.iloc[actual_index_h]['price']
                     actual_log_return_t_plus_h = full_data_indexed.iloc[actual_index_h]['log_return']
                     actual_var_proxy_t_plus_h = actual_log_return_t_plus_h**2 if pd.notna(actual_log_return_t_plus_h) else np.nan

                     # Get corresponding forecasts
                     fc_price_ag_t_plus_h = price_forecasts_ag_h[forecast_index_h]
                     naive_forecast_t_plus_h = actual_price_t # Naive price forecast is price at origin t
                     fc_var_ag_t_plus_h = variance_forecasts_ag_h[forecast_index_h]
                     fc_var_ewma_t_plus_h = ewma_var_fc_h[forecast_index_h]

                     # Ensure non-negative variance forecasts before calculating VaR/ES
                     fc_var_ag_t_plus_h = max(0, fc_var_ag_t_plus_h) if pd.notna(fc_var_ag_t_plus_h) else np.nan
                     fc_var_ewma_t_plus_h = max(0, fc_var_ewma_t_plus_h) if pd.notna(fc_var_ewma_t_plus_h) else np.nan

                     # Calculate volatility forecasts
                     fc_vol_ag_h = np.sqrt(fc_var_ag_t_plus_h) if pd.notna(fc_var_ag_t_plus_h) else np.nan
                     fc_vol_ewma_h = np.sqrt(fc_var_ewma_t_plus_h) if pd.notna(fc_var_ewma_t_plus_h) else np.nan

                     # Get the h-step ahead UNDIFFERENCED log return forecast mean
                     mean_fc_ag_h_undiff = np.nan
                     if d_final == 0:
                          mean_fc_ag_h_undiff = log_return_forecasts_diff_ag_h[forecast_index_h]
                     else:
                          # Placeholder: Use differenced forecast if d>0, needs proper implementation
                          mean_fc_ag_h_undiff = log_return_forecasts_diff_ag_h[forecast_index_h]
                          # log.debug(f"H={h}: Using differenced logret fc ({mean_fc_ag_h_undiff:.4f}) for VaR calc (approximate if d>0).") # Reduced verbosity


                     # --- VaR/ES Calculation & Violation Tracking for Horizon h ---
                     var_thresh_ag_h, es_ag_h = np.nan, np.nan
                     var_thresh_ewma_h, es_ewma_h = np.nan, np.nan
                     violation_ag_h = np.nan
                     violation_ewma_h = np.nan

                     # Extract distribution parameters from static GARCH fit if needed
                     dist_params_model = None
                     if garch_static_fit is not None:
                          params_garch = getattr(garch_static_fit,'params',{})
                          if chosen_garch_dist_final == 't' and 'nu' in params_garch:
                              dist_params_model = {'nu': params_garch['nu']}
                          elif chosen_garch_dist_final == 'skewt' and 'nu' in params_garch and 'lambda' in params_garch:
                              dist_params_model = {'nu': params_garch['nu'], 'lambda': params_garch['lambda']}

                     # Calculate Model VaR/ES
                     if pd.notna(mean_fc_ag_h_undiff) and pd.notna(fc_vol_ag_h):
                           var_thresh_ag_h, es_ag_h = calculate_parametric_var_es(
                               mean_fc_ag_h_undiff, fc_vol_ag_h, chosen_garch_dist_final, alpha, dist_params_model
                           )

                     # Calculate Benchmark VaR/ES (EWMA + Normal assumption, 0 mean)
                     if pd.notna(fc_vol_ewma_h):
                           var_thresh_ewma_h, es_ewma_h = calculate_parametric_var_es(
                               0.0, fc_vol_ewma_h, 'normal', alpha
                           )

                     # Determine Violations if actual return and VaR threshold are valid
                     if pd.notna(actual_log_return_t_plus_h):
                         if pd.notna(var_thresh_ag_h):
                             violation_ag_h = 1 if actual_log_return_t_plus_h < var_thresh_ag_h else 0
                         if pd.notna(var_thresh_ewma_h):
                             violation_ewma_h = 1 if actual_log_return_t_plus_h < var_thresh_ewma_h else 0


                     # Store results for this horizon, including new VaR/ES data
                     store_flag = all(pd.notna(v) for v in [
                         actual_price_t_plus_h, fc_price_ag_t_plus_h, naive_forecast_t_plus_h,
                         actual_var_proxy_t_plus_h, fc_var_ag_t_plus_h, fc_var_ewma_t_plus_h,
                         var_thresh_ag_h, var_thresh_ewma_h # Only check VaR thresholds for storing basic metrics
                     ]) and actual_var_proxy_t_plus_h >= 0 and fc_var_ag_t_plus_h >= 0 and fc_var_ewma_t_plus_h >= 0

                     if store_flag:
                          horizon_eval_data[h].append((
                              actual_price_t_plus_h, fc_price_ag_t_plus_h, naive_forecast_t_plus_h,
                              actual_var_proxy_t_plus_h, fc_var_ag_t_plus_h, fc_var_ewma_t_plus_h,
                              current_eval_origin_idx, # Keep origin index
                              var_thresh_ag_h, es_ag_h, violation_ag_h, # Model risk metrics
                              var_thresh_ewma_h, es_ewma_h, violation_ewma_h, # Benchmark risk metrics
                              actual_log_return_t_plus_h # Store actual return needed for backtest
                          ))
                     else:
                          # Log only if essential components are missing
                          if pd.isna(actual_price_t_plus_h) or pd.isna(actual_var_proxy_t_plus_h): log.debug(f"Step {step_number}, H={h}: Actual price/var_proxy NaN.")
                          elif pd.isna(fc_price_ag_t_plus_h) or pd.isna(fc_var_ag_t_plus_h) : log.debug(f"Step {step_number}, H={h}: AG price/var forecast NaN.")
                          elif pd.isna(fc_var_ewma_t_plus_h): log.debug(f"Step {step_number}, H={h}: EWMA var forecast NaN.")
                          elif pd.isna(var_thresh_ag_h): log.debug(f"Step {step_number}, H={h}: AG VaR threshold NaN.")
                          elif pd.isna(var_thresh_ewma_h): log.debug(f"Step {step_number}, H={h}: EWMA VaR threshold NaN.")


                 else:
                     log.warning(f" Step {step_number}: Actual index {actual_index_h} is out of bounds for horizon h={h}.")
                     break # No point checking further horizons for this origin
        # --- End of Multi-Horizon Evaluation Loop ---

        # --- Calculate and Report Multi-Horizon Metrics and DM Tests ---
        results["multi_horizon_eval"]["horizons"] = horizons
        results["multi_horizon_eval"]["n_eval_points"] = n_eval_points
        results["multi_horizon_eval"]["forecast_generation_errors"] = fc_generation_errors
        results["multi_horizon_eval"]["forecast_inversion_errors"] = fc_inversion_errors
        results["multi_horizon_eval"]["ewma_forecast_errors"] = ewma_fc_errors

        print(f"\n--- Multi-Horizon Evaluation Results (Test Set, d={d_final}) ---")
        log.info(f"Forecasts generated from {n_eval_points} origin points.")
        if fc_generation_errors > 0: log.warning(f"{fc_generation_errors} errors encountered during AG forecast generation.")
        if fc_inversion_errors > 0: log.warning(f"~{fc_inversion_errors} issues encountered during price forecast inversion (check logs).")
        if ewma_fc_errors > 0: log.warning(f"~{ewma_fc_errors} issues encountered during EWMA forecast calculation (check logs).")

        horizon_metrics_dict = {}
        multi_horizon_results_records = [] # For detailed CSV output
        dm_results_price_horizon = {}
        dm_results_variance_horizon = {}

        for h in horizons:
            eval_tuples_h = horizon_eval_data.get(h, [])
            n_valid_pairs_h = len(eval_tuples_h)

            # Initialize metrics for this horizon
            metrics_h = {
                'mape_ag': np.nan, 'mape_naive': np.nan,
                'rmse_price_ag': np.nan, 'rmse_price_naive': np.nan,
                'rmse_var_ag': np.nan, 'qlike_var_ag': np.nan,
                'rmse_var_ewma': np.nan, 'qlike_var_ewma': np.nan,
                'valid_pairs': n_valid_pairs_h,
                # --- Init VaR/ES/Backtest metrics ---
                'avg_var_ag': np.nan, 'avg_es_ag': np.nan, 'violations_ag': 0, 'kupiec_p_ag': np.nan, 'christ_p_ag': np.nan,
                'avg_var_ewma': np.nan, 'avg_es_ewma': np.nan, 'violations_ewma': 0, 'kupiec_p_ewma': np.nan, 'christ_p_ewma': np.nan,
                'n_backtest_obs': 0

            }
            dm_result_price_h = {'dm_stat': np.nan, 'p_value': np.nan, 'interpretation': 'N/A', 'is_significant': False, 'n_obs': 0, 'error': 'No valid pairs', 'loss_type': dm_loss_type_price}
            dm_result_var_h = {'dm_stat': np.nan, 'p_value': np.nan, 'interpretation': 'N/A', 'is_significant': False, 'n_obs': 0, 'error': 'No valid pairs', 'loss_type': dm_loss_type_var}

            if n_valid_pairs_h > 0:
                 # Unpack data for horizon h 
                 actuals_p_h = np.array([t[0] for t in eval_tuples_h])
                 forecasts_p_ag_h = np.array([t[1] for t in eval_tuples_h])
                 forecasts_p_n_h = np.array([t[2] for t in eval_tuples_h])
                 actuals_v_h = np.array([t[3] for t in eval_tuples_h])
                 forecasts_v_ag_h = np.array([t[4] for t in eval_tuples_h])
                 forecasts_v_ewma_h = np.array([t[5] for t in eval_tuples_h])
                 # --- Unpack VaR/ES/Violation data ---
                 var_thresh_ag_h_all   = np.array([t[7] for t in eval_tuples_h])
                 es_ag_h_all           = np.array([t[8] for t in eval_tuples_h])
                 violations_ag_h_all   = np.array([t[9] for t in eval_tuples_h])
                 var_thresh_ewma_h_all = np.array([t[10] for t in eval_tuples_h])
                 es_ewma_h_all         = np.array([t[11] for t in eval_tuples_h])
                 violations_ewma_h_all = np.array([t[12] for t in eval_tuples_h])
                 actual_returns_h      = np.array([t[13] for t in eval_tuples_h])

                 # Calculate Price Metrics
                 metrics_h['mape_ag'] = mean_absolute_percentage_error(actuals_p_h, forecasts_p_ag_h)
                 metrics_h['mape_naive'] = mean_absolute_percentage_error(actuals_p_h, forecasts_p_n_h)
                 metrics_h['rmse_price_ag'] = root_mean_squared_error(actuals_p_h, forecasts_p_ag_h)
                 metrics_h['rmse_price_naive'] = root_mean_squared_error(actuals_p_h, forecasts_p_n_h)

                 # Calculate Variance Metrics
                 metrics_h['rmse_var_ag'] = root_mean_squared_error_variance(actuals_v_h, forecasts_v_ag_h)
                 metrics_h['qlike_var_ag'] = qlike_loss(actuals_v_h, forecasts_v_ag_h, epsilon=qlike_eps)
                 metrics_h['rmse_var_ewma'] = root_mean_squared_error_variance(actuals_v_h, forecasts_v_ewma_h)
                 metrics_h['qlike_var_ewma'] = qlike_loss(actuals_v_h, forecasts_v_ewma_h, epsilon=qlike_eps)

                 print(f"  Horizon h={h:<2}: Valid Pairs={n_valid_pairs_h:<5}")
                 print(f"    Price  MAPE AG={metrics_h['mape_ag']:>8.2f}% | MAPE Naive={metrics_h['mape_naive']:>8.2f}%")
                 print(f"    Var    RMSE AG={metrics_h['rmse_var_ag']:.8f} | QLIKE AG={metrics_h['qlike_var_ag']:.6f}")
                 print(f"    Var    RMSE EWMA={metrics_h['rmse_var_ewma']:.8f} | QLIKE EWMA={metrics_h['qlike_var_ewma']:.6f}")

                 # --- VaR Backtesting for Horizon h ---
                 violations_ag_series = violations_ag_h_all[pd.notna(violations_ag_h_all)].astype(int)
                 violations_ewma_series = violations_ewma_h_all[pd.notna(violations_ewma_h_all)].astype(int)
                 n_backtest_ag = len(violations_ag_series)
                 n_backtest_ewma = len(violations_ewma_series)
                 metrics_h['n_backtest_obs'] = n_backtest_ag # Use AG count as reference

                 kupiec_ag = {'p_value':np.nan, 'LR_stat':np.nan, 'error':None}
                 christ_ag = {'p_value':np.nan, 'LR_stat':np.nan, 'error':None}
                 kupiec_ewma = {'p_value':np.nan, 'LR_stat':np.nan, 'error':None}
                 christ_ewma = {'p_value':np.nan, 'LR_stat':np.nan, 'error':None}

                 print(f"    VaR Backtest (h={h}, alpha={alpha:.2%}):")
                 if n_backtest_ag >= min_resid_len_diag: # Check if enough observations for tests
                      n_violations_ag = int(np.sum(violations_ag_series))
                      metrics_h[f'violations_ag'] = n_violations_ag
                      print(f"      AG Model: N={n_backtest_ag} | Violations={n_violations_ag} (Expected: {n_backtest_ag*alpha:.1f})")
                      kupiec_ag = kupiec_test(n_violations_ag, n_backtest_ag, alpha)
                      christ_ag = christoffersen_test(violations_ag_series, alpha)
                      metrics_h[f'kupiec_p_ag'] = kupiec_ag.get('p_value')
                      metrics_h[f'christ_p_ag'] = christ_ag.get('p_value')
                      print(f"        Kupiec p={metrics_h[f'kupiec_p_ag']:.4f} | Christoffersen p={metrics_h[f'christ_p_ag']:.4f}")
                 else:
                      print(f"      AG Model: N={n_backtest_ag} | Skipped (Too few obs < {min_resid_len_diag})")

                 if n_backtest_ewma >= min_resid_len_diag:
                      n_violations_ewma = int(np.sum(violations_ewma_series))
                      metrics_h[f'violations_ewma'] = n_violations_ewma
                      print(f"      EWMA Bench: N={n_backtest_ewma} | Violations={n_violations_ewma} (Expected: {n_backtest_ewma*alpha:.1f})")
                      kupiec_ewma = kupiec_test(n_violations_ewma, n_backtest_ewma, alpha)
                      christ_ewma = christoffersen_test(violations_ewma_series, alpha) # This test is technically possible for EWMA, might show dependence
                      metrics_h[f'kupiec_p_ewma'] = kupiec_ewma.get('p_value')
                      metrics_h[f'christ_p_ewma'] = christ_ewma.get('p_value') # Store Christoffersen for EWMA too
                      print(f"        Kupiec p={metrics_h[f'kupiec_p_ewma']:.4f} | Christoffersen p={metrics_h[f'christ_p_ewma']:.4f}") # Print Christoffersen for EWMA
                 else:
                      print(f"      EWMA Bench: N={n_backtest_ewma} | Skipped (Too few obs < {min_resid_len_diag})")


                 # Calculate average VaR/ES for reporting
                 metrics_h[f'avg_var_ag'] = np.nanmean(var_thresh_ag_h_all)
                 metrics_h[f'avg_es_ag'] = np.nanmean(es_ag_h_all)
                 metrics_h[f'avg_var_ewma'] = np.nanmean(var_thresh_ewma_h_all)
                 metrics_h[f'avg_es_ewma'] = np.nanmean(es_ewma_h_all)
                 print(f"      Avg VaR AG={metrics_h[f'avg_var_ag']:.4f}, ES AG={metrics_h[f'avg_es_ag']:.4f}")
                 print(f"      Avg VaR EWMA={metrics_h[f'avg_var_ewma']:.4f}, ES EWMA={metrics_h[f'avg_es_ewma']:.4f}")
                 # --- VaR Backtesting ---


                 # --- Keep existing DM Tests ---
                 # Perform DM Test for Price (AG vs Naive)
                 if DIEBOLDMARIANO_LIB_AVAILABLE:
                      dm_test_p_output_h = diebold_mariano_test(actuals_p_h, forecasts_p_ag_h, forecasts_p_n_h, h=h, loss_type=dm_loss_type_price, alternative='less')
                      dm_result_price_h.update(dm_test_p_output_h) # Update dict with test results
                      # --- DM Interpretation ---
                      if dm_test_p_output_h['error'] is None and pd.notna(dm_test_p_output_h['p_value']):
                          p_val, dm_stat = dm_test_p_output_h['p_value'], dm_test_p_output_h['dm_stat']
                          if p_val < alpha: # Verwende die oben definierte Variable alpha
                              if dm_stat < 0: dm_result_price_h['interpretation'] = f"AG better (p={p_val:.4f})"
                              else: dm_result_price_h['interpretation'] = f"Sig. (p={p_val:.4f}), but Stat>=0"
                              dm_result_price_h['is_significant'] = True
                          else: dm_result_price_h['interpretation'] = f"No Diff. (p={p_val:.4f})"
                      elif dm_test_p_output_h['error']:
                         dm_result_price_h['interpretation'] = f"Error: {dm_test_p_output_h['error']}"
                      else: # NaN result
                         dm_result_price_h['interpretation'] = "NaN result"

                      print(f"    DM Price (h={h}): N={dm_result_price_h['n_obs']:<5} | Loss='{dm_loss_type_price}' | p={dm_result_price_h['p_value']:>8.4f} | {dm_result_price_h['interpretation']}")
                 else:
                      print(f"    DM Price (h={h}): Skipped (library missing)")
                      dm_result_price_h['error'] = "Library missing"

                 # Perform DM Test for Variance (AG vs EWMA)
                 if DIEBOLDMARIANO_LIB_AVAILABLE:
                      eps_arg_dm_h = {'qlike_epsilon': qlike_eps} if dm_loss_type_var == "QLIKE" else {}
                      dm_test_v_output_h = diebold_mariano_test(actuals_v_h, forecasts_v_ag_h, forecasts_v_ewma_h, h=h, loss_type=dm_loss_type_var, alternative='less', **eps_arg_dm_h)
                      dm_result_var_h.update(dm_test_v_output_h)
                      # --- DM Interpretation ---
                      if dm_test_v_output_h['error'] is None and pd.notna(dm_test_v_output_h['p_value']):
                          p_val, dm_stat = dm_test_v_output_h['p_value'], dm_test_v_output_h['dm_stat']
                          if p_val < alpha:
                              if dm_stat < 0: dm_result_var_h['interpretation'] = f"AG better (p={p_val:.4f})"
                              else: dm_result_var_h['interpretation'] = f"Sig. (p={p_val:.4f}), but Stat>=0"
                              dm_result_var_h['is_significant'] = True
                          else: dm_result_var_h['interpretation'] = f"No Diff. (p={p_val:.4f})"
                      elif dm_test_v_output_h['error']:
                         dm_result_var_h['interpretation'] = f"Error: {dm_test_v_output_h['error']}"
                      else: # NaN result
                         dm_result_var_h['interpretation'] = "NaN result"

                      print(f"    DM Var   (h={h}): N={dm_result_var_h['n_obs']:<5} | Loss='{dm_loss_type_var}' | p={dm_result_var_h['p_value']:>8.4f} | {dm_result_var_h['interpretation']}")
                 else:
                      print(f"    DM Var   (h={h}): Skipped (library missing)")
                      dm_result_var_h['error'] = "Library missing"


                 # Store detailed results for CSV export (Adjust tuple indices)
                 origin_indices_h = [t[6] for t in eval_tuples_h] # index 6 = origin_index
                 valid_origin_indices_h = [idx for idx in origin_indices_h if 0 <= idx < len(full_data_indexed.index)]
                 valid_actual_indices_h = [idx + h for idx in valid_origin_indices_h if 0 <= idx + h < len(full_data_indexed.index)]

                 if len(valid_origin_indices_h) == n_valid_pairs_h and len(valid_actual_indices_h) == n_valid_pairs_h:
                      origin_dates_h = full_data_indexed.index[valid_origin_indices_h]
                      actual_dates_h = full_data_indexed.index[valid_actual_indices_h]
                      for idx in range(n_valid_pairs_h):
                           # Ensure the correct indices based on the tuple structure defined earlier
                           t_data = eval_tuples_h[idx]
                           multi_horizon_results_records.append({
                               'horizon': h,
                               'forecast_origin_date': origin_dates_h[idx].strftime('%Y-%m-%d'),
                               'actual_date': actual_dates_h[idx].strftime('%Y-%m-%d'),
                               'actual_price': t_data[0],
                               'fc_price_ag': t_data[1],
                               'fc_price_naive': t_data[2],
                               'actual_var_proxy': t_data[3],
                               'fc_var_ag': t_data[4],
                               'fc_var_ewma': t_data[5],
                               'actual_log_return': t_data[13], # Actual Log Return
                               'var_thresh_ag': t_data[7],
                               'es_ag': t_data[8],
                               'violation_ag': t_data[9],
                               'var_thresh_ewma': t_data[10],
                               'es_ewma': t_data[11],
                               'violation_ewma': t_data[12],
                           })
                 else:
                     log.warning(f"Index length mismatch for H={h}. Skipping detailed CSV rows for this horizon.")


            else: # n_valid_pairs_h == 0
                 print(f"  Horizon h={h:<2}: Valid Pairs=0 | Metrics N/A")
                 print(f"    DM Tests (h={h}): N/A")
                 print(f"    VaR Backtest (h={h}): N/A")


            # Store results for this horizon
            horizon_metrics_dict[f"h{h}"] = metrics_h # metrics_h now contains VaR/ES/Backtest results
            dm_results_price_horizon[f"h{h}_dm_p_test"] = dm_result_price_h
            dm_results_variance_horizon[f"h{h}_dm_v_test"] = dm_result_var_h


        # Store aggregated horizon results
        results["multi_horizon_eval"]["metrics"] = horizon_metrics_dict
        results["multi_horizon_eval"]["price_dm_tests"] = dm_results_price_horizon
        results["multi_horizon_eval"]["variance_dm_tests"] = dm_results_variance_horizon


        # --- Generate Thesis Recommendations based on DM tests and Backtests ---
        # Define alpha here, inheriting from the main config or backtest block if preferred
        alpha_reco = config.get("dm_test_alpha", 0.05)
        dm_loss_price_reco = config.get("dm_test_loss_type", "Squared Error")
        dm_loss_var_reco = config.get("dm_test_variance_loss_type", "QLIKE")

        # --- Price DM Recommendation (Horizon) ---
        reco_price_text = f"Diebold-Mariano Test (Price, AG vs Naive, alpha={alpha_reco:.2%}, H1: AG better, Loss={dm_loss_price_reco}):\n"
        if not DIEBOLDMARIANO_LIB_AVAILABLE: reco_price_text += "- Tests skipped (library missing).\n"
        else:
            sig_p_horizons = [h for h in horizons if dm_results_price_horizon.get(f"h{h}_dm_p_test",{}).get('is_significant',False) and dm_results_price_horizon.get(f"h{h}_dm_p_test",{}).get('dm_stat',0)<0]
            ns_p_horizons = [h for h in horizons if not dm_results_price_horizon.get(f"h{h}_dm_p_test",{}).get('is_significant',False) and dm_results_price_horizon.get(f"h{h}_dm_p_test",{}).get('error') is None]
            err_p_horizons = [h for h in horizons if dm_results_price_horizon.get(f"h{h}_dm_p_test",{}).get('error') is not None]
            if sig_p_horizons: reco_price_text += f"- AG significantly better for horizons: {sig_p_horizons}.\n"
            if ns_p_horizons: reco_price_text += f"- No significant difference for horizons {ns_p_horizons} (p >= {alpha_reco:.2f}).\n"
            if err_p_horizons: reco_price_text += f"- Test errors occurred for horizons {err_p_horizons}.\n"
            if not sig_p_horizons and not ns_p_horizons and not err_p_horizons: reco_price_text += "- No valid tests performed or all failed.\n"
        # Store using the SHORT key
        results["thesis_reco_price_dm"] = reco_price_text

        # --- Variance DM Recommendation (Horizon) ---
        reco_var_text = f"Diebold-Mariano Test (Variance, AG vs EWMA, alpha={alpha_reco:.2%}, H1: AG better, Loss={dm_loss_var_reco}):\n"
        if not DIEBOLDMARIANO_LIB_AVAILABLE: reco_var_text += "- Tests skipped (library missing).\n"
        else:
            sig_v_horizons = [h for h in horizons if dm_results_variance_horizon.get(f"h{h}_dm_v_test",{}).get('is_significant',False) and dm_results_variance_horizon.get(f"h{h}_dm_v_test",{}).get('dm_stat',0)<0]
            ns_v_horizons = [h for h in horizons if not dm_results_variance_horizon.get(f"h{h}_dm_v_test",{}).get('is_significant',False) and dm_results_variance_horizon.get(f"h{h}_dm_v_test",{}).get('error') is None]
            err_v_horizons = [h for h in horizons if dm_results_variance_horizon.get(f"h{h}_dm_v_test",{}).get('error') is not None]
            if sig_v_horizons: reco_var_text += f"- AG significantly better for horizons: {sig_v_horizons}.\n"
            if ns_v_horizons: reco_var_text += f"- No significant difference for horizons {ns_v_horizons} (p >= {alpha_reco:.2f}).\n"
            if err_v_horizons: reco_var_text += f"- Test errors occurred for horizons {err_v_horizons}.\n"
            if not sig_v_horizons and not ns_v_horizons and not err_v_horizons: reco_var_text += "- No valid tests performed or all failed.\n"
        # Store using the SHORT key
        results["thesis_reco_var_dm"] = reco_var_text

        # --- VaR Backtest Recommendation (Horizon) ---
        reco_var_backtest_text = f"VaR Backtesting (AG vs EWMA, alpha={alpha_reco:.2%}):\n"
        kupiec_reject_ag = []
        kupiec_reject_ewma = []
        christ_reject_ag = []
        christ_reject_ewma = []
        for h in horizons:
            metrics_h_reco = horizon_metrics_dict.get(f"h{h}",{})
            # Check for existence and value before appending
            kup_p_ag_val = metrics_h_reco.get('kupiec_p_ag')
            if pd.notna(kup_p_ag_val) and kup_p_ag_val < alpha_reco: kupiec_reject_ag.append(h)

            kup_p_ewma_val = metrics_h_reco.get('kupiec_p_ewma')
            if pd.notna(kup_p_ewma_val) and kup_p_ewma_val < alpha_reco: kupiec_reject_ewma.append(h)

            chr_p_ag_val = metrics_h_reco.get('christ_p_ag')
            if pd.notna(chr_p_ag_val) and chr_p_ag_val < alpha_reco: christ_reject_ag.append(h)

            # Christoffersen test might be NaN for EWMA, so check is important
            chr_p_ewma_val = metrics_h_reco.get('christ_p_ewma')
            if pd.notna(chr_p_ewma_val) and chr_p_ewma_val < alpha_reco: christ_reject_ewma.append(h)


        reco_var_backtest_text += f"- Kupiec Test rejects correct AG coverage for horizons: {kupiec_reject_ag if kupiec_reject_ag else 'None'}.\n"
        reco_var_backtest_text += f"- Christoffersen Test rejects correct AG coverage/independence for horizons: {christ_reject_ag if christ_reject_ag else 'None'}.\n"
        reco_var_backtest_text += f"- Kupiec Test rejects correct EWMA coverage for horizons: {kupiec_reject_ewma if kupiec_reject_ewma else 'None'}.\n"
        if christ_reject_ewma: # Only mention if Christoffersen rejects for EWMA
             reco_var_backtest_text += f"- Christoffersen Test rejects EWMA independence for horizons: {christ_reject_ewma}.\n"
        # Store using the SHORT key
        results["thesis_reco_var_backtest"] = reco_var_backtest_text

        # Print recommendations during the run
        print("\n--- Thesis Recommendations Summary (during run_analysis) ---")
        print(" Price DM Test (AG vs Naive):")
        print(reco_price_text)
        print(" Variance DM Test (AG vs EWMA):")
        print(reco_var_text)
        print(" VaR Backtesting:")
        print(reco_var_backtest_text)

        results['multi_horizon_duration_min'] = (time.time() - t_multi_start) / 60
        log.info(f"--- Multi-Horizon Evaluation Finished: {results['multi_horizon_duration_min']:.2f} min ---")

        # Save detailed horizon results to CSV
        try:
             if multi_horizon_results_records:
                 # Create DataFrame from list of dicts
                 mh_df = pd.DataFrame(multi_horizon_results_records)
                 # Define column order for better readability
                 col_order = [
                      'horizon', 'forecast_origin_date', 'actual_date',
                      'actual_price', 'fc_price_ag', 'fc_price_naive',
                      'actual_log_return', # Added actual log return
                      'actual_var_proxy', 'fc_var_ag', 'fc_var_ewma',
                      'var_thresh_ag', 'es_ag', 'violation_ag',
                      'var_thresh_ewma', 'es_ewma', 'violation_ewma'
                 ]
                 # Ensure all desired columns exist before reordering
                 mh_df = mh_df[[col for col in col_order if col in mh_df.columns]]
                 mh_csv_path = os.path.join(coin_plot_dir, f"{coin_id}_multi_horizon_eval_details_d{d_final}.csv")
                 mh_df.to_csv(mh_csv_path, index=False, float_format="%.8f")
                 log.info(f"Multi-horizon details saved: {mh_csv_path}")
                 results["multi_horizon_eval_csv_path"] = mh_csv_path
             else:
                 log.warning("No valid pairs found for multi-horizon details CSV.")
                 results["multi_horizon_eval_csv_path"] = "N/A"
        except Exception as e:
            log.error(f"Saving multi-horizon results CSV failed: {e}")
            results["multi_horizon_eval_csv_path"] = f"Error: {e}"

    # --- Mode 3: Future Forecast ('future') ---
    elif forecast_mode == 'future':
        
        log.info(f"--- [{coin_id}] Starting Future Forecast ({config['future_forecast_horizon']} steps) ---")
        results["future_forecast"]={}
        # Use the initial fit on Train+Val
        if initial_arima_fit is None: raise ValueError("Initial static ARIMA/GARCH fit failed, cannot perform future forecast.")
        arima_future = initial_arima_fit
        garch_future = initial_garch_fit
        scale_factor_future = config['default_scale_factor']

        try:
            # Get last actual price and date from the fit_base_df (Train+Val)
            last_actual_price = fit_base_df['price'].iloc[-1]
            last_actual_date = fit_base_df['date'].iloc[-1]
            log.info(f"Forecasting {config['future_forecast_horizon']} steps from {last_actual_date.date()} (Last Price: {last_actual_price:.4f}). Using model: {final_model_desc}")

            # Get recent history for undifferencing if needed
            recent_log_returns_history_fut = None
            if d_final > 0:
                 if len(fit_base_df) >= d_final:
                     recent_log_returns_history_fut = fit_base_df['log_return'].iloc[-d_final:].tolist()
                 else:
                     log.warning(f"Future Forecast: Not enough history ({len(fit_base_df)} < {d_final}) for undifferencing d={d_final}. Price forecast might be inaccurate.")

            # Generate forecast
            horizon_fut = config['future_forecast_horizon']
            fc_mean_series_fut, fc_variance_series_fut=forecast_arima_garch(arima_future, garch_future, horizon_fut, scale_factor_future)

            # Extract forecasts
            fc_log_return_diff_fut=fc_mean_series_fut.values
            fc_variance_fut=fc_variance_series_fut.values
            # Calculate volatility
            fc_volatility_fut = np.sqrt(np.maximum(0, fc_variance_fut)) # Ensure non-negative before sqrt

            # Invert to price level
            fc_price_fut = invert_price_forecast(last_actual_price, fc_log_return_diff_fut, d_final, recent_log_returns_history_fut)

            # Create future dates index
            future_dates=pd.date_range(start=last_actual_date+timedelta(days=1),periods=horizon_fut,freq='D')

            # Create DataFrame
            fc_df_future=pd.DataFrame({'date':future_dates, 'fc_volatility':fc_volatility_fut, 'fc_variance':fc_variance_fut, 'fc_price':fc_price_fut}).set_index('date')

            # Store results
            results['future_forecast']['data']=fc_df_future.reset_index().to_dict('records')
            results['future_forecast']['last_actual_date']=last_actual_date.strftime('%Y-%m-%d')
            results['future_forecast']['last_actual_price']=last_actual_price
            log.info("Future forecast generated.")

            # Save future forecast to CSV
            try:
                future_csv_filename=os.path.join(coin_plot_dir,f"{coin_id}_future_fc_d{d_final}.csv")
                fc_df_future.to_csv(future_csv_filename,index=True,float_format="%.8f")
                log.info(f"Future forecast CSV saved: {future_csv_filename}")
                results["future_forecast_csv_path"]=future_csv_filename
            except Exception as e:
                log.error(f"Saving future forecast CSV failed: {e}")
                results["future_forecast_csv_path"]=f"Failed: {e}"

            # Plot future forecast if enabled
            if is_plot_available and plt is not None:
                  try:
                      fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
                      # Get recent history for context
                      history_points=90
                      history_data = fit_base_df.set_index('date').iloc[-history_points:]

                      # Price Plot
                      axes[0].plot(history_data.index, history_data['price'], label='Historical Price',color='gray',lw=1.2)
                      if not fc_df_future.empty and 'fc_price' in fc_df_future.columns:
                           axes[0].plot(fc_df_future.index,fc_df_future['fc_price'],label='Forecast Price',color='red',marker='.',ls='-',lw=1.5)
                      axes[0].set_title(f'{coin_id} Future Price Forecast (d={d_final})\n({final_model_desc})')
                      axes[0].set_ylabel("Price")
                      axes[0].legend(); axes[0].grid(True,ls=':')

                      # Volatility Plot
                      # Use absolute log return as historical volatility proxy
                      hist_vol_proxy = np.abs(history_data['log_return'])
                      axes[1].plot(history_data.index, hist_vol_proxy, label='Historical |LogRet|',color='purple',alpha=0.6,lw=1.2)
                      if not fc_df_future.empty and 'fc_volatility' in fc_df_future.columns:
                           axes[1].plot(fc_df_future.index,fc_df_future['fc_volatility'],label='Forecast Volatility',color='orange',marker='.',ls='-',lw=1.5)
                      axes[1].set_title('Future Volatility Forecast')
                      axes[1].set_ylabel("Daily Volatility")
                      axes[1].legend(); axes[1].grid(True,ls=':')

                      plt.setp(axes[1].xaxis.get_majorticklabels(),rotation=45,ha='right')
                      fig.tight_layout()
                      future_plot_filename=os.path.join(coin_plot_dir,f"{coin_id}_future_plot_d{d_final}.png")
                      fig.savefig(future_plot_filename); plt.close(fig)
                      log.info(f"Future forecast plot saved: {future_plot_filename}")
                  except Exception as e:
                      log.error(f"Plotting future forecast failed: {e}"); plt.close('all')

        except Exception as future_e:
            log.critical(f"Future forecast generation failed: {future_e}", exc_info=True)
            results["status"]="ERROR"
            results["error_message"]=f"Future Forecast Error: {future_e}"
            results["future_forecast"]["error"]=str(future_e)


    # --- Unknown Forecast Mode ---
    else:
        log.error(f"Unknown forecast_mode specified: {forecast_mode}")
        results["status"]="ERROR"
        results["error_message"]=f"Invalid forecast_mode: {forecast_mode}"

    # --- Finalize Coin Analysis ---
    run_end_time=time.time()
    results['total_runtime_min']=(run_end_time-start_time_coin)/60
    log.info(f"========== {coin_id.upper()} Analysis Finished in {results['total_runtime_min']:.2f} min (Status: {results.get('status','UNKNOWN')}) ==========")
    current_coin_filter.coin_id="N/A" # Reset logger context
    return results

# ==============================================================================
# --- MAIN EXECUTION SCRIPT ---
# ==============================================================================

def main(): # Wrap the main execution logic in a function
    global PLOT_AVAILABLE # Allow modification of the global variable
    overall_start_time = time.time()
    log.info(f"--- Starting Multi-Coin ARIMA-GARCH Analysis (v28.5 - English Output / Vertical Summary) ---")
    config_main = copy.deepcopy(CONFIG)
    config_main['plot_available'] = PLOT_AVAILABLE

    # --- Define alpha here for use in summary key creation ---
    alpha_summary = config_main.get("dm_test_alpha", 0.05) # Use config alpha

    # --- Validate config specific to forecast mode again at top level ---
    if config_main['forecast_mode'] == 'horizon_evaluation':
        if not config_main.get('evaluation_horizons'):
            log.critical("Error: 'horizon_evaluation' mode requires 'evaluation_horizons' to be set in config.")
            exit(1)
        valid_horizons_main = [h for h in config_main.get('evaluation_horizons', []) if isinstance(h, int) and h > 0]
        if not valid_horizons_main:
            log.critical("Error: 'evaluation_horizons' contains no valid positive integers.")
            exit(1)
        config_main['evaluation_horizons'] = sorted(list(set(valid_horizons_main))) # Store sorted unique horizons
        current_max_horizon_main = max(config_main['evaluation_horizons']) if config_main['evaluation_horizons'] else 0
        min_required_test_size_main = current_max_horizon_main + 10
        if config_main['min_test_set_size'] < min_required_test_size_main:
             log.warning(f"MAIN CHECK: min_test_set_size ({config_main['min_test_set_size']}) too small for max horizon ({current_max_horizon_main}). Setting to {min_required_test_size_main}.")
             config_main['min_test_set_size'] = min_required_test_size_main
    elif config_main['forecast_mode'] == 'backtest':
        if config_main.get('fitting_window_size', 0) <= 0:
            log.critical("Error: 'backtest' mode requires a valid 'fitting_window_size' > 0.")
            exit(1)
        if config_main.get('refit_interval', 0) <= 0:
            log.critical("Error: 'backtest' mode requires a valid 'refit_interval' >= 1.")
            exit(1)
    elif config_main['forecast_mode'] == 'future':
        if config_main.get('future_forecast_horizon', 0) <= 0:
            log.critical("Error: 'future' mode requires a valid 'future_forecast_horizon' > 0.")
            exit(1)

    # --- Log key configurations ---
    log.info(f"Analyzing Coins: {', '.join(config_main['coins_to_analyze'])}")
    log.info(f"Forecast Mode: {config_main['forecast_mode']}")
    if config_main['forecast_mode'] == 'backtest':
        log.info(f" Backtest Params: Window={config_main['fitting_window_size']} days, Refit Interval={config_main['refit_interval']} days")
        log.info(f" DM Test (Price, AG vs Naive): Loss='{config_main.get('dm_test_loss_type','Squared Error')}', Alpha={config_main.get('dm_test_alpha',0.05)}")
        log.info(f" DM Test (Variance, AG vs EWMA): Loss='{config_main.get('dm_test_variance_loss_type','QLIKE')}', Alpha={config_main.get('dm_test_alpha',0.05)}")
        log.info(f" Risk Metric Alpha: {config_main.get('dm_test_alpha', 0.05)}")
    elif config_main['forecast_mode'] == 'horizon_evaluation':
        log.info(f" Horizon Eval Params: Horizons={config_main['evaluation_horizons']}")
        log.info(f" Horizon Eval Params: Min Test Set Size={config_main['min_test_set_size']}")
        log.info(f" DM Test (Price, AG vs Naive): Loss='{config_main.get('dm_test_loss_type','Squared Error')}', Alpha={config_main.get('dm_test_alpha',0.05)}")
        log.info(f" DM Test (Variance, AG vs EWMA): Loss='{config_main.get('dm_test_variance_loss_type','QLIKE')}', Alpha={config_main.get('dm_test_alpha',0.05)}")
        log.info(f" Risk Metric Alpha: {config_main.get('dm_test_alpha', 0.05)}")
    elif config_main['forecast_mode'] == 'future':
        log.info(f" Future Forecast Params: Horizon={config_main['future_forecast_horizon']} days")

    if not DIEBOLDMARIANO_LIB_AVAILABLE and config_main['forecast_mode'] != 'future':
        log.warning("DM Tests will be skipped because 'dieboldmariano' library is not installed.")
    if not scipy_chi2_available and config_main['forecast_mode'] != 'future':
        log.warning("VaR Backtests (Kupiec/Christoffersen) will be skipped because 'scipy.stats.chi2' is not available.")

    log.info(f"Auto-Tune Enabled: {config_main['use_auto_tune']} (Criterion: {config_main['tune_criterion']}, Sig. Level: {config_main['param_significance_level']})")
    log.info(f" GARCH Types for Tuning: {config_main.get('garch_types_to_tune')}")
    log.info(f"Max Differencing Order: {config_main['max_differencing_order']}")
    log.info(f"EWMA Lambda: {config_main['ewma_lambda']}")
    log.info(f"Output Directory Base: {config_main['plot_dir_base']}")

    # --- Create base output directory if it doesn't exist ---
    if PLOT_AVAILABLE and not os.path.exists(config_main['plot_dir_base']):
        try:
            os.makedirs(config_main['plot_dir_base'], exist_ok=True)
            log.info(f"Base output directory created: {os.path.abspath(config_main['plot_dir_base'])}")
        except Exception as e:
            log.error(f"Could not create base output directory: {e}. Plotting may fail.")
            PLOT_AVAILABLE = False
            config_main['plot_available'] = False

    # --- Run Analysis for Each Coin ---
    all_coin_results = []
    coins_to_process = config_main.get("coins_to_analyze", [])
    if not coins_to_process:
        log.critical("No coins specified in 'coins_to_analyze'. Exiting.")
        exit(1)

    for coin in coins_to_process:
        log.info(f"===== Processing Coin: {coin.upper()} =====")
        try:
            # Ensure run_analysis_for_coin adds necessary keys
            coin_result = run_analysis_for_coin(coin, config_main)
            if coin_result:
                 all_coin_results.append(coin_result)
            else:
                 log.warning(f"No result returned for {coin}.")
                 all_coin_results.append({'coin_id':coin,'status':'FAILED','error_message':'Analysis function returned None'})
        except Exception as e:
            log.critical(f"!!! UNHANDLED ERROR processing {coin}: {type(e).__name__}: {e} !!!", exc_info=True)
            error_result = {
                "coin_id":coin,
                "status":"CRITICAL_ERROR",
                "error_message":f"Unhandled Exception: {type(e).__name__}: {e}",
                "total_runtime_min":(time.time()-overall_start_time)/60
            }
            all_coin_results.append(error_result)
            current_coin_filter.coin_id = "N/A"

    # --- Aggregate and Summarize Results ---
    log.info("--- Aggregating Results from All Coins ---")
    if not all_coin_results:
        log.warning("No results were generated for any coin.")
    else:
        # --- Start of aggregation block ---
        results_df_raw = pd.DataFrame()
        try:
            # --- FLATTENING CODE  ---
            processed_results_list = []
            for res in all_coin_results:
                flat_res = res.copy()

                # --- Extract nested dictionaries using pop ---
                desc_stats = flat_res.pop('descriptive_stats_full', {})
                benchmarks = flat_res.pop('benchmarks', {})
                variance_dm = flat_res.pop('variance_dm', {}) # Contains 'backtest' and 'horizon' keys
                mh_eval = flat_res.pop('multi_horizon_eval', {}) # Contains 'metrics', 'price_dm_tests', 'variance_dm_tests'
                roll_fc_data = flat_res.pop('roll_fc', {}) # Contains backtest metrics like 'bt_avg_var_ag'
                roll_fc_price_dm_test = flat_res.pop('roll_fc_price_dm_test', {})
                roll_fc_logret_dm_test = flat_res.pop('roll_fc_logret_dm_test', {})
                flat_res.pop('future_forecast', None) # Remove future forecast data if present

                # --- Flatten Descriptive Stats ---
                if isinstance(desc_stats, dict):
                    for k, v in desc_stats.items():
                        flat_res[f"desc_{k}"] = v

                # --- Flatten Benchmark Results ---
                ewma_var_metrics = benchmarks.get('ewma_var', {})
                flat_res['bm_ewma_rmse'] = ewma_var_metrics.get('rmse') 
                flat_res['bm_ewma_qlike'] = ewma_var_metrics.get('qlike')

                # --- Flatten Backtest-Specific Results ---
                # Variance DM Test Results (Backtest Mode)
                bt_var_dm = variance_dm.get('backtest', {})
                flat_res['bt_dm_v_stat'] = bt_var_dm.get('dm_stat')
                flat_res['bt_dm_v_pval'] = bt_var_dm.get('p_value')
                flat_res['bt_dm_v_sig'] = bt_var_dm.get('is_significant')
                flat_res['bt_dm_v_n'] = bt_var_dm.get('n_obs')
                flat_res['bt_dm_v_loss'] = bt_var_dm.get('loss_type')
                flat_res['bt_dm_v_err'] = bt_var_dm.get('error')

                # Backtest Price DM Test Results (popped earlier)
                if isinstance(roll_fc_price_dm_test, dict):
                    flat_res["bt_dm_p_stat"] = roll_fc_price_dm_test.get('dm_stat')
                    flat_res["bt_dm_p_pval"] = roll_fc_price_dm_test.get('p_value')
                    flat_res["bt_dm_p_sig"] = roll_fc_price_dm_test.get('is_significant')
                    flat_res["bt_dm_p_n"] = roll_fc_price_dm_test.get('n_obs')
                    flat_res["bt_dm_p_loss"] = roll_fc_price_dm_test.get('loss_type')
                    flat_res["bt_dm_p_err"] = roll_fc_price_dm_test.get('error')

                # Backtest LogRet DM Test Results (popped earlier)
                if isinstance(roll_fc_logret_dm_test, dict):
                    flat_res["bt_dm_lr_stat"] = roll_fc_logret_dm_test.get('dm_stat')
                    flat_res["bt_dm_lr_pval"] = roll_fc_logret_dm_test.get('p_value')
                    flat_res["bt_dm_lr_sig"] = roll_fc_logret_dm_test.get('is_significant')
                    flat_res["bt_dm_lr_n"] = roll_fc_logret_dm_test.get('n_obs')
                    flat_res["bt_dm_lr_loss"] = roll_fc_logret_dm_test.get('loss_type')
                    flat_res["bt_dm_lr_err"] = roll_fc_logret_dm_test.get('error')

                # Other Backtest Metrics (from roll_fc_data or top-level res)
                bt_keys_to_flatten = [
                    # VaR/ES AG
                    'backtest_avg_var_ag', 'backtest_avg_es_ag', 'backtest_violations_ag',
                    'backtest_kupiec_p_ag', 'backtest_christ_p_ag', 'backtest_kupiec_stat_ag', 'backtest_christ_stat_ag',
                    # VaR/ES EWMA
                    'backtest_avg_var_ewma', 'backtest_avg_es_ewma', 'backtest_violations_ewma',
                    'backtest_kupiec_p_ewma', 'backtest_christ_p_ewma', 'backtest_kupiec_stat_ewma', 'backtest_christ_stat_ewma',
                    'backtest_n_obs_var',
                    # Accuracy Metrics
                    'price_mae_ag','price_rmse_ag','price_mape_ag', 'price_mae_naive','price_rmse_naive','price_mape_naive',
                    'logret_mae_ag', 'logret_rmse_ag', 'logret_mae_naive', 'logret_rmse_naive',
                    'vol_rmse_ag', 'vol_qlike_ag', # AG Var metrics
                    'vol_rmse_ewma', 'vol_qlike_ewma', # EWMA Var metrics
                    # Residual Diagnostics
                    'resid_lb_pvalue','resid_lb_wn', 'resid_arch_pvalue','resid_arch_het',
                    # Summary Stats
                    'rolling_valid_steps_price','rolling_valid_steps_lr','rolling_valid_steps_variance',
                    'rolling_skipped_steps', 'rolling_fit_errors', 'rolling_forecast_duration_min'
                ]
                
                if isinstance(roll_fc_data, dict):
                    flat_res['bt_vol_rmse_ewma'] = roll_fc_data.get('vol_rmse_ewma', benchmarks.get('ewma_var', {}).get('rmse'))
                    flat_res['bt_vol_qlike_ewma'] = roll_fc_data.get('vol_qlike_ewma', benchmarks.get('ewma_var', {}).get('qlike'))
                else: # Fallback if roll_fc_data is not a dict 
                    flat_res['bt_vol_rmse_ewma'] = res.get('vol_rmse_ewma', benchmarks.get('ewma_var', {}).get('rmse'))
                    flat_res['bt_vol_qlike_ewma'] = res.get('vol_qlike_ewma', benchmarks.get('ewma_var', {}).get('qlike'))

                for key in bt_keys_to_flatten:
                    # Exclude EWMA var metrics handled above
                    if key not in ['vol_rmse_ewma', 'vol_qlike_ewma']:
                        flat_key = key.replace('backtest_', 'bt_') if key.startswith('backtest_') else key
                        value = roll_fc_data.get(key, res.get(key)) # Prefer roll_fc, fallback to top-level res
                        flat_res[flat_key] = value

                # --- Flatten Multi-Horizon-Specific Results ---
                mh_metrics = mh_eval.get('metrics', {})
                if isinstance(mh_metrics, dict):
                    for h_key, h_data in mh_metrics.items(): # h_key is like "h1", "h3"
                        if isinstance(h_data, dict):
                            for metric_name, metric_val in h_data.items():
                                flat_res[f"{h_key}_{metric_name}"] = metric_val # e.g., h1_mape_ag, h3_kupiec_p_ewma

                mh_price_dm = mh_eval.get('price_dm_tests', {})
                if isinstance(mh_price_dm, dict):
                    for h_key_dm, dm_data in mh_price_dm.items(): # h_key_dm is like "h1_dm_p_test"
                        prefix = h_key_dm.replace('_test', '') # -> "h1_dm_p"
                        if isinstance(dm_data, dict):
                            flat_res[f"{prefix}_stat"] = dm_data.get('dm_stat')
                            flat_res[f"{prefix}_pval"] = dm_data.get('p_value')
                            flat_res[f"{prefix}_sig"] = dm_data.get('is_significant')
                            flat_res[f"{prefix}_n"] = dm_data.get('n_obs')
                            flat_res[f"{prefix}_loss"] = dm_data.get('loss_type')
                            flat_res[f"{prefix}_err"] = dm_data.get('error')

                mh_var_dm = mh_eval.get('variance_dm_tests', {})
                if isinstance(mh_var_dm, dict):
                    for h_key_dm, dm_data in mh_var_dm.items(): # h_key_dm is like "h1_dm_v_test"
                        prefix = h_key_dm.replace('_test', '') # -> "h1_dm_v"
                        if isinstance(dm_data, dict):
                            flat_res[f"{prefix}_stat"] = dm_data.get('dm_stat')
                            flat_res[f"{prefix}_pval"] = dm_data.get('p_value')
                            flat_res[f"{prefix}_sig"] = dm_data.get('is_significant')
                            flat_res[f"{prefix}_n"] = dm_data.get('n_obs')
                            flat_res[f"{prefix}_loss"] = dm_data.get('loss_type')
                            flat_res[f"{prefix}_err"] = dm_data.get('error')

                # Multi-Horizon Summary Stats
                flat_res['mh_n_eval_points'] = mh_eval.get('n_eval_points')
                flat_res['mh_fc_errors'] = mh_eval.get('forecast_generation_errors')
                flat_res['mh_inv_errors'] = mh_eval.get('forecast_inversion_errors')
                flat_res['mh_ewma_errors'] = mh_eval.get('ewma_forecast_errors')
                flat_res['mh_duration_min'] = mh_eval.get('multi_horizon_duration_min') # Add duration

                # --- Keep thesis recommendations 
                flat_res['thesis_reco_price_dm'] = res.get('thesis_reco_price_dm', 'N/A')
                flat_res['thesis_reco_var_dm'] = res.get('thesis_reco_var_dm', 'N/A')
                flat_res['thesis_reco_var_backtest'] = res.get('thesis_reco_var_backtest', 'N/A')

                processed_results_list.append(flat_res)
            # --- [END FLATTENING CODE] ---

            results_df_raw = pd.DataFrame(processed_results_list)

        except Exception as e:
            log.error(f"Error processing results for summary DataFrame: {e}", exc_info=True)
            results_df_raw = pd.DataFrame(all_coin_results) # Fallback to raw data

        if not results_df_raw.empty:
            # --- Define column groups based on flattened results_df_raw ---
            cols_core = ['coin_id','status','error_message','total_runtime_min','total_rows_preprocessed','train_size','val_size','test_size']
            cols_desc = sorted([c for c in results_df_raw.columns if c.startswith('desc_')])
            cols_model = ['initial_adf_is_stationary','initial_kpss_is_stationary', 'applied_differencing_order','selection_method','tuning_criterion','autotune_criterion_value','param_significance_level','garch_type','garch_distribution','arima_order_p','arima_order_q','garch_order_p','garch_order_q','final_model_description']
            cols_initial_fit = sorted([c for c in results_df_raw.columns if 'initial_aic' in c or 'initial_bic' in c or 'initial_llf' in c])
            cols_reco = ['thesis_reco_price_dm', 'thesis_reco_var_dm', 'thesis_reco_var_backtest'] # Uses short keys

            # Backtest specific columns 
            cols_backtest = sorted([c for c in results_df_raw.columns if c.startswith(('price_', 'logret_', 'vol_', 'resid_', 'rolling_', 'bt_'))])

            # Horizon specific columns 
            cols_horizon = sorted([c for c in results_df_raw.columns if c.startswith('h') and '_' in c])
            cols_horizon_meta = sorted([c for c in results_df_raw.columns if c.startswith('mh_') or c.startswith('empirical_')])

            # --- Select ALL relevant columns that EXIST in the DataFrame ---
            all_possible_cols = cols_core + cols_desc + cols_model + cols_initial_fit + cols_backtest + cols_horizon + cols_horizon_meta + cols_reco
            cols_to_select = [c for c in all_possible_cols if c in results_df_raw.columns]
            seen = set()
            cols_to_select = [x for x in cols_to_select if not (x in seen or seen.add(x))] # Keep unique, preserve order

            summary_df = results_df_raw[cols_to_select].copy()
            # --- DEBUG PRINT ---
            print("\nDEBUG: summary_df columns:", summary_df.columns.tolist())
            if 'thesis_reco_price_dm' in summary_df.columns:
                 print("DEBUG: Example reco_price_dm from summary_df:", summary_df['thesis_reco_price_dm'].iloc[0])
            else:
                 print("DEBUG: thesis_reco_price_dm column NOT FOUND in summary_df")
             # --- END DEBUG PRINT ---

            # --- Define Formatters ---
            fmts = {}
            float_cols = summary_df.select_dtypes(include=['float']).columns
            for col in float_cols:
                 if 'mape' in col: fmts[col] = '{:.2f}%'.format
                 elif 'rmse' in col or 'mae' in col: fmts[col] = '{:.4f}'.format
                 elif 'qlike' in col: fmts[col] = '{:.6f}'.format
                 elif 'pvalue' in col or '_pval' in col or 'kupiec_p' in col or 'christ_p' in col: fmts[col] = '{:.3f}'.format
                 elif 'runtime' in col or 'duration' in col: fmts[col] = '{:.2f}'.format
                 elif 'criterion_value' in col or 'initial_aic' in col or 'initial_bic' in col or 'initial_llf' in col: fmts[col] = '{:.2f}'.format
                 elif '_stat' in col: fmts[col] = '{:.3f}'.format
                 elif '_var_' in col or '_es_' in col or col.startswith('desc_') or col.startswith('empirical_'): fmts[col] = '{:.4f}'.format
                 else: fmts[col] = '{:.3f}'.format # Default float format
            bool_cols = summary_df.select_dtypes(include=['bool', 'boolean']).columns
            for col in bool_cols: fmts[col] = lambda x: 'Yes' if x is True else ('No' if x is False else 'N/A')
            # Explicitly format integer columns
            int_cols = summary_df.select_dtypes(include=['int', 'int32', 'int64', 'Int64']).columns
            for col in int_cols:
                # Format specific integer columns as integers (no decimals)
                if any(s in col for s in ['_violations', '_n_obs', '_n', 'steps', 'errors', 'size', 'order', 'points']):
                     fmts[col] = lambda x: '{:.0f}'.format(x) if pd.notna(x) else 'N/A'

            # --- Helper function to format values for console output ---
            def format_console_value(value, col_name, formatters):
                if pd.isna(value): return "N/A"
                if col_name in formatters:
                    try: return formatters[col_name](value)
                    except Exception: return str(value) # Fallback if formatter fails
                # Ensure boolean-like values (like numpy.bool_) are handled
                if isinstance(value, (bool, np.bool_)): return "Yes" if value else "No"
                # Format integers if no specific format is set
                if isinstance(value, (int, np.integer)) and col_name not in formatters: return f"{value:,}" # Use comma separator
                return str(value) # Default string conversion

            # --- Helper function to extract recommendation text ---
            def extract_reco_text(text):
                 if isinstance(text, str) and ':' in text:
                     # Find the first colon and split
                     parts = text.split(':', 1)
                     # Add title line back with indentation
                     title_line = parts[0].strip() + ":"
                     # Split the rest by newline, filter empty lines, add indentation and bullet
                     sub_lines = [f"      - {line.strip()}" for line in parts[1].strip().split('\n') if line.strip().startswith('- ')]
                     return title_line + '\n' + '\n'.join(sub_lines)
                 elif isinstance(text, str):
                     return f"- {text}" # Add bullet point if it's just a simple string
                 else:
                     return "- N/A"

            # --- Print Console Summary (Coin by Coin) ---
            print("\n\n" + "=" * 80 + "\n--- FINAL SUMMARY (Console - Coin by Coin) ---\n" + "=" * 80)

            max_key_len = 0
            if not summary_df.empty:
                # Find max length of keys across all sections for alignment
                all_keys_for_len = cols_core + cols_desc + cols_model + cols_initial_fit + cols_backtest + cols_horizon + cols_horizon_meta # Exclude reco keys here
                all_keys_for_len = [k for k in all_keys_for_len if k in summary_df.columns and k != 'coin_id'] # Exclude coin_id, use existing keys
                if all_keys_for_len:
                    # Clean up keys for length calculation
                    display_keys_for_len = [key.replace('desc_', '', 1).replace('bt_', 'BT ', 1).replace('mh_', 'MH ', 1).replace('initial_', 'Initial ', 1).replace('_', ' ').title() for key in all_keys_for_len]
                    max_key_len = max(len(str(key)) for key in display_keys_for_len)
            max_key_len = max(max_key_len, 40) # Set a minimum reasonable width

            for index, coin_data in summary_df.iterrows():
                coin_id_print = coin_data.get('coin_id','Unknown').upper()
                print(f"\n----- Coin: {coin_id_print} -----") # Add extra newline before coin

                # Function to print a section with aligned keys and spacing
                def print_section(title, keys_to_include):
                    # Filter keys that exist AND have non-NA values in the current coin_data
                    # OR are the error_message when status is not OK
                    keys_in_data = []
                    for k in keys_to_include:
                        if k in coin_data.index:
                            if pd.notna(coin_data[k]):
                                keys_in_data.append(k)
                            # Special case for error_message
                            elif k == 'error_message' and coin_data.get('status') != 'OK':
                                keys_in_data.append(k)

                    if not keys_in_data:
                        # Skip printing the section title if there's nothing relevant to show
                        return

                    print(f"\n  {title}:") # Print title with newline before
                    for key in sorted(keys_in_data):
                        if key != 'coin_id':
                             value = coin_data[key]
                             formatted_value = format_console_value(value, key, fmts)
                             # Clean up key name for display
                             display_key = key.replace('desc_', '', 1).replace('bt_', 'BT ', 1).replace('mh_', 'MH ', 1).replace('initial_', 'Initial ', 1).replace('_', ' ').title()
                             print(f"    {display_key:<{max_key_len}} : {formatted_value}")

                # Print Common Sections
                print_section("Core Info", cols_core)
                print_section("Descriptive Statistics (Full Period)", cols_desc)
                print_section("Model Specification & Initial Stationarity", cols_model)
                print_section("Initial Fit (Train+Val) Metrics", cols_initial_fit)

                # --- Print Mode-Specific Sections ---
                current_mode = config_main['forecast_mode']

                if current_mode == 'backtest':
                    # (Code for backtest printing)
                    print(f"\n  Backtest Results (1-Step Rolling, Alpha={alpha_summary:.0%}):")
                    valid_steps_p = format_console_value(coin_data.get('rolling_valid_steps_price'), 'rolling_valid_steps_price', fmts)
                    valid_steps_v = format_console_value(coin_data.get('rolling_valid_steps_variance'), 'rolling_valid_steps_variance', fmts)
                    n_backtest_obs = format_console_value(coin_data.get('bt_n_obs_var'), 'bt_n_obs_var', fmts)
                    print(f"    Valid Steps (Price): {valid_steps_p} | (Variance): {valid_steps_v} | (VaR Backtest): {n_backtest_obs}")

                    # Price Accuracy
                    print("\n    --- Price Accuracy (AG vs Naive) ---") # Add newline
                    print(f"      Metric      {'AG Model':<15} {'Naive Model':<15}")
                    print(f"      MAE         {format_console_value(coin_data.get('price_mae_ag'), 'price_mae_ag', fmts):<15} {format_console_value(coin_data.get('price_mae_naive'), 'price_mae_naive', fmts):<15}")
                    print(f"      RMSE        {format_console_value(coin_data.get('price_rmse_ag'), 'price_rmse_ag', fmts):<15} {format_console_value(coin_data.get('price_rmse_naive'), 'price_rmse_naive', fmts):<15}")
                    print(f"      MAPE        {format_console_value(coin_data.get('price_mape_ag'), 'price_mape_ag', fmts):<15} {format_console_value(coin_data.get('price_mape_naive'), 'price_mape_naive', fmts):<15}")
                    # Price DM Test
                    print(f"\n    --- Price DM Test (AG vs Naive, H1: AG better) ---") # Add newline
                    loss_p = coin_data.get('bt_dm_p_loss', 'N/A')
                    n_p = format_console_value(coin_data.get('bt_dm_p_n'), 'bt_dm_p_n', fmts)
                    stat_p = format_console_value(coin_data.get('bt_dm_p_stat'), 'bt_dm_p_stat', fmts)
                    pval_p = format_console_value(coin_data.get('bt_dm_p_pval'), 'bt_dm_p_pval', fmts)
                    sig_p = format_console_value(coin_data.get('bt_dm_p_sig'), 'bt_dm_p_sig', fmts)
                    print(f"      Loss='{loss_p}' N={n_p:<5} Stat={stat_p:>8} p-value={pval_p:>7} Significant={sig_p}")

                    # Variance Accuracy
                    print("\n    --- Variance Accuracy (AG vs EWMA) ---") # Add newline
                    print(f"      Metric      {'AG Model':<15} {'EWMA Model':<15}")
                    print(f"      RMSE        {format_console_value(coin_data.get('vol_rmse_ag'), 'vol_rmse_ag', fmts):<15} {format_console_value(coin_data.get('bt_vol_rmse_ewma'), 'bt_vol_rmse_ewma', fmts):<15}")
                    print(f"      QLIKE       {format_console_value(coin_data.get('vol_qlike_ag'), 'vol_qlike_ag', fmts):<15} {format_console_value(coin_data.get('bt_vol_qlike_ewma'), 'bt_vol_qlike_ewma', fmts):<15}")
                    # Variance DM Test
                    print(f"\n    --- Variance DM Test (AG vs EWMA, H1: AG better) ---") # Add newline
                    loss_v = coin_data.get('bt_dm_v_loss', 'N/A')
                    n_v = format_console_value(coin_data.get('bt_dm_v_n'), 'bt_dm_v_n', fmts)
                    stat_v = format_console_value(coin_data.get('bt_dm_v_stat'), 'bt_dm_v_stat', fmts)
                    pval_v = format_console_value(coin_data.get('bt_dm_v_pval'), 'bt_dm_v_pval', fmts)
                    sig_v = format_console_value(coin_data.get('bt_dm_v_sig'), 'bt_dm_v_sig', fmts)
                    print(f"      Loss='{loss_v}' N={n_v:<5} Stat={stat_v:>8} p-value={pval_v:>7} Significant={sig_v}")

                    # VaR Backtesting
                    print("\n    --- VaR Backtesting ---") # Add newline
                    print(f"      Metric             {'AG Model':<15} {'EWMA Model':<15}")
                    print(f"      Violations         {format_console_value(coin_data.get('bt_violations_ag'), 'bt_violations_ag', fmts):<15} {format_console_value(coin_data.get('bt_violations_ewma'), 'bt_violations_ewma', fmts):<15}")
                    print(f"      Kupiec p-val       {format_console_value(coin_data.get('bt_kupiec_p_ag'), 'bt_kupiec_p_ag', fmts):<15} {format_console_value(coin_data.get('bt_kupiec_p_ewma'), 'bt_kupiec_p_ewma', fmts):<15}")
                    print(f"      Christoff. p-val   {format_console_value(coin_data.get('bt_christ_p_ag'), 'bt_christ_p_ag', fmts):<15} {'N/A':<15}") # Christoffersen usually not done for EWMA
                    print(f"      Avg VaR ({alpha_summary:.0%})      {format_console_value(coin_data.get('bt_avg_var_ag'), 'bt_avg_var_ag', fmts):<15} {format_console_value(coin_data.get('bt_avg_var_ewma'), 'bt_avg_var_ewma', fmts):<15}")
                    print(f"      Avg ES ({alpha_summary:.0%})       {format_console_value(coin_data.get('bt_avg_es_ag'), 'bt_avg_es_ag', fmts):<15} {format_console_value(coin_data.get('bt_avg_es_ewma'), 'bt_avg_es_ewma', fmts):<15}")

                    # Residual Diagnostics
                    print("\n    --- Residual Diagnostics (AG Model Std. Residuals) ---") # Add newline
                    lb_wn = format_console_value(coin_data.get('resid_lb_wn'), 'resid_lb_wn', fmts)
                    lb_p = format_console_value(coin_data.get('resid_lb_pvalue'), 'resid_lb_pvalue', fmts)
                    arch_het = format_console_value(coin_data.get('resid_arch_het'), 'resid_arch_het', fmts)
                    arch_p = format_console_value(coin_data.get('resid_arch_pvalue'), 'resid_arch_pvalue', fmts)
                    print(f"      Ljung-Box White Noise? {lb_wn:<5} (p={lb_p})")
                    print(f"      ARCH LM Heteroskedastic? {arch_het:<5} (p={arch_p})")

                elif current_mode == 'horizon_evaluation':
                    # (Code for horizon evaluation printing)
                    print(f"\n  Horizon Evaluation Results (Alpha={alpha_summary:.0%}):")
                    eval_horizons = config_main.get('evaluation_horizons', [])
                    if not eval_horizons:
                        print("    (No evaluation horizons found in config)")
                    else:
                        n_eval_pts = format_console_value(coin_data.get('mh_n_eval_points'), 'mh_n_eval_points', fmts)
                        print(f"    Evaluation Points: {n_eval_pts}")
                        for h in eval_horizons:
                             h_prefix = f"h{h}"
                             valid_pairs_h = format_console_value(coin_data.get(f'{h_prefix}_valid_pairs'), f'{h_prefix}_valid_pairs', fmts)
                             n_backtest_h = format_console_value(coin_data.get(f'{h_prefix}_n_backtest_obs'), f'{h_prefix}_n_backtest_obs', fmts)
                             print(f"\n    --- Horizon h={h} (Valid Pairs: {valid_pairs_h}, VaR N: {n_backtest_h}) ---") # Added newline

                             # Price Accuracy
                             print(f"      Price Accuracy (AG vs Naive):")
                             print(f"        RMSE: AG={format_console_value(coin_data.get(f'{h_prefix}_rmse_price_ag'), f'{h_prefix}_rmse_price_ag', fmts):<12} Naive={format_console_value(coin_data.get(f'{h_prefix}_rmse_price_naive'), f'{h_prefix}_rmse_price_naive', fmts):<12}")
                             print(f"        MAPE: AG={format_console_value(coin_data.get(f'{h_prefix}_mape_ag'), f'{h_prefix}_mape_ag', fmts):<12} Naive={format_console_value(coin_data.get(f'{h_prefix}_mape_naive'), f'{h_prefix}_mape_naive', fmts):<12}")
                             # Price DM Test
                             loss_ph = coin_data.get(f'{h_prefix}_dm_p_loss', 'N/A')
                             n_ph = format_console_value(coin_data.get(f'{h_prefix}_dm_p_n'), f'{h_prefix}_dm_p_n', fmts)
                             stat_ph = format_console_value(coin_data.get(f'{h_prefix}_dm_p_stat'), f'{h_prefix}_dm_p_stat', fmts)
                             pval_ph = format_console_value(coin_data.get(f'{h_prefix}_dm_p_pval'), f'{h_prefix}_dm_p_pval', fmts)
                             sig_ph = format_console_value(coin_data.get(f'{h_prefix}_dm_p_sig'), f'{h_prefix}_dm_p_sig', fmts)
                             print(f"      Price DM (AG vs Naive, H1:AG better): Loss='{loss_ph}' N={n_ph:<5} Stat={stat_ph:>7} p={pval_ph:>6} Sig={sig_ph}")

                             # Variance Accuracy
                             print(f"      Variance Accuracy (AG vs EWMA):")
                             print(f"        RMSE: AG={format_console_value(coin_data.get(f'{h_prefix}_rmse_var_ag'), f'{h_prefix}_rmse_var_ag', fmts):<12} EWMA={format_console_value(coin_data.get(f'{h_prefix}_rmse_var_ewma'), f'{h_prefix}_rmse_var_ewma', fmts):<12}")
                             print(f"        QLIKE: AG={format_console_value(coin_data.get(f'{h_prefix}_qlike_var_ag'), f'{h_prefix}_qlike_var_ag', fmts):<12} EWMA={format_console_value(coin_data.get(f'{h_prefix}_qlike_var_ewma'), f'{h_prefix}_qlike_var_ewma', fmts):<12}")
                             # Variance DM Test
                             loss_vh = coin_data.get(f'{h_prefix}_dm_v_loss', 'N/A')
                             n_vh = format_console_value(coin_data.get(f'{h_prefix}_dm_v_n'), f'{h_prefix}_dm_v_n', fmts)
                             stat_vh = format_console_value(coin_data.get(f'{h_prefix}_dm_v_stat'), f'{h_prefix}_dm_v_stat', fmts)
                             pval_vh = format_console_value(coin_data.get(f'{h_prefix}_dm_v_pval'), f'{h_prefix}_dm_v_pval', fmts)
                             sig_vh = format_console_value(coin_data.get(f'{h_prefix}_dm_v_sig'), f'{h_prefix}_dm_v_sig', fmts)
                             print(f"      Var DM (AG vs EWMA, H1:AG better):   Loss='{loss_vh}' N={n_vh:<5} Stat={stat_vh:>7} p={pval_vh:>6} Sig={sig_vh}")

                             # VaR Backtesting
                             print(f"      VaR Backtesting (AG vs EWMA):")
                             print(f"        Metric          {'AG Model':<15} {'EWMA Model':<15}")
                             print(f"        Violations      {format_console_value(coin_data.get(f'{h_prefix}_violations_ag'), f'{h_prefix}_violations_ag', fmts):<15} {format_console_value(coin_data.get(f'{h_prefix}_violations_ewma'), f'{h_prefix}_violations_ewma', fmts):<15}")
                             print(f"        Kupiec p-val    {format_console_value(coin_data.get(f'{h_prefix}_kupiec_p_ag'), f'{h_prefix}_kupiec_p_ag', fmts):<15} {format_console_value(coin_data.get(f'{h_prefix}_kupiec_p_ewma'), f'{h_prefix}_kupiec_p_ewma', fmts):<15}")
                             print(f"        Christ. p-val   {format_console_value(coin_data.get(f'{h_prefix}_christ_p_ag'), f'{h_prefix}_christ_p_ag', fmts):<15} {'N/A':<15}") # Christoffersen usually not done for EWMA
                             print(f"        Avg VaR ({alpha_summary:.0%})   {format_console_value(coin_data.get(f'{h_prefix}_avg_var_ag'), f'{h_prefix}_avg_var_ag', fmts):<15} {format_console_value(coin_data.get(f'{h_prefix}_avg_var_ewma'), f'{h_prefix}_avg_var_ewma', fmts):<15}")
                             print(f"        Avg ES ({alpha_summary:.0%})    {format_console_value(coin_data.get(f'{h_prefix}_avg_es_ag'), f'{h_prefix}_avg_es_ag', fmts):<15} {format_console_value(coin_data.get(f'{h_prefix}_avg_es_ewma'), f'{h_prefix}_avg_es_ewma', fmts):<15}")


                elif current_mode == 'future':
                    print("\n  Future Forecast Summary:")
                    print(f"    (Details saved to CSV/plot if generated)")
                    # Add any specific summary stats for future mode if you store them


                # --- Print Thesis Recommendations ---
                print("\n  Recommendations:") # Add newline before title
                reco_price = coin_data.get('thesis_reco_price_dm', 'N/A') 
                reco_var = coin_data.get('thesis_reco_var_dm', 'N/A')   
                reco_backtest = coin_data.get('thesis_reco_var_backtest', 'N/A') 

                # Use the helper function for better formatting
                print(f"    Price DM:\n      {extract_reco_text(reco_price)}")
                print(f"    Var DM:\n      {extract_reco_text(reco_var)}")
                print(f"    VaR Backtest:\n      {extract_reco_text(reco_backtest)}")

            print("\n" + "=" * 80) # End of coin loop spacing

            # --- [LaTeX, CSV, PNG output code remains the same] ---
            # --- Save Full Flattened Summary to CSV ---
            try:
                summary_filename = os.path.join(config_main['plot_dir_base'], f"FINAL_SUMMARY_All_Coins_{config_main['forecast_mode']}.csv")
                # Use the 'summary_df' which contains all selected columns
                summary_df.to_csv(summary_filename, index=False, float_format="%.8f")
                log.info(f"FINAL SUMMARY CSV saved: {summary_filename}")
            except Exception as e:
                log.error(f"Failed to save FINAL SUMMARY CSV: {e}")

            # --- Optional: Save Summary to PNG ---
            if DATAFRAME_IMAGE_AVAILABLE and dfi is not None and not summary_df.empty:
                try:
                    summary_png_filename = os.path.join(config_main['plot_dir_base'], f"FINAL_SUMMARY_All_Coins_{config_main['forecast_mode']}.png")
                    # Select fewer columns for better PNG readability, or use transpose
                    cols_for_png = cols_core + ['final_model_description']
                    if config_main['forecast_mode'] == 'backtest':
                         cols_for_png.extend(['price_rmse_ag','vol_qlike_ag','bt_dm_p_sig','bt_dm_v_sig','bt_kupiec_p_ag','bt_christ_p_ag'])
                    elif config_main['forecast_mode'] == 'horizon_evaluation':
                         eval_horizons_png = config_main.get('evaluation_horizons', [])
                         h_first = f"h{eval_horizons_png[0]}" if eval_horizons_png else None
                         h_last = f"h{eval_horizons_png[-1]}" if eval_horizons_png else None
                         if h_first: cols_for_png.extend([f'{h_first}_rmse_price_ag',f'{h_first}_qlike_var_ag',f'{h_first}_dm_p_sig',f'{h_first}_dm_v_sig',f'{h_first}_kupiec_p_ag'])
                         if h_last and h_last != h_first: cols_for_png.extend([f'{h_last}_rmse_price_ag',f'{h_last}_qlike_var_ag',f'{h_last}_dm_v_sig',f'{h_last}_kupiec_p_ag'])

                    cols_for_png_exist = [c for c in cols_for_png if c in summary_df.columns]
                    summary_df_png = summary_df[cols_for_png_exist].set_index('coin_id')
                    # Transpose might be better for PNG
                    dfi.export(summary_df_png.T, summary_png_filename, table_conversion='matplotlib', dpi=200)
                    log.info(f"FINAL SUMMARY PNG saved: {summary_png_filename}")
                except Exception as e:
                    log.error(f"Failed to save FINAL SUMMARY PNG: {e}")

        else:
            log.error("Results DataFrame is empty after processing all coins. Cannot generate summary.")
        # --- End of summary generation block ---

    # --- Final Script Completion Message ---
    overall_end_time=time.time()
    total_duration_min=(overall_end_time-overall_start_time)/60
    log.info(f"--- Multi-Coin Analysis Finished. Total Duration: {total_duration_min:.2f} Minutes ---")
    final_output_dir = os.path.abspath(config_main['plot_dir_base'])
    print(f"\nAnalysis complete. Results saved under: {final_output_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        os.chdir(script_dir)
        log.info(f"Changed working directory to: {script_dir}")
    except Exception as e:
        log.error(f"Could not change working directory to {script_dir}: {e}. Relative paths might be incorrect.")
    main()
    
#python3 "ARIMA GARCH FINAL.py" > output_doge_sol.txt 2>&1
