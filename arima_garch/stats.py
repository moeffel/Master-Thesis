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
    if adfuller is None:
        return {'p_value':np.nan,'is_stationary':False,'test_statistic':np.nan,'critical_values':{}, 'error': 'adfuller unavailable'}
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
    if kpss is None:
        return {'p_value':np.nan,'is_stationary':False,'test_statistic':np.nan,'critical_values':{}, 'lags':None, 'error': 'kpss unavailable'}
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
