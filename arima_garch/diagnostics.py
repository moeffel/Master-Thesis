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
