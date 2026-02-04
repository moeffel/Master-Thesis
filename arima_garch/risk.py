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
