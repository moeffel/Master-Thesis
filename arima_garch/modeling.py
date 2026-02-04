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
    data: pd.Series, min_p: int, max_p: int, min_q: int, max_q: int, min_d: int, max_d: int, # ARIMA orders (d is *additional*)
    min_gp: int, max_gp: int, min_gq: int, max_gq: int, # GARCH orders
    garch_types_list: List[str], # List of GARCH types ('GARCH', 'EGARCH'...)
    sf: float = 100.0, tune_criterion: str = 'BIC',
    param_sig_level: float = 0.10, verbose: bool = False # <-- Added verbose parameter here
) -> Dict:
    """Automatically tunes ARIMA-GARCH model orders based on criterion and significance."""

    if data.isnull().any(): raise ValueError("Input data for tuning contains NaN.")
    min_len_tune = 30 # Minimum length for reliable tuning fits
    if len(data) < min_len_tune: raise ValueError(f"Series length {len(data)} < {min_len_tune} required for tuning.")

    best_model = {
        'arima': (1, 0, 1), # (p, 0, q) - d is handled separately
        'add_d_recommended': 0, # Recommended *additional* differencing
        'garch_order': (1, 1),
        'garch_type': 'GARCH',
        'garch_dist': 't',
        'criterion_value': np.inf,
        'criterion_used': tune_criterion.upper(),
        'stable_and_significant': False, # Flag if a good model was found
        'error': None
    }

    # Define ranges for orders
    p_range = range(min_p, max_p + 1)
    q_range = range(min_q, max_q + 1)
    add_d_range = range(min_d, max_d + 1) # Range for *additional* d
    actual_min_gp = max(0, min_gp) # GARCH p,q can be 0 for pure ARIMA
    actual_min_gq = max(0, min_gq)
    gp_range = range(actual_min_gp, max_gp + 1)
    gq_range = range(actual_min_gq, max_gq + 1)
    dist_options = ['normal', 't'] if scipy_stats_available else ['normal', 't']
    if not garch_types_list: garch_types_list = ['GARCH'] # Default if empty

    # Create combinations, applying FIGARCH constraints
    combo_iter = itertools.product(p_range, add_d_range, q_range, gp_range, gq_range, garch_types_list, dist_options)
    valid_combinations = []
    for combo in combo_iter:
        p_c, add_d_c, q_c, gp_c, gq_c, garch_type_c, g_dist_c = combo
        is_figarch = garch_type_c.upper() == 'FIGARCH'
        is_pure_arima_candidate = (gp_c == 0 and gq_c == 0)

        if is_figarch:
            # FIGARCH in 'arch' might have constraints like p<=1, q<=1. Check documentation.
            # Assume p=0 or q=0 is allowed if the other is 1. Allow (1,1). Don't allow (0,0).
            if gp_c <= 1 and gq_c <= 1 and not is_pure_arima_candidate:
                valid_combinations.append(combo)
        elif not is_pure_arima_candidate: # Standard GARCH models require p>0 or q>0
             valid_combinations.append(combo)
        elif is_pure_arima_candidate: # Allow pure ARIMA (gp=0, gq=0)
             # Ensure only one GARCH type ('None') and one dist ('None') for pure ARIMA
             temp_combo = (p_c, add_d_c, q_c, 0, 0, 'None', 'None')
             if temp_combo not in valid_combinations: # Add only once
                 valid_combinations.append(temp_combo)


    combinations = valid_combinations # Final list of combinations to test
    total_combinations = len(combinations)

    if total_combinations == 0:
        log.warning("No valid model combinations generated for tuning (check constraints/ranges).")
        best_model['error'] = "No valid combinations"
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
