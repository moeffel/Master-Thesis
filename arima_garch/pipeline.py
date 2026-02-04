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
            results['selection_method'] = 'Auto-Tune'
            log.info(f"[{coin_id}] Starting Auto-Tuning (Criterion: {tuning_criterion}, Sig Level: {param_sig_level})...")
            garch_types_for_tuning = config.get('garch_types_to_tune', ['GARCH'])
            # Call the tuning function
            tuning_result = auto_tune_arima_garch(
                selection_data,
                config['tune_min_p'], config['tune_max_p'], config['tune_min_q'], config['tune_max_q'],
                config['tune_min_d'], config['tune_max_d'], # Note: this is *additional* d
                config['tune_min_gp'], config['tune_max_gp'], config['tune_min_gq'], config['tune_max_gq'],
                garch_types_for_tuning,
                sf=config['default_scale_factor'], tune_criterion=tuning_criterion,
                param_sig_level=param_sig_level, verbose=config['verbose_tuning']
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
    # --- Initial Diagnostic Plots (ACF/PACF on differenced Train+Val, QQ on original Train+Val LogRet) ---
    # ANSTATT die Plots hier zu speichern, speichern wir die Daten für später.
    plotting_data = {
        'qq_series': fit_base_df['log_return'].dropna(),
        'acf_pacf_series': differenced_series.copy() if differenced_series is not None else pd.Series(dtype=float),
        'd_final': d_final,
        'coin_id': coin_id
    }
    results['plotting_data'] = plotting_data # Fügen Sie dies dem results-Dictionary hinzu
    
        
        
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
        last_fit_arima_cached = None
        last_fit_garch_cached = None
        last_fit_step_idx = -config['refit_interval'] # Ensure initial fit

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
                     ax.plot(volatility_index, actual_vol_proxy, label='Actual Proxy |LogRet|',color='black',lw=1.2,alpha=0.5)
                     # Plot AG volatility forecast if available
                     eval_vol_ag_valid = evaluation_df['forecast_volatility_ag'].dropna()
                     if not eval_vol_ag_valid.empty:
                         ax.plot(eval_vol_ag_valid.index, eval_vol_ag_valid, label='AG Forecast Vol (t+1)',color='red',ls='-',lw=1.5)
                     # Plot EWMA volatility forecast if available
                     ewma_vol_valid = np.sqrt(evaluation_df['forecast_variance_ewma'].dropna().clip(0))
                     if not ewma_vol_valid.empty:
                         ax.plot(ewma_vol_valid.index, ewma_vol_valid, label='EWMA Forecast Vol (t+1)',color='blue',ls=':',lw=1.2,alpha=0.8)

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

def main(config_override: Optional[Dict] = None): # Wrap the main execution logic in a function
    global PLOT_AVAILABLE # Allow modification of the global variable
    overall_start_time = time.time()
    log.info(f"--- Starting Multi-Coin ARIMA-GARCH Analysis (v28.5 - English Output / Vertical Summary) ---")
    config_main = copy.deepcopy(CONFIG)
    if config_override:
        config_main.update(config_override)
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

    # --- NEUER BLOCK: Generate Combined Diagnostic Plots ---
    log.info("--- Generating Combined Diagnostic Plots ---")
    # Extrahieren Sie die notwendigen Daten aus den Ergebnissen
    plotting_data_for_all_coins = [res.get('plotting_data') for res in all_coin_results if res.get('plotting_data')]
    
    if len(plotting_data_for_all_coins) == 4:
        # Sortieren, um eine konsistente Reihenfolge sicherzustellen (BTC, ETH, DOGE, SOL)
        coin_order = config_main.get("coins_to_analyze", [])
        plotting_data_for_all_coins.sort(key=lambda x: coin_order.index(x['coin_id']))
        
        plot_combined_qq(plotting_data_for_all_coins, config_main['plot_dir_base'])
        plot_combined_acf_pacf(plotting_data_for_all_coins, config_main['plot_dir_base'])
    else:
        log.warning("Could not generate combined plots because plotting data was not available for all 4 assets.")
    
    # --- Final Script Completion Message ---
    overall_end_time=time.time()
    total_duration_min=(overall_end_time-overall_start_time)/60
    log.info(f"--- Multi-Coin Analysis Finished. Total Duration: {total_duration_min:.2f} Minutes ---")
    final_output_dir = os.path.abspath(config_main['plot_dir_base'])
    print(f"\nAnalysis complete. Results saved under: {final_output_dir}")

if __name__ == "__main__":
    from .cli import main as cli_main
    raise SystemExit(cli_main())
