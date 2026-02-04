# Configuration and symbols

CRYPTO_COINS = ["bitcoin", "ethereum", "dogecoin", "solana"]

CONFIG = {
    "coins_to_analyze": ["bitcoin", "ethereum", "dogecoin", "solana"],
    "start_date": "2020-05-11",  # YYYY-MM-DD
    "end_date": "2024-04-20",    # None = today #"2025-04-01"
    "split_ratios": (0.70, 0.15, 0.15),  # Train, Validation, Test
    "forecast_mode": 'backtest',  # 'horizon_evaluation', 'backtest', 'future' 
    "evaluation_horizons": [1, 3, 7, 14,30],  # Days for horizon evaluation
    "dm_test_loss_type": "Squared Error",  # For price/return DM tests
    "dm_test_alpha": 0.05,  # Significance level for Diebold-Mariano tests
    "ewma_lambda": 0.94,  # Smoothing factor for EWMA benchmark volatility
    "dm_test_variance_loss_type": "QLIKE",  # qlike or squared error
    "fitting_window_size": 60,  # Days for rolling window in 'backtest' mode ##90
    "refit_interval": 1,  # Refit model every N steps in 'backtest' mode
    "future_forecast_horizon": 70,  # Days for 'future' mode forecast
    "use_auto_tune": True,  # Use automated model selection
    "compare_garch_dists": True,  # If not auto-tuning, compare dists
    "manual_arima_order": (1, 0, 1),  # Manual ARIMA order (p, d, q)
    "manual_garch_order": (1, 1),
    "garch_distribution_manual_choice": 't',
    "garch_vol_model_manual": 'FIGARCH', # GARCH, EGARCH, GJR, FIGARCH
    "tune_min_p": 0, "tune_max_p": 3,
    "tune_min_q": 0, "tune_max_q": 3,
    "tune_min_d": 0, "tune_max_d": 0,
    "tune_min_gp": 1, "tune_max_gp": 2,
    "tune_min_gq": 1, "tune_max_gq": 2,
    "garch_types_to_tune": ['GARCH', 'FIGARCH','GJR','EGARCH'], # EGARCH unstable in tuning
    "tune_criterion": 'AIC', # 'AIC' or 'BIC'
    "param_significance_level": 0.10,  # Significance level for parameter significance
    "plot_dir_base": "thesis_results", # Base directory for plots
    "generate_parameter_tables": True,
    "generate_console_parameter_output": True,
    "generate_stability_plots": True, # 
    "verbose_tuning": False,
    "default_scale_factor": 100.0, 
    "max_differencing_order": 0,
    "min_data_length": 400,
    "min_fitting_window_size": 30,  # Minimum fitting window size for backtest
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
