# Forecast error metrics
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

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
