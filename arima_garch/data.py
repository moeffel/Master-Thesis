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

def fetch_data_yahoo(coin_id: str, start: Optional[str]=None, end: Optional[str]=None) -> pd.DataFrame:
    """Fetches historical price data from Yahoo Finance."""
    log.info(f"Fetching data for {coin_id} from Yahoo ({start} to {end})...")
    if yf is None:
        raise ImportError("yfinance is not installed. Install with `pip install yfinance`.")
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
        df_copy['log_return'] = df_copy['log_return'].replace([np.inf, -np.inf], np.nan)

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
