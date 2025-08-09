# src/data/preprocess.py
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame, fill_method: Optional[str] = 'ffill', dropna: bool = True, min_valid_tickers: int = 10, price_col: str = 'Close', max_na_ratio: float = 0.1) -> pd.DataFrame:
    """
    Clean stock data by handling missing values and infinite values.
    
    Args:
        df (pd.DataFrame): Input DataFrame with stock data.
        fill_method (Optional[str], optional): Method to fill NaNs ('ffill', 'bfill', etc.). Defaults to 'ffill'.
        dropna (bool, optional): Whether to drop remaining NaNs after filling. Defaults to True.
        min_valid_tickers (int, optional): Minimum number of valid tickers required. Defaults to 10.
        price_col (str, optional): Column name for price data (e.g., 'Close' or 'Adj Close'). Defaults to 'Close'.
        max_na_ratio (float, optional): Maximum allowed NaN ratio per ticker before dropping. Defaults to 0.1 (10%).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with tickers as columns and price data.
    
    Raises:
        ValueError: If insufficient valid tickers remain after cleaning or price_col is missing.
        Exception: If cleaning fails.
    """
    try:
        # Handle multi-index (yfinance format)
        if isinstance(df.columns, pd.MultiIndex):
            if price_col not in df.columns.levels[1]:
                raise ValueError(f"Price column '{price_col}' not found in DataFrame columns: {df.columns.levels[1]}")
            df = df.xs(price_col, level=1, axis=1, drop_level=True)
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many NaNs
        na_ratio = df.isna().mean()
        valid_columns = na_ratio[na_ratio <= max_na_ratio].index
        df = df[valid_columns]
        
        if df.shape[1] < min_valid_tickers:
            raise ValueError(f"Too few valid tickers after NaN filtering: {df.shape[1]} (minimum required: {min_valid_tickers})")
        
        # Fill missing values
        if fill_method:
            df = df.fillna(method=fill_method)
        
        # Drop any remaining NaN rows if dropna is True
        if dropna:
            df = df.dropna(how='any')
        
        if df.empty:
            raise ValueError("Cleaned DataFrame is empty after processing.")
        
        logger.info(f"Data cleaned successfully: {df.shape[0]} days, {df.shape[1]} stocks.")
        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise

def clean_index_data(df: pd.DataFrame, fill_method: Optional[str] = 'ffill', dropna: bool = True, price_col: str = 'Close') -> pd.DataFrame:
    """
    Clean single-ticker index data by handling missing values and infinite values.
    
    Args:
        df (pd.DataFrame): Input DataFrame with index data (single ticker).
        fill_method (Optional[str], optional): Method to fill NaNs ('ffill', 'bfill', etc.). Defaults to 'ffill'.
        dropna (bool, optional): Whether to drop remaining NaNs after filling. Defaults to True.
        price_col (str, optional): Column name for price data (e.g., 'Close' or 'Adj Close'). Defaults to 'Close'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with a single column for the index.
    
    Raises:
        ValueError: If price_col is missing or DataFrame is empty.
        Exception: If cleaning fails.
    """
    try:
        # Handle multi-index (yfinance format) or single-ticker DataFrame
        if isinstance(df.columns, pd.MultiIndex):
            if price_col not in df.columns.levels[1]:
                raise ValueError(f"Price column '{price_col}' not found in DataFrame columns: {df.columns.levels[1]}")
            df = df.xs(price_col, level=1, axis=1, drop_level=True)
        elif price_col in df.columns:
            df = df[[price_col]]
        else:
            raise ValueError(f"Price column '{price_col}' not found in DataFrame columns: {df.columns}")
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values
        if fill_method:
            df = df.fillna(method=fill_method)
        
        # Drop any remaining NaN rows if dropna is True
        if dropna:
            df = df.dropna(how='any')
        
        if df.empty:
            raise ValueError("Cleaned index DataFrame is empty after processing.")
        
        logger.info(f"Index data cleaned successfully: {df.shape[0]} days, {df.shape[1]} index.")
        return df
    except Exception as e:
        logger.error(f"Error cleaning index data: {e}")
        raise

def calculate_daily_returns(df: pd.DataFrame, price_col: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate daily returns from stock or index price data.
    
    Args:
        df (pd.DataFrame): DataFrame with prices (tickers as columns or multi-index with price_col).
        price_col (Optional[str], optional): Column name for prices if multi-index. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame with daily returns (tickers as columns).
    
    Raises:
        ValueError: If DataFrame is empty or invalid.
        Exception: If calculation fails.
    """
    try:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        
        # If DataFrame is already single-level (tickers as columns), use it directly
        if not isinstance(df.columns, pd.MultiIndex):
            prices = df
        else:
            # Handle multi-index case
            if price_col is None:
                raise ValueError("price_col must be specified for multi-index DataFrame.")
            prices = df.xs(price_col, level=1, axis=1, drop_level=True)
        
        # Replace zeros with NaN to avoid invalid pct_change results
        prices = prices.replace(0, np.nan)
        
        # Calculate returns and drop rows with any NaNs
        returns = prices.pct_change().dropna(how='any')
        
        if returns.empty:
            raise ValueError("Returns DataFrame is empty after calculation.")
        
        logger.info(f"Daily returns calculated: {returns.shape[0]} days, {returns.shape[1]} stocks.")
        return returns
    except Exception as e:
        logger.error(f"Error calculating daily returns: {e}")
        raise