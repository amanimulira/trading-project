# Data Cleaning and Transformation - src/data/preprocess.py

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame, fill_method: Optional[str] = 'ffill', dropna: bool = True) -> pd.DataFrame:
    """
    Clean stock data by handling missing values and infinite values.
    
    Args:
        df (pd.DataFrame): Input DataFrame with stock data.
        fill_method (Optional[str], optional): Method to fill NaNs ('ffill', 'bfill', etc.). Defaults to 'ffill'.
        dropna (bool, optional): Whether to drop remaining NaNs after filling. Defaults to True.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    
    Raises:
        Exception: If cleaning fails.
    """
    try:
        if fill_method:
            df = df.fillna(method=fill_method)
        df = df.replace([np.inf, -np.inf], np.nan)
        if dropna:
            df = df.dropna()
        logger.info("Data cleaned successfully.")
        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise

def calculate_daily_returns(df: pd.DataFrame, price_col: str = 'Adj Close') -> pd.DataFrame:
    """
    Calculate daily returns from stock price data.
    
    Args:
        df (pd.DataFrame): DataFrame with stock prices (multi-ticker support).
        price_col (str, optional): Column name for prices. Defaults to 'Adj Close'.
    
    Returns:
        pd.DataFrame: DataFrame with daily returns (tickers as columns if multi-ticker).
    
    Raises:
        Exception: If calculation fails.
    """
    try:
        if isinstance(df.columns, pd.MultiIndex):
            # For multi-ticker data from yfinance
            prices = df[price_col]
        else:
            prices = df[[price_col]]
        returns = prices.pct_change().dropna()
        logger.info("Daily returns calculated.")
        return returns
    except Exception as e:
        logger.error(f"Error calculating daily returns: {e}")
        raise