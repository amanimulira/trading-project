# Data Collection and Loading - src/data/fetch.py

import pandas as pd
import yfinance as yf 
from typing import List
from fredapi import Fred
import logging 

logger = logging.getLogger(__name__)

def get_sp500_tickers() -> List[str]:
    """
    Fetch the list of current S&P 500 tickers from Wikipedia.

    Returns:
        List[str]: List of S&P 500 stock tickers.

    Raises:
        Exception: If fetching the tickers fails.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url)[0]
        tickers = table['Symbol'].tolist()
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")
        raise

def fetch_stock_data(tickers: List[str], start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """
    Fetch historical stock data for given tickers using yfinance.
    
    Args:
        tickers (List[str]): List of stock tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str, optional): Data interval (e.g., '1d' for daily). Defaults to '1d'.
    
    Returns:
        pd.DataFrame: Multi-index DataFrame with stock data.
    
    Raises:
        Exception: If data fetching fails.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
        logger.info(f"Stock data fetched successfully for {len(tickers)} tickers.")
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        raise

def fetch_fed_data(series_ids: List[str], start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    """
    Fetch economic data from the Federal Reserve Economic Data (FRED) API using fredapi.
    
    Args:
        series_ids (List[str]): List of FRED series IDs (e.g., ['FEDFUNDS', 'UNRATE']).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        api_key (str, optional): FRED API key. Defaults to None, uses FRED_API_KEY environment variable.
    
    Returns:
        pd.DataFrame: DataFrame with FRED data, indexed by date, with series IDs as columns.
    
    Raises:
        ValueError: If API key is not provided or set in environment.
        Exception: If data fetching fails.
    """
    try:
        if api_key is None:
            api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            raise ValueError("Please set FRED_API_KEY environment variable or pass it explicitly.")
        
        fred = Fred(api_key=api_key)
        data_frames = []
        
        for series_id in series_ids:
            series_data = fred.get_series(series_id, start_date, end_date)
            series_data.name = series_id
            data_frames.append(series_data)
        
        result = pd.concat(data_frames, axis=1).sort_index()
        logger.info(f"Fetched FRED data for series: {', '.join(series_ids)}")
        return result
    except Exception as e:
        logger.error(f"Error fetching FRED data: {e}")
        raise