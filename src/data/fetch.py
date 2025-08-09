# Data Collection and Loading - src/data/fetch.py
import pandas as pd
import yfinance as yf
from typing import List
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
    Fetch historical stock data for given tickers using yfinance, filtering out invalid tickers.
    
    Args:
        tickers (List[str]): List of stock tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str, optional): Data interval (e.g., '1d' for daily). Defaults to '1d'.
    
    Returns:
        pd.DataFrame: Multi-index DataFrame with stock data for valid tickers.
    
    Raises:
        Exception: If data fetching fails or no valid data is returned.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval, group_by='ticker', auto_adjust=True)
        
        # Check for valid tickers (non-empty data)
        valid_tickers = []
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0) and not data[ticker].isna().all().all():
                valid_tickers.append(ticker)
        
        if not valid_tickers:
            raise ValueError("No valid ticker data retrieved.")
        
        # Filter data to include only valid tickers
        if len(valid_tickers) < len(tickers):
            data = data[valid_tickers]
            logger.warning(f"Filtered to {len(valid_tickers)} valid tickers. Invalid: {set(tickers) - set(valid_tickers)}")
        
        logger.info(f"Stock data fetched successfully for {len(valid_tickers)} tickers.")
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        raise