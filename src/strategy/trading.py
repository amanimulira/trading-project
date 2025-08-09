# src/strategy/trading.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

from ..data.fetch import fetch_stock_data
from ..data.preprocess import clean_data, calculate_daily_returns
from ..analysis.pca import PCA  # For type hinting, assuming PCA from sklearn

logger = logging.getLogger(__name__)

def create_pca_basket_weights(components_df: pd.DataFrame) -> pd.Series:
    """
    Create portfolio weights based on the first principal component loadings.
    These weights replicate a basket that mimics the market factor identified by PCA.
    
    Args:
        components_df (pd.DataFrame): DataFrame of PCA component loadings (from get_principal_components).
    
    Returns:
        pd.Series: Normalized weights for the PCA basket (sum to 1 in absolute terms to handle longs/shorts).
    
    Raises:
        ValueError: If 'PC1' column is missing or DataFrame is empty.
        Exception: If weight calculation fails.
    """
    try:
        if 'PC1' not in components_df.columns or components_df.empty:
            raise ValueError("Components DataFrame must contain 'PC1' column and not be empty.")
        
        weights = components_df['PC1']
        weights = weights / weights.abs().sum()  # Normalize to unit exposure
        logger.info("PCA basket weights created successfully.")
        return weights
    except Exception as e:
        logger.error(f"Error creating PCA basket weights: {e}")
        raise

def compute_basket_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Compute the returns of the PCA-based basket portfolio.
    
    Args:
        returns (pd.DataFrame): DataFrame of stock daily returns.
        weights (pd.Series): Portfolio weights aligned with returns columns.
    
    Returns:
        pd.Series: Time series of basket returns.
    
    Raises:
        ValueError: If weights and returns columns do not match.
        Exception: If computation fails.
    """
    try:
        if not returns.columns.equals(weights.index):
            raise ValueError("Weights index must match returns columns.")
        
        basket_returns = returns.dot(weights)
        logger.info("Basket returns computed successfully.")
        return basket_returns
    except Exception as e:
        logger.error(f"Error computing basket returns: {e}")
        raise

def fetch_index_returns(start_date: str, end_date: str, index_ticker: str = '^GSPC') -> pd.Series:
    """
    Fetch and compute daily returns for the market index (e.g., S&P 500).
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        index_ticker (str, optional): Ticker for the index (e.g., '^GSPC' for S&P 500). Defaults to '^GSPC'.
    
    Returns:
        pd.Series: Daily returns of the index.
    
    Raises:
        Exception: If data fetching or processing fails.
    """
    try:
        index_data = fetch_stock_data([index_ticker], start_date, end_date)
        cleaned = clean_data(index_data)
        index_returns = calculate_daily_returns(cleaned)[index_ticker]
        logger.info(f"Index returns fetched for {index_ticker}.")
        return index_returns
    except Exception as e:
        logger.error(f"Error fetching index returns: {e}")
        raise

def compute_spread(basket_returns: pd.Series, index_returns: pd.Series) -> pd.Series:
    """
    Compute the spread between PCA basket returns and index returns.
    
    Args:
        basket_returns (pd.Series): Basket returns.
        index_returns (pd.Series): Index returns.
    
    Returns:
        pd.Series: Spread (basket - index).
    
    Raises:
        ValueError: If series indices do not match.
        Exception: If computation fails.
    """
    try:
        if not basket_returns.index.equals(index_returns.index):
            raise ValueError("Basket and index returns must have matching indices.")
        
        spread = basket_returns - index_returns
        logger.info("Spread computed successfully.")
        return spread
    except Exception as e:
        logger.error(f"Error computing spread: {e}")
        raise

def generate_mean_reversion_signals(
    spread: pd.Series,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5
) -> pd.Series:
    """
    Generate trading signals based on mean-reversion of the spread using z-scores.
    
    Signals:
    - 1: Long the spread (long basket, short index)
    - -1: Short the spread (short basket, long index)
    - 0: No position or exit
    
    Args:
        spread (pd.Series): Spread time series.
        window (int, optional): Rolling window for mean and std. Defaults to 20.
        entry_z (float, optional): Z-score threshold for entry. Defaults to 2.0.
        exit_z (float, optional): Z-score threshold for exit. Defaults to 0.5.
    
    Returns:
        pd.Series: Trading signals.
    
    Raises:
        ValueError: If window is invalid or spread is too short.
        Exception: If signal generation fails.
    """
    try:
        if window <= 0 or len(spread) < window:
            raise ValueError("Window must be positive and less than spread length.")
        
        rolling_mean = spread.rolling(window).mean()
        rolling_std = spread.rolling(window).std()
        z_score = (spread - rolling_mean) / rolling_std
        
        signals = pd.Series(0, index=spread.index)
        position = 0
        
        for i in range(len(z_score)):
            z = z_score.iloc[i]
            if position == 0:
                if z > entry_z:
                    signals.iloc[i] = -1
                    position = -1
                elif z < -entry_z:
                    signals.iloc[i] = 1
                    position = 1
            elif position == 1:
                if z > -exit_z:
                    signals.iloc[i] = -1  # Exit by closing (signal to adjust)
                    position = 0
            elif position == -1:
                if z < exit_z:
                    signals.iloc[i] = 1  # Exit
                    position = 0
        
        logger.info("Mean-reversion signals generated.")
        return signals
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        raise

def backtest_strategy(
    spread: pd.Series,
    signals: pd.Series,
    transaction_cost: float = 0.001,
    risk_free_rate: float = 0.0
) -> Tuple[pd.Series, float, float]:
    """
    Backtest the trading strategy on the spread.
    
    Args:
        spread (pd.Series): Spread returns.
        signals (pd.Series): Trading signals.
        transaction_cost (float, optional): Cost per trade (as fraction). Defaults to 0.001.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.0.
    
    Returns:
        Tuple[pd.Series, float, float]: Cumulative returns, Sharpe ratio, max drawdown.
    
    Raises:
        ValueError: If inputs are mismatched.
        Exception: If backtest fails.
    """
    try:
        if not spread.index.equals(signals.index):
            raise ValueError("Spread and signals must have matching indices.")
        
        # Portfolio returns = signals.shift(1) * spread
        portfolio_returns = signals.shift(1).fillna(0) * spread
        
        # Apply transaction costs on position changes
        position_changes = signals.diff().abs().fillna(0)
        portfolio_returns -= position_changes * transaction_cost
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Sharpe ratio (annualized)
        mean_ret = portfolio_returns.mean() * 252
        std_ret = portfolio_returns.std() * np.sqrt(252)
        sharpe = (mean_ret - risk_free_rate) / std_ret if std_ret != 0 else 0.0
        
        # Max drawdown
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        logger.info(f"Backtest completed: Sharpe {sharpe:.2f}, Max Drawdown {max_drawdown:.2%}")
        return cumulative_returns, sharpe, max_drawdown
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        raise