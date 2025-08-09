# src/analysis/risk.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_portfolio_variance(returns: pd.DataFrame, weights: Optional[np.ndarray] = None) -> float:
    """
    Calculate portfolio variance as part of econometric risk analysis based on stock returns covariance.
    
    Args:
        returns (pd.DataFrame): DataFrame of daily returns (rows: dates, columns: stocks).
        weights (Optional[np.ndarray], optional): Portfolio weights (array of length n_stocks). Defaults to equal weights.
    
    Returns:
        float: Portfolio variance, representing total risk.
    
    Raises:
        ValueError: If weights length doesn't match number of stocks or returns data is invalid.
        Exception: If calculation fails.
    """
    try:
        if returns.empty or returns.isna().all().all():
            raise ValueError("Returns DataFrame is empty or contains only NaNs.")
        
        cov_matrix = returns.cov()
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        if len(weights) != len(returns.columns):
            raise ValueError("Weights length must match number of stocks.")
        
        portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        logger.info(f"Portfolio annualized volatility calculated: {portfolio_variance:.4f}")
        return portfolio_variance
    except Exception as e:
        logger.error(f"Error calculating portfolio variance: {e}")
        raise

def analyze_pca_risk_factors(
    pca: PCA,
    components_df: pd.DataFrame,
    explained_variance: np.ndarray,
    top_n: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Analyze econometric risk factors from PCA by identifying top contributing stocks per principal component.
    Supports portfolio risk assessment by highlighting key drivers of variance.
    
    Args:
        pca (PCA): Fitted PCA model from econometric analysis.
        components_df (pd.DataFrame): DataFrame of component loadings (rows: stocks, columns: components).
        explained_variance (np.ndarray): Explained variance ratios from PCA.
        top_n (int, optional): Number of top contributing stocks to return per component. Defaults to 5.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: DataFrame of top contributors per component, and Series of cumulative explained variance.
    
    Raises:
        ValueError: If inputs are invalid or inconsistent.
        Exception: If analysis fails.
    """
    try:
        if components_df.shape[0] == 0 or len(explained_variance) == 0:
            raise ValueError("Invalid PCA components or explained variance data.")
        
        top_contributors = {}
        for col in components_df.columns:
            top_stocks = components_df[col].abs().sort_values(ascending=False).head(top_n).index.tolist()
            top_contributors[col] = top_stocks
        
        top_contributors_df = pd.DataFrame(top_contributors)
        
        cumulative_variance = pd.Series(
            np.cumsum(explained_variance),
            index=[f'PC{i+1}' for i in range(len(explained_variance))]
        )
        
        logger.info(f"Analyzed top {top_n} risk factors per principal component, explaining {cumulative_variance.iloc[-1]:.2%} of variance.")
        return top_contributors_df, cumulative_variance
    except Exception as e:
        logger.error(f"Error analyzing PCA risk factors: {e}")
        raise

def calculate_value_at_risk(
    returns: pd.DataFrame,
    confidence_level: float = 0.95,
    time_horizon: int = 1
) -> float:
    """
    Calculate Value-at-Risk (VaR) as an additional econometric risk metric for the portfolio.
    
    Args:
        returns (pd.DataFrame): DataFrame of daily returns (rows: dates, columns: stocks).
        confidence_level (float, optional): Confidence level for VaR (e.g., 0.95 for 95%). Defaults to 0.95.
        time_horizon (int, optional): Time horizon in days. Defaults to 1.
    
    Returns:
        float: Value-at-Risk at the specified confidence level.
    
    Raises:
        ValueError: If returns data is invalid or confidence level is out of range.
        Exception: If calculation fails.
    """
    try:
        if returns.empty or returns.isna().all().all():
            raise ValueError("Returns DataFrame is empty or contains only NaNs.")
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
        
        portfolio_returns = returns.mean(axis=1)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        z_score = np.percentile(np.random.normal(0, 1, 1000000), (1 - confidence_level) * 100)
        var = mean_return - z_score * std_return * np.sqrt(time_horizon)
        
        logger.info(f"Calculated {confidence_level*100}% VaR for {time_horizon}-day horizon: {var:.4f}")
        return var
    except Exception as e:
        logger.error(f"Error calculating Value-at-Risk: {e}")
        raise