# Risk Analysis Functions - src/analysis/risk.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_portfolio_variance(returns: pd.DataFrame, weights: Optional[np.ndarray] = None) -> float:
    """
    Calculate the variance of a portfolio based on stock returns.
    
    Args:
        returns (pd.DataFrame): DataFrame of daily returns (rows: dates, columns: stocks).
        weights (Optional[np.ndarray], optional): Portfolio weights (array of length n_stocks). Defaults to equal weights.
    
    Returns:
        float: Portfolio variance.
    
    Raises:
        ValueError: If weights length doesn't match number of stocks.
        Exception: If calculation fails.
    """
    try:
        cov_matrix = returns.cov()
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        if len(weights) != len(returns.columns):
            raise ValueError("Weights length must match number of stocks.")
        
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        logger.info(f"Portfolio variance calculated: {portfolio_variance:.4f}")
        return portfolio_variance
    except Exception as e:
        logger.error(f"Error calculating portfolio variance: {e}")
        raise

def analyze_pca_risk_factors(pca: PCA, components_df: pd.DataFrame, explained_variance: np.ndarray, top_n: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Analyze key risk factors from PCA results by identifying top contributing stocks per component.
    
    Args:
        pca (PCA): Fitted PCA model.
        components_df (pd.DataFrame): DataFrame of component loadings.
        explained_variance (np.ndarray): Explained variance ratios from PCA.
        top_n (int, optional): Number of top contributors to return per component. Defaults to 5.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: DataFrame of top contributors per component, and Series of cumulative explained variance.
    
    Raises:
        Exception: If analysis fails.
    """
    try:
        top_contributors = {}
        for col in components_df.columns:
            top_stocks = components_df[col].abs().sort_values(ascending=False).head(top_n).index.tolist()
            top_contributors[col] = top_stocks
        
        top_contributors_df = pd.DataFrame(top_contributors)
        
        cumulative_variance = pd.Series(np.cumsum(explained_variance), index=[f'PC{i+1}' for i in range(len(explained_variance))])
        
        logger.info(f"Analyzed top {top_n} risk factors per principal component.")
        return top_contributors_df, cumulative_variance
    except Exception as e:
        logger.error(f"Error analyzing PCA risk factors: {e}")
        raise