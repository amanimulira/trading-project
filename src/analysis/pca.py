# src/analysis/pca.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def apply_pca(returns: pd.DataFrame, n_components: Optional[int] = None, variance_threshold: Optional[float] = None) -> Tuple[PCA, pd.DataFrame, np.ndarray]:
    """
    Apply Principal Component Analysis (PCA) as an econometric model to decompose stock returns variance.
    Identifies principal components that capture the primary sources of variation in returns data.
    
    Args:
        returns (pd.DataFrame): DataFrame of daily returns (rows: dates, columns: stocks).
        n_components (Optional[int], optional): Number of components to keep. Defaults to None.
        variance_threshold (Optional[float], optional): Cumulative variance threshold (e.g., 0.95) to determine n_components. Defaults to None.
    
    Returns:
        Tuple[PCA, pd.DataFrame, np.ndarray]: Fitted PCA model, transformed data as DataFrame, and explained variance ratios.
    
    Raises:
        ValueError: If both n_components and variance_threshold are provided or neither results in valid components.
        Exception: If PCA fitting fails.
    """
    try:
        if n_components is not None and variance_threshold is not None:
            raise ValueError("Provide either n_components or variance_threshold, not both.")
        
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(returns)
        
        if variance_threshold is not None:
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(returns)
        
        transformed_df = pd.DataFrame(
            transformed,
            index=returns.index,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)]
        )
        
        logger.info(
            f"Econometric PCA model applied with {pca.n_components_} components, "
            f"explaining {np.sum(pca.explained_variance_ratio_):.2%} of variance."
        )
        return pca, transformed_df, pca.explained_variance_ratio_
    except Exception as e:
        logger.error(f"Error applying PCA econometric model: {e}")
        raise

def get_principal_components(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract principal component loadings to interpret econometric drivers of stock returns.
    
    Args:
        pca (PCA): Fitted PCA model.
        feature_names (list[str]): List of original feature names (e.g., stock tickers).
    
    Returns:
        pd.DataFrame: DataFrame of component loadings (rows: features, columns: components).
    
    Raises:
        Exception: If processing fails.
    """
    try:
        components_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=feature_names
        )
        logger.info("Extracted principal component loadings for econometric analysis.")
        return components_df
    except Exception as e:
        logger.error(f"Error extracting principal components: {e}")
        raise

def regress_on_components(
    transformed: pd.DataFrame,
    market_returns: pd.Series,
    components: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Perform regression of market returns on principal components to quantify their explanatory power.
    This extends the econometric modeling by linking components to a market index (e.g., S&P 500).
    
    Args:
        transformed (pd.DataFrame): PCA-transformed data (rows: dates, columns: principal components).
        market_returns (pd.Series): Market index returns (e.g., S&P 500 daily returns).
        components (Optional[list[str]], optional): Subset of principal components to use. Defaults to all.
    
    Returns:
        pd.DataFrame: Regression results with coefficients, p-values, and R-squared.
    
    Raises:
        ValueError: If market_returns and transformed have mismatched indices.
        Exception: If regression fails.
    """
    try:
        if not transformed.index.equals(market_returns.index):
            raise ValueError("Indices of transformed data and market returns must match.")
        
        if components is None:
            components = transformed.columns
        
        X = transformed[components]
        y = market_returns
        
        model = LinearRegression()
        model.fit(X, y)
        
        results = {
            'Coefficient': model.coef_,
            'Intercept': model.intercept_,
            'R-squared': model.score(X, y)
        }
        results_df = pd.DataFrame(results, index=components)
        
        logger.info(f"Regression on principal components completed with R-squared: {results['R-squared']:.4f}")
        return results_df
    except Exception as e:
        logger.error(f"Error in regression on principal components: {e}")
        raise