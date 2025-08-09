# src/strategy/__init__.py
from .trading import (
    create_pca_basket_weights,
    compute_basket_returns,
    fetch_index_returns,
    compute_spread,
    generate_mean_reversion_signals,
    backtest_strategy
)

__all__ = [
    'create_pca_basket_weights',
    'compute_basket_returns',
    'fetch_index_returns',
    'compute_spread',
    'generate_mean_reversion_signals',
    'backtest_strategy'
]