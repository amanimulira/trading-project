import argparse
import logging 
import yaml
import pandas as pd

from src.data import *
from src.analysis import *
from src.strategy import *

# setup logging 

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load Configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="S&P 500 Trading Analysis Pipeline")
    parser.add_argument("--config", default="src/config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Step 1: Data Collection and Preprocessing
    logger.info("Starting data collection...")
    tickers = get_sp500_tickers()  # Fetch ~500 S&P 500 tickers
    # For performance, optionally limit tickers during testing: tickers = tickers[:50]
    data = fetch_stock_data(
        tickers,
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    cleaned_data = clean_data(data, fill_method='ffill', dropna=True)
    returns = calculate_daily_returns(cleaned_data, price_col='Adj Close')
    logger.info(f"Data preprocessed: {returns.shape[0]} days, {returns.shape[1]} stocks.")

    # Step 2: Econometric Analysis (PCA and Risk)
    logger.info("Starting econometric analysis...")
    pca, transformed, explained_variance = apply_pca(
        returns,
        variance_threshold=config['pca']['variance_threshold']
    )
    components_df = get_principal_components(pca, returns.columns.tolist())

    portfolio_vol = calculate_portfolio_variance(returns)
    top_contributors, cumulative_variance = analyze_pca_risk_factors(
        pca, components_df, explained_variance, top_n=5
    )
    var_95 = calculate_value_at_risk(returns, confidence_level=0.95)

    logger.info(f"Portfolio Annualized Volatility: {portfolio_vol:.4f}")
    logger.info(f"95% 1-Day VaR: {var_95:.4f}")
    logger.info("Top PCA Risk Factors:\n" + top_contributors.to_string())
    logger.info("Cumulative Variance Explained:\n" + cumulative_variance.to_string())

    # Step 3: Strategy Development and Backtesting
    logger.info("Starting trading strategy...")
    basket_weights = create_pca_basket_weights(components_df)
    basket_returns = compute_basket_returns(returns, basket_weights)

    index_returns = fetch_index_returns(
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        index_ticker='^GSPC'
    )

    spread = compute_spread(basket_returns, index_returns)
    signals = generate_mean_reversion_signals(
        spread,
        window=20,
        entry_z=2.0,
        exit_z=0.5
    )
    cumulative_returns, sharpe, max_drawdown = backtest_strategy(
        spread,
        signals,
        transaction_cost=0.001,
        risk_free_rate=0.0
    )

    logger.info(f"Strategy Backtest Results: Sharpe Ratio = {sharpe:.2f}, Max Drawdown = {max_drawdown:.2%}")
    logger.info("Cumulative Returns (last 5):\n" + cumulative_returns.tail().to_string())

    # Optional: Save results to CSV
    cumulative_returns.to_csv('outputs/strategy_cumulative_returns.csv')
    top_contributors.to_csv('outputs/top_pca_contributors.csv')
    logger.info("Analysis completed. Results saved to outputs/ directory.")

if __name__ == "__main__":
    main()