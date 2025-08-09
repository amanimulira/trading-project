# main.py
import argparse
import logging
import yaml
import pandas as pd

from src.data import get_sp500_tickers, fetch_stock_data, clean_data, calculate_daily_returns
from src.analysis import apply_pca, get_principal_components, calculate_portfolio_variance, analyze_pca_risk_factors, calculate_value_at_risk
from src.strategy import create_pca_basket_weights, compute_basket_returns, fetch_index_returns, compute_spread, generate_mean_reversion_signals, backtest_strategy

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
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
    cleaned_data = clean_data(
        data,
        fill_method=config['data']['fill_method'],
        dropna=config['data']['dropna'],
        price_col=config['data']['price_col']
    )
    returns = calculate_daily_returns(cleaned_data, price_col=None)
    
    # Check if returns DataFrame is valid
    if returns.empty or returns.shape[0] < 1:
        raise ValueError(f"Returns DataFrame is empty or invalid: {returns.shape}")
    
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
        pca=pca,
        components_df=components_df,
        explained_variance=explained_variance,
        top_n=config['risk']['top_n']
    )
    var_95 = calculate_value_at_risk(
        returns, 
        confidence_level=config['risk']['confidence_level'],
        time_horizon=config['risk']['time_horizon']
    )

    logger.info(f"Portfolio Annualized Volatility: {portfolio_vol:.4f}")
    logger.info(f"{config['risk']['confidence_level']*100}% {config['risk']['time_horizon']}-Day VaR: {var_95:.4f}")
    logger.info("Top PCA Risk Factors:\n" + top_contributors.to_string())
    logger.info("Cumulative Variance Explained:\n" + cumulative_variance.to_string())

    # Step 3: Strategy Development and Backtesting
    logger.info("Starting trading strategy...")
    basket_weights = create_pca_basket_weights(components_df)
    basket_returns = compute_basket_returns(returns, basket_weights)

    index_returns = fetch_index_returns(
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        index_ticker=config['strategy']['index_ticker']
    )

    spread = compute_spread(basket_returns, index_returns)
    signals = generate_mean_reversion_signals(
        spread,
        window=config['strategy']['window'],
        entry_z=config['strategy']['entry_z'],
        exit_z=config['strategy']['exit_z']
    )
    cumulative_returns, sharpe, max_drawdown = backtest_strategy(
        spread,
        signals,
        transaction_cost=config['strategy']['transaction_cost'],
        risk_free_rate=config['strategy']['risk_free_rate']
    )

    logger.info(f"Strategy Backtest Results: Sharpe Ratio = {sharpe:.2f}, Max Drawdown = {max_drawdown:.2%}")
    logger.info("Cumulative Returns (last 5):\n" + cumulative_returns.tail().to_string())

    # Save results to CSV
    output_dir = config['outputs']['results_dir']
    cumulative_returns.to_csv(f"{output_dir}/strategy_cumulative_returns.csv")
    top_contributors.to_csv(f"{output_dir}/top_pca_contributors.csv")
    logger.info(f"Analysis completed. Results saved to {output_dir}/ directory.")

if __name__ == "__main__":
    main()