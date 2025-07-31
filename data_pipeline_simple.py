import os
from backtest_strategy_function import backtest_strategy
from generate_report import generate_report
from helper_functions.economic_correlation import analyze_economic_impact
import yfinance as yf
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions.trend_identification import identify_trends
from helper_functions.company_evaluation import evaluate_companies
from data_visualisation_functions.plot_economic_indicators import plot_economic_indicators
from data_visualisation_functions.plot_sp500 import plot_sp500_with_mas
from trading_strategy.trading_strategy import MeanReversionStrategy 
from risk_management.risk_management import apply_risk_management

# Initialize FRED with API key (using enviroment variable)

fred_api_key = os.getenv('FRED_API_KEY')
if not fred_api_key:
	raise ValueError("Please set FRED_API_KEY environment variable.")
fred = Fred(api_key=fred_api_key)

""" 

Data Ingestion Funcitons

"""

def fetch_sp500_data(start_date, end_date):
	# S&P 500 historical data.
	try:
		sp500 = yf.Ticker("^GSPC")
		return sp500.history(start=start_date, end=end_date)
	except Exception as e:
		print(f"Error fetching S&P 500 data: {e}")
		return None

def fetch_economic_indicators(start_date, end_date):
	# Economic indicators from FRED
	fed_rate = fred.get_series('FEDFUNDS', start_date, end_date)
	cpi = fred.get_series('CPIAUCSL', start_date, end_date)
	return pd.DataFrame({'FEDFUNDS': fed_rate, 'CPI': cpi})

def fetch_company_metrics(ticker):
	# Financial metrics for a company
	company = yf.Ticker(ticker)
	info = company.info
	return {
		'P/E Ratio': info.get('trailingPE'),
		'Profit Margin': info.get('profitMargins'),
		'Revenue Growth': info.get('revenueGrowth')
	}

""" 

Data Processing Funcitons

"""

def calculate_moving_averages(data, windows=[50,200]):
	# Calculate moving averages for S&P 500.
	for window in windows:
		data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
	return data

"""

Data Storage Funcition

"""

def save_to_csv(data, filename):
	# Save data to CSV
	data.to_csv(filename)

"""

Main Pipeline Execution

"""

def main():
	start_date = '2010-01-01'
	end_date = '2023-12-31'

	# Get data
	sp500_data = fetch_sp500_data(start_date, end_date)
	economic_data = fetch_economic_indicators(start_date, end_date)
	companies = ['AAPL', 'JNJ', 'JPM']
	company_metrics = {ticker: fetch_company_metrics(ticker) for ticker in companies}

	# evaluations = evaluate_companies(company_metrics)
	# for ticker, eval in evaluations.i():
	# 	print(f"{ticker}: {', '.join(eval)}")

	# Process data
	sp500_data = calculate_moving_averages(sp500_data)

	# Identifing trends
	bullish, bearish = identify_trends(sp500_data)
	print("Bullish Crossovers:", bullish)
	print("Bearish Crossovers:", bearish)

	# Generating signals
	meanReversion = MeanReversionStrategy()
	sp500_data = meanReversion.generate_signals(data=sp500_data)
	print("Trading Signals Sample:")
	print(sp500_data[['Close', 'MA_50', 'MA_200', 'Signal']].tail())

	# Risk Management
	sp500_data = apply_risk_management(sp500_data)
	print("Risk Management Sample:")
	print(sp500_data[['Close', 'Signal', 'Stop_Loss', 'Position_Size']].tail())

	cumulative_return = backtest_strategy(sp500_data)
	print(f"Strategy Cumulative Return: {cumulative_return:.2%}")
	
	evaluations = evaluate_companies(company_metrics)
	generate_report(sp500_data, economic_data, evaluations)

	correlations = analyze_economic_impact(sp500_data, economic_data)
	print("Correlations with S&P 500 Close:")
	print(correlations['Close'])

	# Data Visulaisation
	plot_sp500_with_mas(sp500_data)
	plot_economic_indicators(economic_data)

	# Store data
	save_to_csv(sp500_data, 'sp500_data.csv')
	save_to_csv(economic_data, 'economic_data.csv')

	# Output sample results
	print("S&P 500 Data Sample:")
	print(sp500_data.tail())
	print("\nEconomic Indicators Sample:")
	print(economic_data.tail())
	print('\nCompany Metrics:')
	for ticker, metrics in company_metrics.items():
		print(f"{ticker}: {metrics}")

if __name__ == "__main__":
	main()