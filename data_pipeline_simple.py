import os
import yfinance as yf
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt

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
	sp500 = yf.Ticker("^GSPC")
	return sp500.history(start=start_date, end=end_date)

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
	start_date = '2019-01-01'
	end_date = '2023-12-32'

	# Get data
	sp500_data = fetch_sp500_data(start_date, end_date)
	economic_data = fetch_economic_indicators(start_date, end_date)
	companies = ['APPL', 'JNJ', 'JPM']
	company_metrics = {ticker: fetch_company_metrics(ticker) for ticker in companies}

	# Process data
	sp500_data = calculate_moving_averages(sp500_data)

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
