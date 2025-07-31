import os
import yfinance as yf
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt

# initialize FRED with API key (using enviroment variable)

fred_api_key = os.getenv('FRED_API_KEY')
if not fred_api_key:
	raise ValueError("Please set FRED_API_KEY environment variable.")
fred = Fred(api_key=fred_api_key)
