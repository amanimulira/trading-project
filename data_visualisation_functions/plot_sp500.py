"""

Creating S&P 500 Plot

"""
import matplotlib.pyplot as plt

def plot_sp500_with_mas(data):
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.plot(data.index, data['MA_50'], label='50-day MA')
    plt.plot(data.index, data['MA_200'], label='200-day MA')

    plt.title('S&P 500 with Moving Averages')

    plt.xlabel('Data')
    plt.ylabel('Price')

    plt.legend()
    plt.show()
