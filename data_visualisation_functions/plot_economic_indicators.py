"""

Creating Economic Data Plot

"""
import matplotlib.pyplot as plt

def plot_economic_indicators(data):
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(data.index, data['FEDFUNDS'], label='Fed Funds Rate', color='blue')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Fed Funds Rate', color='blue')

    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.plot(data.index, data['CPI'], label='CPI', color='red')
    ax2.set_ylabel('CPI', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    fig.tight_layout()

    plt.title('Economic Indicators')
    plt.show()

    