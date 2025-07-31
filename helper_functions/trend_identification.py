import pandas as pd

def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def identify_trends(data):
    """Identify bullish and bearish signals using MA crossovers and RSI."""
    data['MA_diff'] = data['MA_50'] - data['MA_200']
    data['crossover'] = 0
    data.loc[(data['MA_diff'] > 0) & (data['MA_diff'].shift(1) < 0), 'crossover'] = 1  # Bullish crossover
    data.loc[(data['MA_diff'] < 0) & (data['MA_diff'].shift(1) > 0), 'crossover'] = -1 # Bearish crossover

    # Calculate RSI
    data = calculate_rsi(data)

    # Confirm crossovers with RSI
    confirmed_bullish = data[(data['crossover'] == 1) & (data['RSI'] < 70)].index
    confirmed_bearish = data[(data['crossover'] == -1) & (data['RSI'] > 30)].index

    return confirmed_bullish, confirmed_bearish
