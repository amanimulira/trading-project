from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, data):
        """Generate buy/sell signals for the given data."""
        pass

class GoldenCrossStrategy(Strategy):
    def generate_signals(self, data):
        """Generate buy/sell signals based on moving average crossovers."""
        data['Signal'] = 0
        data.loc[data['crossover'] == 1, 'Signal'] = 1  # Buy signal
        data.loc[data['crossover'] == -1, 'Signal'] = -1  # Sell signal
        return data

class MeanReversionStrategy(Strategy):
    def generate_signals(self, data):
        """Buy when price is >1 std below 20-day MA, sell when >1 std above."""
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['Std_20'] = data['Close'].rolling(window=20).std()
        data['Signal'] = 0
        data.loc[data['Close'] < (data['MA_20'] - data['Std_20']), 'Signal'] = 1  # Buy
        data.loc[data['Close'] > (data['MA_20'] + data['Std_20']), 'Signal'] = -1  # Sell
        return data

class MomentumStrategy(Strategy):
    def generate_signals(self, data):
        """Buy on positive 10-day returns, sell on negative."""
        data['Returns_10'] = data['Close'].pct_change(periods=10)
        data['Signal'] = 0
        data.loc[data['Returns_10'] > 0.02, 'Signal'] = 1  # Buy
        data.loc[data['Returns_10'] < -0.02, 'Signal'] = -1  # Sell
        return data