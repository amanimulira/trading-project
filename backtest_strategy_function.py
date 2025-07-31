def backtest_strategy(data):
    """ Backtest trading strategy based on signals."""
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Returns'] * data['Signal'].shift(1)
    cumulative_returns = (1 + data['Strategy_Returns']).cumprod().iloc[-1] -1
    return cumulative_returns