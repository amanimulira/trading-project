"""

This function identifies crossovers between the 50-day and 
200-day moving averages to detect trends.

"""

def identify_trends(data):
    data['MA_diff'] = data['MA_50'] - data['MA_200']
    data['crossover'] = 0
    data.loc[(data['MA_diff'] > 0) & (data['MA_diff'].shift(1) < 0), 'crossover'] = 1 # Bullish crossover
    data.loc[(data['MA_diff'] > 0) & (data['MA_diff'].shift(1) > 0), 'crossover'] = -1 # Bearish crossover
    bullish_crossovers = data[data['crossover'] == 1].index
    bearish_crossovers = data[data['crossover'] == -1].index
    return bullish_crossovers, bearish_crossovers