import pandas as pd

def analyze_economic_impact(sp500_data, economic_data):
    """Analyze correlation between S&P 500 and economic indicators."""

    # Ensure both indices are timezone-naive
    sp500_data.index = sp500_data.index.tz_localize(None)
    economic_data.index = economic_data.index.tz_localize(None)

    combined_data = pd.merge(
        sp500_data[['Close']],
        economic_data,
        left_index=True,
        right_index=True,
        how="outer"
    ).ffill()

    correlations = combined_data.corr()
    return correlations
