def align_data(sp500_data, economic_data):
    combined_data = pd.merge(sp500_data, economic_data, left_index=True, right_index=True, how='outer')
    combined_data['FEDFUNDS'] = combined_data['FEDFUNDS'].ffill()
    combined_data['CPI'] = combined_data['CPI'].ffill()
    return combined_data