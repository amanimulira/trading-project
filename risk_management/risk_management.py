def apply_risk_management(data):
    """Apply risk managment with stop-loss and position sizing."""
    data['Stop_Loss'] = data['Close'] * 0.95 # 5% stop-loss
    data['Position_Size'] = 0.03 # 3% of portfolio per trade
    return data