"""

This function evaluates companies based on financial 
metrics with simple thresholds.

"""

def evaluate_companies(company_metrics):
    evaluations = {}
    for ticker, metrics in company_metrics.items():
        pe = metrics.get('P/E Ratio')
        pm = metrics.get('Profit Margin')
        rg = metrics.get('Revenue Growth')
        evaluation = []
        if pe and pe < 15:
            evaluation.append("Low P/E ratio")
        if pm and pm > 0.1:
            evaluation.append("Healthy profit margin")
        if rg and rg > 0.05:
            evaluation.append("Strong revenue growth")
        evaluations[ticker] = evaluation
    return evaluation
