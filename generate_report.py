"""

Compile findings into a structured reprot. Automating parts of reprot
using Python.

"""

def generate_report(sp500_data, economic_data, company_evaluations, output_file='sp500_report.md'):

    # Generate a structured markdown report.

    with open(output_file, 'w') as f:
        f.write("# S&P 500 Analysis Report\n\n")
        f.write("## Introduction\n")
        f.write("This report analyzes the S&P 500 to identify trends, evaluate companies, and propose a trading strategy.\n\n")

        f.write("## Market Trends\n")
        bullish, bearish = sp500_data[sp500_data['crossover'] == 1].index, sp500_data[sp500_data['crossover'] == -1].index
        f.write(f"- Bullish Crossovers: {len(bullish)} detected\n")
        f.write(f"- Bearish Crossovers: {len(bearish)} detected\n\n")

        f.write("## Economic Indicators\n")
        f.write(f"- Latest Fed Funds Rate: {economic_data['FEDFUNDS'].iloc[-1]:.2f}%\n")
        f.write(f"- Latest CPI: {economic_data['CPI'].iloc[-1]:.2f}\n\n")

        f.write("## Company Evaluation\n")
        for ticker, eval in company_evaluations.items():
            f.write(f"- {ticker}: {', '.join(eval)}\n")

        f.write("## Trading Strategy\n")
        f.write("Buy on bullish crossovers (50-day MA > 200-day MA), sell on bearish crossovers. Apply 5% stop-loss and 3% position sizing. \n")