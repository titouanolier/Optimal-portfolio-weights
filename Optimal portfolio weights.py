import yfinance as yf
import numpy as np
from scipy.optimize import minimize


# Define the stock tickers
# Replace these tickers with the ones you're interested in
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JNJ', 'V', 'PG', 'NVDA', 'DIS', 'PEP', 'HD', 'INTC', 'CSCO', 'CMCSA', 'PFE', 'NFLX', 'KO', 'NKE', 'MRK', 'T', 'ABT', 'ORCL', 'CRM', 'ABBV', 'ACN', 'VZ', 'WMT']

# Download stock data
data = yf.download(tickers, start="2010-01-01", end="2023-01-01")['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Number of stocks
n = len(tickers)

# Initial allocation (equal distribution)
init_alloc = np.array(n * [1./n])

# Portfolio returns calculation
def portfolio_returns(alloc, returns):
    return np.dot(returns, alloc)

# Portfolio VaR calculation (95% confidence)
def portfolio_var(alloc, returns):
    port_returns = portfolio_returns(alloc, returns)
    return np.percentile(port_returns, 5)

# Constraint: allocations sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds for allocations: between 0 and 1 for each stock
bounds = tuple((0, 1) for _ in range(n))

# Objective function to minimize (negative VaR to maximize VaR)
def min_func_VaR(alloc, returns):
    return -portfolio_var(alloc, returns)

# Optimization
opt_result = minimize(min_func_VaR, init_alloc, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

# Best allocation
best_alloc = opt_result.x

# Print the best allocation
print("Best Allocation:")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {best_alloc[i]:.2f}")

# Calculate and print the portfolio VaR
print(f"Portfolio VaR (95% confidence): {-opt_result.fun:.2%}")
