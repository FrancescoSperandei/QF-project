import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kstest
from scipy.stats import shapiro
import seaborn as sns
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from sklearn.decomposition import PCA
from pypfopt import plotting    
from pypfopt import black_litterman, risk_models, BlackLittermanModel, expected_returns
from scipy.optimize import minimize

df = pd.read_excel('sec.xlsx',skiprows=2, parse_dates=['date'], index_col=0, sheet_name=None)



df_daily = df['stocks daily']

df_monthly = df['stocks monthly']




log_returns_daily = np.log(df_daily / df_daily.shift(1))
log_returns_monthly = np.log(df_monthly / df_monthly.shift(1))


# Calculate the daily returns of each stock
returns = log_returns_daily

# Calculate the market return
market_return = returns.mean(axis=1)

# Calculate the beta for each stock
betas = {}
for stock in returns.columns:
    stock_return = returns[stock]
    cov = stock_return.cov(market_return)
    var = market_return.var()
    beta = cov / var
    betas[stock] = beta

annualized_returns = returns.mean() * 252
market_premium = market_return.mean() * 252 - -0.03

betas_range = np.linspace(min(betas.values()), max(betas.values()), 100)
returns_sml = -0.03 + betas_range * market_premium

selected_stocks = ['RATTI', 'BASTOGI', 'MONRIF', 'GAMBERO ROSSO', 'BEEWIZE', 'ECOSUNTEK', 'VIANINI INDR.', 'ENERVIT', 'SABAF', 'ALERION CLEAN POWER']


# Plot the assets
# Plot the assets
plt.figure(figsize=(10, 7))
for asset in selected_stocks:
    if asset in ['ENERVIT', 'VIANINI INDR.']:
        plt.scatter(betas[asset], annualized_returns[asset], marker='*', label=asset, s=150)
    else:
        plt.scatter(betas[asset], annualized_returns[asset], label=asset)

# Plot the SML
plt.plot(betas_range, returns_sml, 'r', label="SML")

plt.xlabel('Beta')
plt.ylabel('Annualized Return')
plt.title('Security Market Line (SML) and Assets Monthy')
plt.legend()
plt.show()
