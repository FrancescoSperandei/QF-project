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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis



#QUESTION 10 selected stocks daily
df = pd.read_excel('sec.xlsx',skiprows=2, parse_dates=['date'], index_col=0, sheet_name=None)

df_daily = df['stocks daily']
df_monthly = df['stocks monthly']
# Selected stocks
selected_stocks = ['RATTI', 'BASTOGI', 'MONRIF', 'GAMBERO ROSSO', 'BEEWIZE', 'ECOSUNTEK', 'VIANINI INDR.', 'ENERVIT', 'SABAF', 'ALERION CLEAN POWER']
# Filter DataFrame for selected stocks
df_daily_selected = df_daily[selected_stocks]
df_monthly_selected = df_monthly[selected_stocks]

# Calculate log returns for selected stocks
log_return_daily = np.log(df_daily_selected / df_daily_selected.shift(1))
log_return_monthly = np.log(df_monthly_selected / df_monthly_selected.shift(1))
returns_daily = log_return_daily

# Calculate the market return
market_return_daily = returns_daily.mean(axis=1)

# Calculate the beta for each selected stock
betas = {}
for stock in selected_stocks:
 stock_return = log_return_daily[stock]
cov = stock_return.cov(market_return_daily)
var = market_return_daily.var()
beta_daily = cov / var
betas[stock] = beta_daily

# Print the betas for selected stocks
print("Betas for daily Selected Stocks:")
for stock, beta_daily in betas.items():
 print(f"{stock}: {beta_daily}")


#QUESTION 10 BETA stocks MONTHLY
df = pd.read_excel('sec.xlsx',skiprows=2, parse_dates=['date'], index_col=0, sheet_name=None)

df_daily = df['stocks daily']
df_monthly = df['stocks monthly']
# Selected stocks
selected_stocks = ['RATTI', 'BASTOGI', 'MONRIF', 'GAMBERO ROSSO', 'BEEWIZE', 'ECOSUNTEK', 'VIANINI INDR.', 'ENERVIT', 'SABAF', 'ALERION CLEAN POWER']
# Filter DataFrame for selected stocks
df_daily_selected = df_daily[selected_stocks]
df_monthly_selected = df_monthly[selected_stocks]

# Calculate log returns for selected stocks
log_return_daily = np.log(df_daily_selected / df_daily_selected.shift(1))
log_return_monthly = np.log(df_monthly_selected / df_monthly_selected.shift(1))
returns_monthly = log_return_monthly

# Calculate the market return
market_return_monthly = returns_monthly.mean(axis=1)

# Calculate the beta for each selected stock
betas = {}
for stock in selected_stocks:
 stock_return = log_return_monthly[stock]
cov = stock_return.cov(market_return_monthly)
var = market_return_monthly.var()
beta_monthly = cov / var
betas[stock] = beta_monthly

# Print the betas for selected stocks
print("Betas for monthly Selected Stocks:")
for stock, beta_monthly in betas.items():
 print(f"{stock}: {beta_monthly}")


import numpy as np
import pandas as pd


# Filter DataFrame for selected stocks
df_daily_selected = df_daily[selected_stocks]
df_monthly_selected = df_monthly[selected_stocks]

# Calculate log returns for selected stocks
log_return_daily_selected = np.log(df_daily_selected / df_daily_selected.shift(1))
log_return_monthly_selected = np.log(df_monthly_selected / df_monthly_selected.shift(1))
returns_daily_selected = log_return_daily_selected # Use daily returns
returns_monthly_selected = log_return_monthly_selected # Use monthly returns

# Calculate the market return for selected stocks
market_return_daily_selected = returns_daily_selected.mean(axis=1)
market_return_monthly_selected = returns_monthly_selected.mean(axis=1)

weights_daily_selected = [0.205830, -0.490990, -0.533500, 0.017170, -0.224610, 0.218070, 0.075230, 0.235480, 0.497400, 1]
weights_monthly_selected = [0.205210, -0.381610, -0.453630, 0.048020, -0.276530, 0.307000, 0.076580, 0.140650, 0.334300, 1]

# Calculate weighted average beta for the entire portfolio and divide by 10
portfolio_beta_daily = np.sum(np.multiply(weights_daily_selected, beta_daily)) / 10
portfolio_beta_monthly = np.sum(np.multiply(weights_monthly_selected, beta_monthly)) / 10

# Print the weighted average beta for the entire portfolio
print("Portfolio Beta (Daily):", portfolio_beta_daily)
print("Portfolio Beta (Monthly):", portfolio_beta_monthly)

results_df = pd.DataFrame({
'Stock': selected_stocks,
'Beta (Daily)': [betas[stock] for stock in selected_stocks],
'Beta (Monthly)': [betas[stock] for stock in selected_stocks],
'Weight (Daily)': weights_daily_selected,
'Weight (Monthly)': weights_monthly_selected
})

# Add a row for portfolio beta
portfolio_row = pd.DataFrame({
'Stock': ['Portfolio'],
'Beta (Daily)': [portfolio_beta_daily],
'Beta (Monthly)': [portfolio_beta_monthly],
'Weight (Daily)': [sum(weights_daily_selected)],
'Weight (Monthly)': [sum(weights_monthly_selected)]
})

results_df = pd.concat([results_df, portfolio_row], ignore_index=True)
