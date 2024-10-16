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


selected_stocks = ['RATTI', 'BASTOGI', 'MONRIF', 'GAMBERO ROSSO', 'BEEWIZE', 'ECOSUNTEK', 'VIANINI INDR.', 'ENERVIT', 'SABAF', 'ALERION CLEAN POWER']

market_price = df_daily[selected_stocks] #to get the BL with Monthly data just change df_daily with df_monthly

# Calculate the expected returns and the annualized sample covariance matrix of asset returns


# Calculate the expected returns and the annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(market_price, frequency=252)
S = risk_models.CovarianceShrinkage(market_price, frequency=252).ledoit_wolf()
P = pd.DataFrame([
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
])

confidences = [
    0.6,
    0.4,
    0.2,
    0.5,
    0.7, 
    0.7 
]

mcaps = {
    'RATTI': 67000000,
    'BASTOGI': 64330000,
    'MONRIF': 10340000,
    'GAMBERO ROSSO': 7370000,
    'BEEWIZE': 5300000,
    'ECOSUNTEK': 36150000,
    'VIANINI INDR.': 84820000,
    'ENERVIT': 56960000,
    'SABAF': 228000000,
    'ALERION CLEAN POWER': 1360000000
}

delta = black_litterman.market_implied_risk_aversion(market_price)
market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)


Q = pd.Series([0.15, 0.10, 0.05, 0.12, 0.08, 0.1], index=P.index)



bl = BlackLittermanModel(S, pi=market_prior, P=P, Q=Q, omega="idzorek", view_confidences=confidences, market_caps=mcaps)

ret_bl = bl.bl_returns()

S_bl=bl.bl_cov()



ef = EfficientFrontier(ret_bl, S_bl)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
weights = ef.clean_weights()


# Assuming 'prices' is your DataFrame with historical prices
returns = ret_bl

# Calculate statistics
mean = returns.mean()
sd = returns.std()
variance = returns.var()
skewness = returns.skew()
kurtosis = returns.kurt()


# Convert weights to a pandas Series
weights_series = pd.Series(weights)

# Convert weights to a pandas DataFrame
weights_df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])


# Calculate portfolio returns
portfolio_returns = (weights_series * mean).sum()
# Calculate portfolio standard deviation
portfolio_sd = (weights_series * sd).sum()

# Calculate portfolio variance
portfolio_variance = (weights_series * variance).sum()

# Calculate portfolio skewness
portfolio_skewness = (weights_series * skewness).sum()

# Calculate portfolio kurtosis
portfolio_kurtosis = (weights_series * kurtosis).sum()

# Calculate Sharpe Ratio
risk_free_rate = 0.03  # adjust this to your risk free rate
sharpe_ratio = (portfolio_returns - risk_free_rate) / portfolio_sd

# Create DataFrame
statsbl_df = pd.DataFrame({
    'Mean': portfolio_returns,
    'SD': portfolio_sd,
    'Variance': portfolio_variance,
    'Skewness': portfolio_skewness,
    'Kurtosis': portfolio_kurtosis,
    'Sharpe Ratio': sharpe_ratio
}, index=[0])
print(statsbl_df)


