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

market_price_daily = df_daily[selected_stocks]
market_price_monthly = df_monthly[selected_stocks]
# Calculate the expected returns and the annualized sample covariance matrix of asset returns


# Calculate the expected returns and the annualized sample covariance matrix of asset returns
mu_daily = expected_returns.mean_historical_return(market_price_daily, frequency=252)
S_daily = risk_models.CovarianceShrinkage(market_price_daily, frequency=252).ledoit_wolf()
efgmv_daily = EfficientFrontier(mu_daily, S_daily, weight_bounds=(-1, 1))

weightsgmv_daily = efgmv_daily.min_volatility()
cleaned_weightsgmv_daily = efgmv_daily.clean_weights()
print('\n daily')
perfgmv_daily = efgmv_daily.portfolio_performance(verbose=True)


mu_mont = expected_returns.mean_historical_return(market_price_monthly, frequency=12)
S_mont = risk_models.CovarianceShrinkage(market_price_monthly, frequency=12).ledoit_wolf()
efgmv_mont = EfficientFrontier(mu_mont, S_mont, weight_bounds=(-1, 1))

weightsgmv_mont = efgmv_mont.min_volatility()
cleaned_weightsgmv_mont = efgmv_mont.clean_weights()

perfgmv_mont = efgmv_mont.portfolio_performance(verbose=True)

import scipy.stats as stats

# Calculate skewness and kurtosis for daily and monthly portfolios
# Assuming gmv_returns_daily and gmv_returns_mont are your daily and monthly portfolio returns

# Calculate skewness and kurtosis for daily and monthly portfolio returns
skewness_daily = stats.skew(mu_daily)
kurtosis_daily = stats.kurtosis(mu_daily)
skewness_mont = stats.skew(mu_mont)
kurtosis_mont = stats.kurtosis(mu_mont)

# Create a DataFrame with the calculated skewness and kurtosis
stats_df = pd.DataFrame({
    'Daily': [skewness_daily, kurtosis_daily],
    'Monthly': [skewness_mont, kurtosis_mont]
}, index=['Skewness', 'Kurtosis'])



# Convert cleaned_weightsgmv_daily and cleaned_weightsgmv_mont to pandas DataFrames
weights_dailygmv_df = pd.DataFrame.from_dict(cleaned_weightsgmv_daily, orient='index', columns=['Daily'])
weights_montgmv_df = pd.DataFrame.from_dict(cleaned_weightsgmv_mont, orient='index', columns=['Monthly'])

# Join the DataFrames
weightsgmv_df = weights_dailygmv_df.join(weights_montgmv_df)




mean, std, sharpe_ratio = efgmv_mont.portfolio_performance(verbose=False)
datagmvmonth = {
    "Mean": [mean],
    "Standard Deviation": [std],
    "Sharpe Ratio": [sharpe_ratio]
}

datagmvmonthdf=pd.DataFrame(datagmvmonth)
