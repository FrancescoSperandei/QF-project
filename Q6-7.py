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

selected_normal_pricedaily = df_daily[selected_stocks]
selected_normal_pricemonthly = df_monthly[selected_stocks]

# Calculate statistics for daily data
S_daily = CovarianceShrinkage(selected_normal_pricedaily, frequency=252).ledoit_wolf()
mu_daily = mean_historical_return(selected_normal_pricedaily, frequency=252)
print(S_daily)

# Setup EfficientFrontier for daily data
ef_daily = EfficientFrontier(mu_daily, S_daily)
weights_daily = ef_daily.max_sharpe(risk_free_rate=0)
cleaned_weights_daily = ef_daily.clean_weights()
perf_daily = ef_daily.portfolio_performance(verbose=True, risk_free_rate=0)


S_monthly = CovarianceShrinkage(selected_normal_pricemonthly, frequency=12).ledoit_wolf()
mu_monthly = mean_historical_return(selected_normal_pricemonthly, frequency=12)  # Assuming monthly frequency

# Setup EfficientFrontier for monthly data
ef_monthly = EfficientFrontier(mu_monthly, S_monthly)
ef_monthly.add_objective(objective_functions.L2_reg, gamma=3)
weights_monthly = ef_monthly.max_sharpe(risk_free_rate=0)
cleaned_weights_monthly = ef_monthly.clean_weights()


perf_monthly = ef_monthly.portfolio_performance(verbose=True, risk_free_rate=0)


# Create DataFrame for weights
weights_df = pd.DataFrame({
    'weights_daily': cleaned_weights_daily,
    'weights_monthly': cleaned_weights_monthly,
})

# Create DataFrame for performance
perf_df = pd.DataFrame({
    'perf_daily': perf_daily,
    'perf_monthly': perf_monthly
}, index=['Expected Return', 'Volatility', 'Sharpe Ratio'])

# Concatenate the DataFrames
unc_port_perf = pd.concat([weights_df, perf_df])


selected_ln_pricedaily = np.log(selected_normal_pricedaily)
selected_ln_pricemonthly = np.log(selected_normal_pricemonthly)

portfolio_returns_daily = (selected_ln_pricedaily * cleaned_weights_daily).sum(axis=1)
portfolio_returns_monthly = (selected_ln_pricemonthly * cleaned_weights_monthly).sum(axis=1)

daily_data_statdf = {
    'Mean': portfolio_returns_daily.mean(),
    'Variance': portfolio_returns_daily.var(),
    'Standard Deviation': portfolio_returns_daily.std(),
    'Skewness': portfolio_returns_daily.skew(),
    'Kurtosis': portfolio_returns_daily.kurt()
}

monthly_data_statdf = {
    'Mean': portfolio_returns_monthly.mean(),
    'Variance': portfolio_returns_monthly.var(),
    'Standard Deviation': portfolio_returns_monthly.std(),
    'Skewness': portfolio_returns_monthly.skew(),
    'Kurtosis': portfolio_returns_monthly.kurt()
}