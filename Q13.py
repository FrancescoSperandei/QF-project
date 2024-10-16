
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

market_price = df_monthly[selected_stocks]

# Calculate the expected returns and the annualized sample covariance matrix of asset returns


# Calculate the expected returns and the annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(market_price, frequency=12)
S = risk_models.CovarianceShrinkage(market_price, frequency=12).ledoit_wolf()

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


Q = pd.Series([0.15, 0.10, 0.05, 0.12, 0.08, 0.1], index=P.index)






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

# Create the Black-Litterman model
bl1 = BlackLittermanModel(S, pi=market_prior, P=P, Q=Q, omega="idzorek", view_confidences=confidences, market_caps=mcaps)
prior_mean = mu + 1 * mu.std()
prior_cov = 2 * S
# Update the model with the prior

bl2 = BlackLittermanModel(prior_cov, pi=prior_mean, P=P, Q=Q, omega="idzorek", view_confidences=confidences, market_caps=mcaps)
# Generate the posterior estimates for the returns and covariance matrix
posterior_estimates = bl2.bl_returns()
posterior_covariance = bl2.bl_cov()
print(posterior_covariance)
# Convert posterior_covariance to a pandas DataFrame
posterior_cov_df = pd.DataFrame(posterior_covariance)

# Export the DataFrame as a LaTeX table


# Write the LaTeX table to a file



# Create the Efficient Frontier
efbay = EfficientFrontier(posterior_estimates, posterior_covariance)
efbay.add_objective(objective_functions.L2_reg)
efbay.max_sharpe()
weightsbay = efbay.clean_weights()


# Convert weightsbay to a pandas DataFrame
weightsbay_df = pd.DataFrame(list(weightsbay.items()), columns=['Asset', 'Weight'])


# Compute the mean and standard deviation for the standard Bayesian model




# Compute the mean and standard deviation for the standard Bayesian model
mean, std, sharpe_ratio = efbay.portfolio_performance(verbose=False)

# Compute variance, skewness, and kurtosis
variance = std ** 2
skewness = posterior_estimates.skew()
kurtosis = posterior_estimates.kurt()

# Create a DataFrame
databay = {
    "Mean": [mean],
    "Standard Deviation": [std],
    "Variance": [variance],
    "Skewness": [skewness],
    "Kurtosis": [kurtosis],
    "Sharpe Ratio": [sharpe_ratio]
}

df_statsbay = pd.DataFrame(databay)

print(df_statsbay)




# Initialize results array
num_portfolios = 1000
results = np.zeros((3, num_portfolios))


# Generate random portfolios
for i in range(num_portfolios):
    weights = np.random.random(len(posterior_estimates))
    weights /= np.sum(weights)
    weights_dict = {asset: weight for asset, weight in zip(selected_stocks, weights)}
    ef = EfficientFrontier(posterior_estimates, posterior_covariance, weight_bounds=(0, 1))
    ef.set_weights(weights_dict)
    perf = ef.portfolio_performance(verbose=False)
    results[0, i] = perf[0]  # Return
    results[1, i] = perf[1]  # Volatility
    results[2, i] = perf[2]  # Sharpe Ratio
# Plot the random portfolios
fig, ax = plt.subplots()
scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
plt.colorbar(scatter, label='Sharpe Ratio')

# Calculate and plot the efficient frontier for the portfolio with max Sharpe ratio
ef = EfficientFrontier(posterior_estimates, posterior_covariance)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)

# Create a new EfficientFrontier object for plotting
ef_for_plotting = EfficientFrontier(posterior_estimates, posterior_covariance)
plotting.plot_efficient_frontier(ef_for_plotting, ax=ax, show_assets=False)

plt.title('Efficient Frontier with Bayesian Allocation')
plt.xlabel('Volatility')
plt.ylabel('Expected Returns')
plt.show()
