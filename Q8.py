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


S_daily = CovarianceShrinkage(selected_normal_pricedaily, frequency=252).ledoit_wolf()
mu_daily = mean_historical_return(selected_normal_pricedaily, frequency=252)



ef_daily = EfficientFrontier(mu_daily, S_daily, weight_bounds=(-1, 1))
ef_daily.add_constraint(lambda w: w[0] >= 0.2)
ef_daily.add_constraint(lambda w: w[2] == 0.15)
ef_daily.add_constraint(lambda w: w[3] + w[4] <= 0.10)





ef_daily_max_sharpe = ef_daily.deepcopy()


fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef_daily_max_sharpe, ax=ax, show_assets=True)

# Find the tangency portfolio
ret_tangent, std_tangent, _ = ef_daily_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
#plt.show() this plt.show() need to be runned without the plt.show() in line 74 to work properly

# Generate random portfolios
n_samples = 1000
w = np.random.dirichlet(np.ones(ef_daily.n_assets), n_samples)
rets = w.dot(ef_daily.expected_returns)
stds = np.sqrt(np.diag(w @ ef_daily.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
# Output
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()   #This plt.show() need to be runned without the plt.show() in line 60 to work properly