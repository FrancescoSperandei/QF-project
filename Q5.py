
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

selected_normal_pricemonthly = df_monthly[selected_stocks]
mu = mean_historical_return (selected_normal_pricemonthly, frequency=12)
S= CovarianceShrinkage(selected_normal_pricemonthly, frequency=12).ledoit_wolf()



ef= EfficientFrontier(mu, S, weight_bounds=(-1,1)) #weight_bounds=(-1,1) per avere anche short
weights = ef.max_sharpe(risk_free_rate=0)
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True, risk_free_rate=0)