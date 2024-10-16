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



colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple', 'pink', 'brown']

df_daily[['RATTI', 'BASTOGI', 'MONRIF', 'GAMBERO ROSSO', 'BEEWIZE', 'ECOSUNTEK', 'VIANINI INDR.', 'ENERVIT', 'SABAF', 'ALERION CLEAN POWER']].plot(color=colors)
plt.title('Daily price')
plt.xlabel('Date')
plt.ylabel('price')
plt.show()

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple', 'pink', 'brown']

df_monthly[['RATTI', 'BASTOGI', 'MONRIF', 'GAMBERO ROSSO', 'BEEWIZE', 'ECOSUNTEK', 'VIANINI INDR.', 'ENERVIT', 'SABAF', 'ALERION CLEAN POWER']].plot(color=colors)
plt.title('Monthly price')
plt.xlabel('Date')
plt.ylabel('price')
plt.show()



df_stocks_daily = pd.DataFrame({
    'ILLIMITY BANK': df_daily['ILLIMITY BANK'],
    'AQUAFIL': df_daily['AQUAFIL'],
    'VALSOIA': df_daily['VALSOIA'],
    'PIRELLI': df_daily['PIRELLI'],
    'HERA': df_daily['HERA'],
    'ENEL': df_daily['ENEL'],
    'DAVIDE CAMPARI MILANO': df_daily['DAVIDE CAMPARI MILANO'],
    'BUZZI': df_daily['BUZZI'],
    'CEMBRE': df_daily['CEMBRE'],
    'AMPLIFON': df_daily['AMPLIFON'],
    'STELLANTIS': df_daily['STELLANTIS'],
    'ALERION CLEAN POWER': df_daily['ALERION CLEAN POWER'],
})


df_stocks_monthly = pd.DataFrame({
    'ILLIMITY BANK': df_monthly['ILLIMITY BANK'],
    'AQUAFIL': df_monthly['AQUAFIL'],
    'VALSOIA': df_monthly['VALSOIA'],
    'PIRELLI': df_monthly['PIRELLI'],
    'HERA': df_monthly['HERA'],
    'ENEL': df_monthly['ENEL'],
    'DAVIDE CAMPARI MILANO': df_monthly['DAVIDE CAMPARI MILANO'],
    'BUZZI': df_monthly['BUZZI'],
    'CEMBRE': df_monthly['CEMBRE'],
    'AMPLIFON': df_monthly['AMPLIFON'],
    'STELLANTIS': df_monthly['STELLANTIS'],
    'ALERION CLEAN POWER': df_monthly['ALERION CLEAN POWER'],
})