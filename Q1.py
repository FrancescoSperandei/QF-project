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
#ritorni giornalier e mensili percentuali, dati Na sostituiti con la media





#media, deviazione, varianza, skew and kurt dei prezzi e dei ritorni giornalieri
daily_summary_table = pd.DataFrame({
    'MeanRtr': log_returns_daily.mean(numeric_only=True),
    'SDRtr': log_returns_daily.std(numeric_only=True),
    'VarRtr': log_returns_daily.var(numeric_only=True),
    'SkewRtr': log_returns_daily.skew(numeric_only=True),
    'KurtRtr': log_returns_daily.kurt(numeric_only=True),
})
print(daily_summary_table)





daily_summary_tableprice = pd.DataFrame({
    'MeanPrc': df_daily.mean(numeric_only=True),
    'SDPrc': df_daily.std(numeric_only=True),
    'VarPrc': df_daily.var(numeric_only=True),
    'SkewPrc': df_daily.skew(numeric_only=True),
    'KurtPrc': df_daily.kurt(numeric_only=True),
})


#media, deviazione, varianza, skew and kurt dei prezzi e dei ritorni mensili

monthly_summary_tablereturn = pd.DataFrame({
    'MeanRtr': log_returns_monthly.mean(numeric_only=True),
    'SDRtr': log_returns_monthly.std(numeric_only=True),
    'VarRtr': log_returns_monthly.var(numeric_only=True),
    'SkewRtr': log_returns_monthly.skew(numeric_only=True),
    'KurtRtr': log_returns_monthly.kurt(numeric_only=True),
})




monthly_summary_tableprice = pd.DataFrame({
    'MeanPrc': df_monthly.mean(numeric_only=True),
    'SDPrc': df_monthly.std(numeric_only=True),
    'VarPrc': df_monthly.var(numeric_only=True), 
    'SkewPrc': df_monthly.skew(numeric_only=True),
    'KurtPrc': df_monthly.kurt(numeric_only=True),
 })

print(daily_summary_tableprice)