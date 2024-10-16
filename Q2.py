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


corr_day = log_returns_daily.corr()
fig, ax = plt.subplots(figsize=(20, 20))  # Adjust size of figure
sns.heatmap(corr_day, cmap='plasma', cbar=True, ax=ax, vmin=-1, vmax=1,
            xticklabels=True, yticklabels=True)  # Add color bar
plt.title('Correlation Matrix Heatmap of Daily Logaritmic Returns')
plt.yticks(rotation=0)  # Rotate y-axis labels
plt.xticks(fontsize=5)  # Reduce x-axis font size
plt.yticks(fontsize=5)  # Reduce y-axis font size





#calculate the covariance matrix
#print(returndaily)

covariance_matrix_daily = log_returns_daily.cov()
covariance_matrix_monthly = log_returns_monthly.cov()

#visualize the covariance matrix
print("daily covariance Matrix:")
print(covariance_matrix_monthly)

#create a heatmap
plt.figure(figsize=(30, 30))
sns.heatmap(covariance_matrix_monthly, cmap= 'plasma', annot=False, linewidths=.5, vmin=-0.000616, vmax=0.02, xticklabels=True, yticklabels=True)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.title('Heatmap of monthly Covariance Matrix')
plt.show()
#THIS WAS THE PART FOR COVARIANCE-VARIANCE MATRIX BOTH DAILY AND MONTHLY


print(corr_day)