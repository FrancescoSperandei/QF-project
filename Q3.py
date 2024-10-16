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
pca = PCA(n_components=2)
principal_components = pca.fit_transform(corr_day)

pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
pca_df['stocks daily'] = log_returns_daily.columns


# Sort the DataFrame by 'PC1' and 'PC2'
pca_df = pca_df.sort_values(['PC1', 'PC2'])



# Plot 'PC1' vs 'PC2'
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')

# Add labels to the points
for i, stock in enumerate(pca_df['stocks daily']):
    plt.annotate(stock, (pca_df['PC1'].iloc[i], pca_df['PC2'].iloc[i]), fontsize=5)

plt.show()
print(pca_df)