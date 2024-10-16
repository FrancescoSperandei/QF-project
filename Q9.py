import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
#import MIB daily file QUESTION 9
market_daily_ftse= pd.read_excel ('index.xlsx',skiprows=2, parse_dates=['date'],index_col=0,sheet_name='daily')
market_monthly_ftse= pd.read_excel ('index.xlsx', skiprows=2, parse_dates=['date'],index_col=0,sheet_name='monthly')
#market_daily_ftse = market_daily_ftse['FTSEMIB(PI)']
#market_monthly_ftse=market_monthly_ftse['FTSEMIB(PI)']
#print(market_daily_ftse.head())
#print(market_monthly_ftse.head())

# Extract the 'FTSEMIB(PI)' column
selected_fund_daily = market_daily_ftse['FTSEMIB(PI)']
selected_fund_monthly = market_monthly_ftse['FTSEMIB(PI)']

# Calculate log returns for daily and monthly data
log_returns_daily = np.log(selected_fund_daily / selected_fund_daily.shift(1))
log_returns_monthly = np.log(selected_fund_monthly / selected_fund_monthly.shift(1))

# Calculate mean, variance, standard deviation, skewness, and kurtosis
mean_value_daily = np.mean(log_returns_daily)
variance_value_daily = np.var(log_returns_daily)
std_dev_value_daily = np.std(log_returns_daily)
skewness_value_daily = (log_returns_daily).skew()
kurtosis_value_daily = (log_returns_daily).kurt()

mean_value_monthly = np.mean(log_returns_monthly)
variance_value_monthly = np.var(log_returns_monthly)
std_dev_value_monthly = np.std(log_returns_monthly)
skewness_value_monthly = (log_returns_monthly).skew()
kurtosis_value_monthly = (log_returns_monthly).kurt()

# Print the results for daily data
print("Daily Data:")
print(f"Mean: {mean_value_daily}")
print(f"Variance: {variance_value_daily}")
print(f"Standard Deviation: {std_dev_value_daily}")
print(f"Skewness: {skewness_value_daily}")
print(f"Kurtosis: {kurtosis_value_daily}")

# Print the results for monthly data
print("\nMonthly Data:")
print(f"Mean: {mean_value_monthly}")
print(f"Variance: {variance_value_monthly}")
print(f"Standard Deviation: {std_dev_value_monthly}")
print(f"Skewness: {skewness_value_monthly}")
print(f"Kurtosis: {kurtosis_value_monthly}")

# Create a DataFrame to store the results
results_df = pd.DataFrame({
'Statistic': ['Mean', 'Variance', 'Standard Deviation', 'Skewness', 'Kurtosis'],
'Daily Data': [mean_value_daily, variance_value_daily, std_dev_value_daily, skewness_value_daily, kurtosis_value_daily],
'Monthly Data': [mean_value_monthly, variance_value_monthly, std_dev_value_monthly, skewness_value_monthly, kurtosis_value_monthly]
})

# Convert the DataFrame to LaTeX format
latex_output = results_df.to_latex(index=False, column_format='lcc')



print("Table saved as FTSEMIB_statistics.tex")