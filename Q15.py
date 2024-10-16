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


# Creazione di un DataFrame vuoto con gli indici e le colonne desiderate
dfall = pd.DataFrame(index=['MV', 'BL', 'BY', 'GMV'],
                  columns=['Mean', 'Standard Deviation', 'Variance', 'Skewness', 'Kurtosis', 'Sharpe Ratio'])

# Inserimento manuale dei dati nella riga 'MV'
dfall.loc['MV'] = [0.196912,
                0.059593,
                0.003551,
                3.156180,
                9.971121,
                0.682597]

# Inserimento manuale dei dati nelle altre righe, se necessario
dfall.loc['BL'] = [0.051433, 0.076608, 0.005869, 1.812469, 4.264149, 0.279775]
dfall.loc['BY'] = [0.236638, 0.283611, 0.080435, 1.560581, 3.500346, 0.763858]
dfall.loc['GMV'] = [-0.008113, 0.140247, 0.019669, 0.846857, 0.548629, -0.200455]




# Define the weights for each portfolio
weights = {
    'MV': 0.25,
    'BL': 0.25,
    'BY': 0.25,
    'GMV': 0.25
}

# Create a new DataFrame to store the resulting portfolio statistics
resulting_portfolio_equallyweigted = pd.DataFrame(columns=dfall.columns)

# Calculate the resulting portfolio statistics
for stat in dfall.columns:
    resulting_portfolio_equallyweigted.loc['Resulting Portfolio', stat] = sum(weights[port] * dfall.loc[port, stat] for port in weights)

# Display the resulting portfolio statistics
    resulting_portfolio_equallyweigteddf = pd.DataFrame(resulting_portfolio_equallyweigted)

    with open('equallywportfolio.tex', 'w') as f:
     f.write(resulting_portfolio_equallyweigteddf.to_latex())



print(resulting_portfolio_equallyweigteddf)


# Define the weights for each portfolio
weights = {
    'MV': 0.71,
    'BL': 0.13,
    'BY': 0.15,
    'GMV': 0.01
}

# Create a new DataFrame to store the resulting portfolio statistics
resulting_portfolio_ponderated1 = pd.DataFrame(columns=dfall.columns)

# Calculate the resulting portfolio statistics
for stat in dfall.columns:
    resulting_portfolio_ponderated1.loc['Resulting Portfolio', stat] = sum(weights[port] * dfall.loc[port, stat] for port in weights)

# Display the resulting portfolio statistics
    resulting_portfolio_ponderated1df = pd.DataFrame(resulting_portfolio_ponderated1)

   



print(resulting_portfolio_ponderated1)

# Parametri dell'investimento
import matplotlib.pyplot as plt
import numpy as np

# Parametri dell'investimento
P = 100000  # Capitale iniziale
r = 0.18  # Ritorno atteso annuo
sigma = 0.09  # Volatilità
t_max = 10  # Numero massimo di anni

# Calcolo del tasso di ritorno effettivo
r_eff = (r**2 + sigma**2)**0.5

# Periodi di investimento in anni
t_values = np.arange(1, t_max + 1)

# Calcolo del montante finale per ciascun periodo
A_values = P * (1 + r_eff)**t_values

# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.plot(t_values, A_values, marker='o', linestyle='-', color='green')
plt.title('Total principal and interest over time')
plt.xlabel('Yearly Periods')
plt.ylabel('Principal + Interest')
plt.grid(True)
plt.show()