import pandas as pd
import statsmodels.api as sm


data = sm.datasets.macrodata.load()

data = data.data[['year', 'quarter', 'realgdp', 'cpi']]
data = pd.DataFrame(data)
print(data.head())

data = sm.tsa.add_lag(data, 1, lags=2)
df = pd.DataFrame(data)
print(df.head())
