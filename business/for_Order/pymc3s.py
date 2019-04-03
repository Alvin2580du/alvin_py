from pandas_datareader import data
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

returns = data.get_data_google('SPY', start='2008-5-1', end='2009-12-1')['Close'].pct_change()
print(returns)

with pm.Model() as sp500_model:
    nu = pm.Exponential('nu', 1./10, testval=5.)
    sigma = pm.Exponential('sigma', 1./.02, testval=.1)
    s = pm.GaussianRandomWalk('s', sigma**-2, shape=len(returns))
    volatility_process = pm.Deterministic('volatility_process', pm.math.exp(-2*s))
    r = pm.StudentT('r', nu, lam=volatility_process, observed=returns)


with sp500_model:
    trace = pm.sample(2000)
pm.traceplot(trace, [nu, sigma])

fig, ax = plt.subplots(figsize=(15, 8))
returns.plot(ax=ax)
ax.plot(returns.index, 1/np.exp(trace['s', ::5].T), 'r', alpha=.03)
ax.set(title='volatility_process', xlabel='time', ylabel='volatility')
ax.legend(['S&P500', 'stochastic volatility process'])
plt.show()