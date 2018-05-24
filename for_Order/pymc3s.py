from pandas_datareader import data
import pymc3 as pm

returns = data.get_data_google('SPY', start='2008-5-1', end='2009-12-1')['Close'].pct_change()


with pm.Model() as sp500_model:
    nu = pm.Exponential('nu', 1./10, testval=5.)
    sigma = pm.Exponential('sigma', 1./.02, testval=.1)

    s = pm.GaussianRandomWalk('s', sigma**-2, shape=len(returns))
    volatility_process = pm.Deterministic('volatility_process', pm.math.exp(-2*s))

    r = pm.StudentT('r', nu, lam=volatility_process, observed=returns)
