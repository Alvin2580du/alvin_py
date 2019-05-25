import pandas as pd
import arch
import matplotlib.pyplot as plt
import statsmodels.api as sm


def garch_build():
    data = pd.read_excel("Monthly silver Spot.xlsx")
    del data['Log(XAG/USD)']
    train = data.set_index(keys='Date')

    """
    vol : str, optional
        Name of the volatility model.  Currently supported options are:
        'GARCH' (default), 'ARCH', 'EGARCH', 'FIARCH' and 'HARCH'
    """
    model = arch.arch_model(train, vol='GARCH')
    res = model.fit()
    summ = res.summary()
    print(summ)
    res.hedgehog_plot(type='mean')
    plt.savefig("hedgehog_plot.png")


def arma_build():
    data = pd.read_excel("Monthly silver Spot.xlsx")
    del data['Log(XAG/USD)']
    train = data.set_index(keys='Date')
    train.plot(figsize=(12, 8))
    plt.savefig("train.png")
    plt.close()

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(train, lags=31, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(train, lags=31, ax=ax2)
    """
    
    order : iterable
        (p,q,k) - AR lags, MA lags, and number of exogenous variables
        including the constant.
    """
    arma_mod20 = sm.tsa.ARMA(train, order=(2, 0)).fit(disp=False)
    plt.savefig("plot_acf.png")
    plt.close()
    print("模型参数")
    print(arma_mod20.params)
    print('- * -' * 10)
    print("aic:{}, bic:{}".format(arma_mod20.aic, arma_mod20.bic))
    print()

    start = '2015/1/31'
    res = arma_mod20.predict(start, '2018/3/31', dynamic=True)
    pd.DataFrame(res).to_excel("arma_mod20.predict_start_{}.xlsx".format(start.replace("/", '-')), index=None)
    print('建模完成')


garch_build()
print('- * -' * 10)

arma_build()
