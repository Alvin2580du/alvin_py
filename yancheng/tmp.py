def fun1():
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.graphics.api import qqplot
    import pandas as pd
    import os

    root = './datasets/results'

    da = pd.read_csv("./datasets/results/data_train.csv", usecols=['cnt']).values.tolist()

    # dta = [10930, 10318, 10595, 10972, 7706, 6756, 9092, 10551, 9722, 10913, 11151, 8186, 6422,
    #        6337, 11649, 11652, 10310, 12043, 7937, 6476, 9662, 9570, 9981, 9331, 9449, 6773, 6304, 9355,
    #        10477, 10148, 10395, 11261, 8713, 7299, 10424, 10795, 11069, 11602, 11427, 9095, 7707, 10767,
    #        12136, 12812, 12006, 12528, 10329, 7818, 11719, 11683, 12603, 11495, 13670, 11337, 10232,
    #        13261, 13230, 15535, 16837, 19598, 14823, 11622, 19391, 18177, 19994, 14723, 15694, 13248,
    #        9543, 12872, 13101, 15053, 12619, 13749, 10228, 9725, 14729, 12518, 14564, 15085, 14722,
    #        11999, 9390, 13481, 14795, 15845, 15271, 14686, 11054, 10395]
    #
    data_list = []
    for x in da:
        data_list.append(x[0])

    data = pd.Series(data_list)
    data.index = pd.Index(range(1, 1033))
    data.plot(figsize=(12, 8))
    plt.savefig(os.path.join(root, "1.png"))

    # 一阶差分
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    diff1 = data.diff(1)
    diff1.plot(ax=ax1)
    plt.savefig(os.path.join(root, "2.png"))

    # 二阶差分
    fig = plt.figure(figsize=(12, 8))
    ax2 = fig.add_subplot(111)
    diff2 = data.diff(2)
    diff2.plot(ax=ax2)
    plt.savefig(os.path.join(root, "3.png"))

    dta = data.diff(1)  # 我们已经知道要使用一阶差分的时间序列，之前判断差分的程序可以注释掉
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(dta, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)
    plt.savefig(os.path.join(root, "3.png"))

    arma_mod20 = sm.tsa.ARMA(dta, (1, 0)).fit()
    print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)


def fun2():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from mpl_toolkits.mplot3d import Axes3D
    import pandas as pd

    test = np.array([[1032, 4]])
    # data = np.array([
    #     [2001, 100.83, 410], [2005, 90.9, 500], [2007, 130.03, 550], [2004, 78.88, 410], [2006, 74.22, 460],
    #     [2005, 90.4, 497], [1983, 64.59, 370], [2000, 164.06, 610], [2003, 147.5, 560], [2003, 58.51, 408],
    #     [1999, 95.11, 565], [2000, 85.57, 430], [1995, 66.44, 378], [2003, 94.27, 498], [2007, 125.1, 760],
    #     [2006, 111.2, 730], [2008, 88.99, 430], [2005, 92.13, 506], [2008, 101.35, 405], [2000, 158.9, 615]])
    data = pd.read_csv("./datasets/results/data_train.csv").values
    print(data.shape)

    kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    train_x, train_y = data[:, :-1], data[:, -1]

    reg.fit(train_x, train_y)
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xset, yset = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
    output, err = reg.predict(np.c_[xset.ravel(), yset.ravel()], return_std=True)
    output, err = output.reshape(xset.shape), err.reshape(xset.shape)
    sigma = np.sum(reg.predict(data[:, :-1], return_std=True)[1])
    up, down = output * (1 + 1.96 * err), output * (1 - 1.96 * err)

    fig = plt.figure(figsize=(10.5, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_wireframe(xset, yset, output, rstride=10, cstride=2, antialiased=True)
    surf_u = ax1.plot_wireframe(xset, yset, up, colors='lightgreen', linewidths=1,
                                rstride=10, cstride=2, antialiased=True)
    surf_d = ax1.plot_wireframe(xset, yset, down, colors='lightgreen', linewidths=1,
                                rstride=10, cstride=2, antialiased=True)
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')
    ax1.set_title('House Price at (2004, 98.31): {0:.2f}$*10^4$ RMB'.format(reg.predict(test)[0]))
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Area, $m^2$')
    ax1.set_zlabel('Price,$10^4$ RMB')

    ax = fig.add_subplot(122)
    s = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=plt.cm.viridis)
    # ax.contour(xset,yset,output)
    im = ax.imshow(output, interpolation='bilinear', origin='lower',
                   extent=(x_min, x_max - 1, y_min, y_max), aspect='auto')
    plt.colorbar(s, ax=ax)
    ax.set_title('House Price,$10^4$ RMB')
    ax.hlines(test[0, 1], x_min, x_max - 1)
    ax.vlines(test[0, 0], y_min, y_max)
    ax.text(test[0, 0], test[0, 1], '{0:.2f}$*10^4$ RMB'.format(reg.predict(test)[0]), ha='left', va='bottom',
            color='k', size=11, rotation=90)
    ax.set_xlabel('Year')
    ax.set_ylabel('Area, $m^2$')
    plt.subplots_adjust(left=0.05, top=0.95, right=0.95)
    plt.show()


fun2()