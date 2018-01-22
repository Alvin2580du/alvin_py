from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.externals import joblib
import random
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy import stats
import statsmodels.api as sm  # 统计相关的库
import matplotlib.pyplot as plt
import arch  # 条件异方差模型相关的库

from pyduyp.logger.log import log
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from pyduyp.utils.utils import list_reverse_pop

scalar = StandardScaler()


def linear_model():
    data = pd.read_csv("yancheng/datasets/results/train/total_by_day_ex_sorted.csv")
    x, y = data['week'].values, data['cnt'].values
    train_x = np.array(x[: int(len(x) * 0.8)]).reshape(782, 1)
    train_y = np.array(y[: int(len(x) * 0.8)]).reshape(782, 1)
    test_x = np.array(x[int(len(x) * 0.8):]).reshape(196, 1)
    test_y = np.array(x[int(len(x) * 0.8):]).reshape(196, )

    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x).reshape(782, 1)

    scaler = preprocessing.StandardScaler().fit(train_y)
    train_y = scaler.transform(train_y).reshape(782, )

    lm = LinearRegression()

    lm.fit(train_x, train_y)
    joblib.dump(lm, "yancheng/datasets/results/linear_model.m")
    score = lm.score(test_x, test_y)
    mse = mean_squared_error(test_y, lm.predict(test_x))
    # log.info("{}".format(score, mse))

    log.info("{}, {}".format(score, mse))


def lineartest():
    lm = joblib.load("yancheng/datasets/results/linear_model.m")
    data = pd.read_csv("yancheng/datasets/test_A_20171225.txt", sep="\t")
    test_x = data['day_of_week'].values.reshape(len(data), 1)
    mean_y, std_y = np.mean(test_x), np.std(test_x)

    test_x_scaled = preprocessing.scale(test_x)
    p = lm.predict(test_x_scaled)
    print(p)
    # prob = lm.predict_proba(test_x_scaled)
    # log.info("{}".format(prob))
    p_real = p * std_y + mean_y
    print(p_real)


def inear_model_t():
    lm = joblib.load("yancheng/datasets/results/linear_model.m")
    data = pd.read_csv("yancheng/datasets/test_A_20171225.txt", sep="\t")
    test_x = data['day_of_week'].values.reshape(len(data), 1)

    test_x_scaled = preprocessing.scale(test_x)
    mean_y, std_y = np.mean(test_x), np.std(test_x)
    p = lm.predict(X=test_x)
    print(p)
    p_real = p * std_y + mean_y
    print(p_real)


def GP():
    data = pd.read_csv("./datasets/results/data_train.csv").values
    kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    train_x, train_y = data[:, :-1], data[:, -1]
    log.info("{}  {}".format(train_x.shape, train_y.shape))
    reg.fit(train_x, train_y)
    test_data = pd.read_csv("./datasets/test_A_20171225.txt", sep="\t")
    # TODO 新数据的不断迭代训练模型，预测新结果
    resuluts = []
    for x in tqdm(test_data.values):
        predict_x = np.array([[x[0], x[1]]])
        p = reg.predict(predict_x)[0]
        resuluts.append(int(p))
        train_x_update = list_reverse_pop(train_x, x)
        new_train_x = np.array(train_x_update)
        train_y_update = list_reverse_pop(train_y, p)
        new_train_y = np.array(train_y_update)
        log.info("{}, {}".format(new_train_x.shape, new_train_y.shape))
        reg.fit(new_train_x, new_train_y)
        log.info("predict: {}".format(resuluts))

    test_data_copy = test_data.copy()
    test_data_copy['predict'] = resuluts
    test_data_copy.to_csv("./datasets/results/predict_GP.csv", index=None)

    log.info('{}'.format(resuluts))


t_cache = []


def GP_test():
    reg = joblib.load("./datasets/results/GP.m")
    test_data = pd.read_csv("./datasets/test_A_20171225.txt", sep="\t")
    out = test_data.copy()
    predicts = []
    test_data_all = test_data.values

    test = random.choice(test_data_all)
    if test not in t_cache:
        t_cache.append(test)

        data_predict = np.array([[test[0], test[1]]])
        p = reg.predict(data_predict)[0]
        predicts.append(int(p))

    out['predict'] = predicts
    out.to_csv("./datasets/results/predicts_A.csv", index=None)


def plot_acf():
    data = pd.read_csv("yancheng/datasets/results/total_by_day.csv", usecols=['cnt']).values
    data2series = [j for i in data.tolist() for j in i]
    data2shift = pd.Series(data2series)
    t = sm.tsa.stattools.adfuller(data2shift)
    print("p-value: {:.07f}  ".format(t[1]))
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_pacf(data2shift, lags=20, ax=ax1)
    fig.savefig("yancheng/datasets/acf.png")
    order = (12, 0)

    # m = 25  # 我们检验25个自相关系数
    # acf, q, p = sm.tsa.acf(at2, nlags=m, qstat=True)  ## 计算自相关系数 及p-value
    # out = np.c_[range(1, 26), acf[1:], q, p]
    # output = pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
    # output = output.set_index('lag')
    # print(output)


def arma():
    data = pd.read_csv("yancheng/datasets/results/total_by_day.csv")
    cnt = data['cnt'].values
    length = len(data)
    dates = pd.date_range('1/1/2015', periods=length, freq='D')
    df = pd.DataFrame(data=cnt, index=dates, dtype=np.float32)
    df.columns = ['cnt']
    df.to_csv("yancheng/datasets/results/cnt.csv")
    df.index = pd.DatetimeIndex(df.index)
    dta = df['cnt']
    arma_mod20 = sm.tsa.ARMA(dta, (7, 0)).fit()
    print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
    print("= "*20)
    arma_mod30 = sm.tsa.ARMA(dta, (8, 1)).fit()
    print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)
    print("= "*20)

    arma_mod40 = sm.tsa.ARMA(dta, (10, 1)).fit()
    print(arma_mod40.aic, arma_mod40.bic, arma_mod40.hqic)
    print("= "*20)

    arma_mod50 = sm.tsa.ARMA(dta, (12, 0)).fit()
    print(arma_mod50.aic, arma_mod50.bic, arma_mod50.hqic)
    exit(1)
    # aic，bic，hqic均最小，因此是最佳模型。
    # at = data2shift - arma_mod20.fittedvalues
    # at2 = np.square(at)
    # plt.figure(figsize=(10, 6))
    # plt.subplot(211)
    # plt.plot(at, label='at')
    # plt.legend()
    # plt.subplot(212)
    # plt.plot(at2, label='at^2')
    # plt.legend(loc=0)
    # plt.savefig("yancheng/datasets/arma.png")
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(arma_mod50.resid.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(arma_mod50.resid, lags=40, ax=ax2)
    # 模型预测
    predict_sunspots = arma_mod20.predict('2090', '2100', dynamic=True)
    print(predict_sunspots)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = dta.ix['2001':].plot(ax=ax)
    predict_sunspots.plot(ax=ax)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        exit(1)

    method = sys.argv[1]
    if method == 'train':
        linear_model()
    if method == 'test':
        lineartest()

    if method == 'state':
        arma()
