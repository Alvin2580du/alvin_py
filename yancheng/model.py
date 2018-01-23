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
# import bt
from scipy.fftpack import fft, ifft, ifftn, dct, idct, dst, idst
from scipy.fftpack import fftfreq, fftshift, rfft, irfft

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
    predict_dates = pd.date_range('2017-10-28', periods=276, freq='D')

    df = pd.DataFrame(data=cnt, index=dates, dtype=np.float32)
    df.columns = ['cnt']
    df.to_csv("yancheng/datasets/results/cnt.csv")
    df.index = pd.DatetimeIndex(df.index)
    dta = df['cnt']
    arma_mod12 = sm.tsa.ARMA(dta, (12, 0)).fit()
    # 模型预测
    predict_sunspots = arma_mod12.predict(start='28/10/2017', end='30/7/2018', dynamic=True).values
    predicts_data = pd.read_csv("yancheng/datasets/test_A_20171225.txt", sep='\t')
    predicts_data['cnt'] = predict_sunspots

    predicts_data.to_csv("yancheng/datasets/results/arma_predicts.csv", index=None)


def arch_model():
    train = pd.read_csv("yancheng/datasets/results/total_by_day.csv", usecols=['cnt']).values.tolist()
    X = [j for i in train for j in i]
    train_x = []
    length = len(X)
    for i in range(1, length):
        cha = (X[i] - X[i - 1]) / X[i - 1]
        train_x.append(cha)

    am = arch.arch_model(train_x, x=None, mean='ARX', lags=0, vol='Garch', p=1, o=0, q=1,
                         power=2.0, dist='Normal', hold_back=1)
    res = am.fit()
    summ = res.summary()
    print(summ)
    parms = res.params
    print(parms)
    res.hedgehog_plot()


def ar():
    data = pd.read_csv("yancheng/datasets/results/total_by_day.csv")
    cnt = data['cnt'].values
    length = len(data)
    dates = pd.date_range('1/1/2015', periods=length, freq='D')
    df = pd.DataFrame(data=cnt, index=dates, dtype=np.float32)
    df.columns = ['cnt']
    df.index = pd.DatetimeIndex(df.index)
    dta = df['cnt']
    mod_ar2 = sm.tsa.SARIMAX(dta, order=(2, 0, 0))
    res_ar2 = mod_ar2.fit()
    # print(res_ar2.summary())
    mod_sarimax = sm.tsa.SARIMAX(dta, order=(1, 1, 1), seasonal_order=(0, 1, 1, 4))
    res_sarimax = mod_sarimax.fit()
    print(res_sarimax.summary())


def fft_study():
    x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
    y = fft(x)
    print(y[0].real)
    yinv = ifft(y)
    print(yinv)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        exit(1)

    method = sys.argv[1]
    if method == 'train':
        linear_model()
    if method == 'test':
        lineartest()

    if method == 'arma':
        arma()

    if method == 'arch':
        arch_model()

    if method == 'ar':
        ar()

    if method == 'fft':
        fft_study()
