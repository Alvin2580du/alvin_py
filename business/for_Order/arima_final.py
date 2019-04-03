import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import os
from xpinyin import Pinyin
from statsmodels.tsa.stattools import adfuller
import statsmodels

"""
构建arima模型，实现数据时间序列分析
https://blog.csdn.net/u010414589/article/details/49622625
这是一个参照，希望按照我的具体要求改一下。
1.	可以让我比较方便的输入数据运行，我刚开始用python，还不会调用excel的数据。或者你们写好代码教我怎么运行也可以。
2.	Arima模型进行差分后，需要对其进行adf检验，d的取值选择95%置信区间p值小于0.1的阶数
3.	在确定p q时，自动选择bic最小的那个
4.	预测部分，要求预测未来15年的数据
5.	最后进行逆差分，得到真正所需要的预测值

"""

P = Pinyin()
p1 = P.get_pinyin('辽宁', splitter='')

rcParams['figure.figsize'] = 15, 6

pro = []

save_path = './dataSets'


def build_one():
    # 数据按省份分组
    data = pd.read_csv("data.csv")
    for x, y in data.groupby(by='name'):
        print(type(x), x)
        if x == '陕西':
            x_pinyin = 'shanxii'
            pro.append(x_pinyin)
        elif x == '重庆':
            x_pinyin = 'chongqing'
            pro.append(x_pinyin)
        else:
            x_pinyin = P.get_pinyin(x, splitter='')
            pro.append(x_pinyin)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        y.to_csv("./{}/{}.csv".format(save_path, x_pinyin), index=None)


# 移动平均图
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()


# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=lags, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=lags, ax=ax2)
    plt.show()


def Stationaritytest(ts):
    ts = pd.Series(ts)
    ts.index = pd.Index(sm.tsa.datetools.dates_from_range('1996', '2015'))
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


ljhn = ['year', 'pgdp', 'income_farm', 'envrinvera', 'grspacera', 'treasewage', 'consum_fossra', 'consum_foss',
        'consum_coalra', 'carb_pener']  # 辽宁省、吉林省、黑龙江省、内蒙古自治区 ：

bths = ['year', 'carb_pener', 'consum_coalra', 'consum_foss', 'consum_fossra',
        'cosum_enr', 'popu']  # 北京市、天津市、河北省、山西省：

sjzafjs = ['year', 'gr_indsra', 'indsra', 'consum_fossra', 'consum_foss', 'carb_pener',
           'treasewage']  # 上海市、江苏省、浙江省、安徽省、福建省、江西省、山东省：

sgqnx = ['year', 'gr_popu', 'harmlesstrea', 'gr_agrira', 'consum_fossra', 'consum_foss', 'consum_coalra',
         'carb_pener']  # 陕西省、甘肃省、青海省、宁夏回族自治区、新疆维吾尔自治区

sgyc = ['year', 'popu_urba', 'employ', 'consum_foss', 'den_popu', 'ener_pcap',
        'harmlesstrea', 'envrinvera']  # 四川省、贵州省、云南省、重庆市

hhhggh = ['year', 'income_farm', 'pgdp', 'popu_urba', 'consum_foss',
          'consum_coalra']  # 河南省、湖北省、湖南省、广东省、广西壮族自治区、海南省：

columns_list = [ljhn, bths, sjzafjs, sgqnx, sgyc, hhhggh]

p = ['shanghai', 'yunnan', 'neimenggu', 'beijing', 'jilin', 'sichuan', 'tianjin', 'ningxia', 'anhui', 'shandong',
     'shanxi', 'guangdong', 'guangxi', 'xinjiang', 'jiangsu', 'jiangxi', 'hebei', 'henan', 'zhejiang', 'hainan',
     'hubei', 'hunan', 'gansu', 'fujian', 'guizhou', 'liaoning', 'zhongqing', 'shanxii', 'qinghai', 'heilongjiang']


def train2ts(train):
    dta = pd.Series(train)
    dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1996', '2015'))
    return dta


def plots_diff(train, province):
    dta = train2ts(train)
    dta.plot(figsize=(12, 8))
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    diff1 = dta.diff(1)
    diff1.plot(ax=ax1)

    ax1 = fig.add_subplot(212)
    diff2 = dta.diff(2)
    diff2.plot(ax=ax1)
    plt.savefig('./results/diff_1_{}.png'.format(province))


def plots(train, province):
    dta = train2ts(train)
    s1 = dta.diff(1)

    autocorr_plot1, ax1 = plt.subplots(figsize=(12, 10))
    ax1.set_xlabel('Lag')
    autocorr_plot1 = statsmodels.graphics.tsaplots.plot_acf(s1, ax=ax1, label='acf')
    autocorr_plot2 = statsmodels.graphics.tsaplots.plot_pacf(s1, ax=ax1, label='pacf')

    handles, labels = ax1.get_legend_handles_labels()
    handles = handles[:-len(handles) // 3][1::2]
    labels = labels[:-len(handles) // 3][1::2]
    ax1.legend(handles=handles, labels=labels, loc='best', shadow=True, numpoints=2)

    plt.savefig("./results/acf_pacf_{}.png".format(province))
    plt.close()


def get_bic_aic(train):
    dta = train2ts(train)
    arma_mod20 = sm.tsa.ARMA(dta, (7, 0)).fit()
    print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
    arma_mod30 = sm.tsa.ARMA(dta, (0, 1)).fit()
    print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)
    arma_mod40 = sm.tsa.ARMA(dta, (7, 1)).fit()
    print(arma_mod40.aic, arma_mod40.bic, arma_mod40.hqic)
    arma_mod50 = sm.tsa.ARMA(dta, (8, 0)).fit()
    print(arma_mod50.aic, arma_mod50.bic, arma_mod50.hqic)
    return arma_mod20


def predicts(arma_mod20, train):
    predict_sunspots = arma_mod20.predict('2016', '2030', dynamic=True)
    print(predict_sunspots)
    fig, ax = plt.subplots(figsize=(12, 8))

    dta = train2ts(train)
    ax = dta.ix['1996':].plot(ax=ax)
    predict_sunspots.plot(ax=ax)


def build_two():
    for file in os.listdir(save_path):
        file_path = os.path.join(save_path, file)
        province = file.replace(".csv", "")

        # 辽宁省、吉林省、黑龙江省、内蒙古自治区 ：
        if province in ['year', 'liaoning', 'jilin', 'heilongjiang', 'neimenggu']:
            data = pd.read_csv(file_path, usecols=ljhn, index_col='year')
            columns = data.columns.values
            for col in columns:
                train = data[col].values.tolist()
                dfoutput = Stationaritytest(train)
                dfoutput.to_csv("./results/dfoutput_{}.csv".format(province))
                plots(train, province)
                arma_mod20 = get_bic_aic(train)
                predicts(arma_mod20, train)

        # 北京市、天津市、河北省、山西省：
        if province in ['year', 'beijing', 'tianjin', 'hebei', 'shanxi']:
            data = pd.read_csv(file_path, usecols=bths, index_col='year')

            columns = data.columns.values
            for col in columns:
                train = data[col].values.tolist()

        # 上海市、江苏省、浙江省、安徽省、福建省、江西省、山东省：
        if province in ['year', 'shanghai', 'jiangsu', 'zhejiang', 'anhui', 'fujian', 'jiangxi', 'shandong']:
            data = pd.read_csv(file_path, usecols=sjzafjs, index_col='year')

            columns = data.columns.values
            for col in columns:
                train = data[col].values.tolist()

        # 陕西省、甘肃省、青海省、宁夏回族自治区、新疆维吾尔自治区
        if province in ['year', 'shanxii', 'gansu', 'qinghai', 'ningxia', 'xinjiang']:
            data = pd.read_csv(file_path, usecols=sgqnx, index_col='year')

            columns = data.columns.values
            for col in columns:
                train = data[col].values.tolist()

        # # 四川省、贵州省、云南省、重庆市
        if province in ['year', 'sichuan', 'guizhou', 'yunnan', 'chongqing']:
            data = pd.read_csv(file_path, usecols=sgyc, index_col='year')
            columns = data.columns.values
            for col in columns:
                train = data[col].values.tolist()

        # # 河南省、湖北省、湖南省、广东省、广西壮族自治区、海南省：
        if province in ['year', 'henan', 'hubei', 'hunan', 'guangdong', 'guangxi', 'hainan']:
            data = pd.read_csv(file_path, usecols=hhhggh, index_col='year')
            columns = data.columns.values
            for col in columns:
                train = data[col].values.tolist()


if __name__ == "__main__":
    method = 'build_two'

    if method == 'build_one':
        build_one()

    if method == 'build_two':
        build_two()

    if method == 'test':
        import numpy as np
        import statsmodels.tsa.stattools, statsmodels.graphics.tsaplots
        import matplotlib.pyplot as plt


        def plots(s1, s2):
            autocorr_plot1, ax1 = plt.subplots(figsize=(12, 10))
            ax1.set_xlabel('Lag')

            autocorr_plot1 = statsmodels.graphics.tsaplots.plot_acf(s1, ax=ax1, label='Asset A')
            autocorr_plot2 = statsmodels.graphics.tsaplots.plot_acf(s2, ax=ax1, label='Asset B')

            handles, labels = ax1.get_legend_handles_labels()
            handles = handles[:-len(handles) // 3][1::2]
            labels = labels[:-len(handles) // 3][1::2]
            ax1.legend(handles=handles, labels=labels, loc='best', shadow=True, numpoints=2)

            plt.show()
