import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import statsmodels.api as sm
import os
from xpinyin import Pinyin
from statsmodels.tsa.stattools import adfuller

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

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists("./results"):
    os.makedirs("./results")


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


def Stationaritytest(ts):
    ts = pd.Series(ts)
    ts.index = pd.Index(sm.tsa.datetools.dates_from_range('1996', '2015'))
    dftest = adfuller(ts)
    print("dftest:{}".format(dftest))
    print("- " * 20)
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


def plots_diff(train, province, col):
    dta = train2ts(train)
    dta.plot(figsize=(12, 8))
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    diff1 = dta.diff(1)
    diff1.plot(ax=ax1)

    ax1 = fig.add_subplot(212)
    diff2 = dta.diff(2)
    diff2.plot(ax=ax1)
    plt.savefig('./results/{}_{}_diff1.png'.format(province, col))


def plots_acf(train, province, col, lags=8):
    dta = train2ts(train)
    s1 = dta.diff(1)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(s1, ax=ax1, lags=lags)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(s1, ax=ax2, lags=lags)
    plt.savefig("./results/{}_{}_acf_pacf.png".format(province, col))
    print("./results/{}_{}_acf_pacf.png".format(province, col))
    plt.close()


def get_bic_aic(train):
    p, q, n = 0, 7, 8
    dta = train2ts(train)
    arma_mod20 = sm.tsa.ARIMA(dta, (7, 0)).fit()
    print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
    bic_1 = arma_mod20.bic
    arma_mod30 = sm.tsa.ARMA(dta, (0, 1)).fit()
    print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)
    bic_2 = arma_mod30.bic
    arma_mod40 = sm.tsa.ARMA(dta, (7, 1)).fit()
    print(arma_mod40.aic, arma_mod40.bic, arma_mod40.hqic)
    bic_3 = arma_mod40.bic
    arma_mod50 = sm.tsa.ARMA(dta, (8, 0)).fit()
    print(arma_mod50.aic, arma_mod50.bic, arma_mod50.hqic)
    bic_4 = arma_mod50.bic
    # 这里根据最小的bic， 选择要返回的模型
    rows = {bic_1: arma_mod20, bic_2: arma_mod30, bic_3: arma_mod40, bic_4: arma_mod50}

    return rows[min(list(rows.keys()))]


def predicts(arma_mod20, train, province, col):
    predict_sunspots = arma_mod20.predict('2016', '2030', dynamic=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    dta = train2ts(train)
    ax = dta.ix['1996':].plot(ax=ax)
    predict_sunspots.plot(ax=ax)
    df = pd.Series(predict_sunspots)
    df.to_csv("./results/{}_{}_predicts.csv".format(province, col))


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
                plots_diff(train, province, col)
                dfoutput = Stationaritytest(train)
                dfoutput.to_csv("./results/{}_{}_dfoutput.csv".format(province, col))
                plots_acf(train, province, col)
                arma_mod20 = get_bic_aic(train)
                predicts(arma_mod20, train, province, col)

        # 北京市、天津市、河北省、山西省：
        if province in ['year', 'beijing', 'tianjin', 'hebei', 'shanxi']:
            data = pd.read_csv(file_path, usecols=bths, index_col='year')
            columns = data.columns.values
            for col in columns:
                train = data[col].values.tolist()
                plots_diff(train, province, col)
                dfoutput = Stationaritytest(train)
                dfoutput.to_csv("./results/{}_{}_dfoutput.csv".format(province, col))
                plots_acf(train, province, col)
                # arma_mod20 = get_bic_aic(train)
                # predicts(arma_mod20, train, province, col)

        # 上海市、江苏省、浙江省、安徽省、福建省、江西省、山东省：
        if province in ['year', 'shanghai', 'jiangsu', 'zhejiang', 'anhui', 'fujian', 'jiangxi', 'shandong']:
            data = pd.read_csv(file_path, usecols=sjzafjs, index_col='year')

            columns = data.columns.values
            for col in columns:
                train = data[col].values.tolist()
                plots_diff(train, province, col)

                dfoutput = Stationaritytest(train)
                dfoutput.to_csv("./results/{}_{}_dfoutput.csv".format(province, col))
                plots_acf(train, province, col)
                # arma_mod20 = get_bic_aic(train)
                # predicts(arma_mod20, train, province, col)

        # 陕西省、甘肃省、青海省、宁夏回族自治区、新疆维吾尔自治区
        if province in ['year', 'shanxii', 'gansu', 'qinghai', 'ningxia', 'xinjiang']:
            data = pd.read_csv(file_path, usecols=sgqnx, index_col='year')

            columns = data.columns.values
            for col in columns:
                train = data[col].values.tolist()
                plots_diff(train, province, col)

                dfoutput = Stationaritytest(train)
                dfoutput.to_csv("./results/{}_{}_dfoutput.csv".format(province, col))
                plots_acf(train, province, col)
                # arma_mod20 = get_bic_aic(train)
                # predicts(arma_mod20, train, province, col)

        # # 四川省、贵州省、云南省、重庆市
        if province in ['year', 'sichuan', 'guizhou', 'yunnan', 'chongqing']:
            data = pd.read_csv(file_path, usecols=sgyc, index_col='year')
            columns = data.columns.values
            for col in columns:
                train = data[col].values.tolist()
                plots_diff(train, province, col)

                dfoutput = Stationaritytest(train)
                dfoutput.to_csv("./results/{}_{}_dfoutput.csv".format(province, col))
                plots_acf(train, province, col)
                # arma_mod20 = get_bic_aic(train)
                # predicts(arma_mod20, train, province, col)

        # # 河南省、湖北省、湖南省、广东省、广西壮族自治区、海南省：
        if province in ['year', 'henan', 'hubei', 'hunan', 'guangdong', 'guangxi', 'hainan']:
            data = pd.read_csv(file_path, usecols=hhhggh, index_col='year')
            columns = data.columns.values
            for col in columns:
                train = data[col].values.tolist()
                plots_diff(train, province, col)

                dfoutput = Stationaritytest(train)
                dfoutput.to_csv("./results/{}_{}_dfoutput.csv".format(province, col))
                plots_acf(train, province, col)
                # arma_mod20 = get_bic_aic(train)
                # predicts(arma_mod20, train, province, col)


if __name__ == "__main__":
    method = 'build_two'

    if method == 'build_one':
        build_one()

    if method == 'build_two':
        build_two()
