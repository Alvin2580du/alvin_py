import numpy as np
import pandas as pd
from collections import OrderedDict
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def str2float(inputs):
    return inputs.replace(" ", "").replace(" ", "").replace("\xa0", "").replace("\ufeff", "").replace("\n", "").replace(
        '?', "")


# 一阶差分

def make_train_datasets_big():
    save = []
    with open('./datasets/gdp/gdp_newTrain.csv', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            if "年份" in line:
                continue
            data = str2float(line).split("\t")
            rows = OrderedDict()
            rows['年份'] = data[0]
            rows['上海市生产总值'] = data[1]
            rows['住户存款'] = data[2]
            rows['个人存款'] = data[3]
            rows['投资总额'] = data[4]
            rows['一般公共预算收入'] = data[5]
            rows['税收收入'] = data[6]
            rows['非税收收入'] = data[7]
            rows['一般公共预算支出'] = data[8]
            rows['关口出口'] = data[9]
            rows['农业总产值'] = data[10]
            rows['工业总产值'] = data[11]
            rows['建筑业年末从业人员'] = data[12]
            rows['建筑业总产值'] = data[13]
            rows['批发和零售业、住宿和餐饮业从业人员'] = data[14]
            rows['社会消费品零售总额'] = data[15]
            rows['职工工资总额'] = data[16]
            rows['常住人口'] = data[17]
            rows['年末户籍人口'] = data[18]
            save.append(rows)

    df = pd.DataFrame(save, dtype=np.float32)
    if False:
        df['人口流动'] = df.apply(lambda row: subs(row['常住人口'], row['年末户籍人口']), axis=1)
        del df['常住人口']
        del df['年末户籍人口']
        df.to_csv("./datasets/gdp/gdp051311.csv", index=None)
    train = pd.DataFrame()
    train['年份'] = df['年份']
    train['上海市生产总值'] = df['上海市生产总值'].pct_change()
    train['住户存款'] = df['住户存款'].pct_change()
    train['个人存款'] = df['个人存款'].pct_change()
    train['投资总额'] = df['投资总额'].pct_change()
    train['一般公共预算收入'] = df['一般公共预算收入'].pct_change()
    train['税收收入'] = df['税收收入'].pct_change()
    train['非税收收入'] = df['非税收收入'].pct_change()
    train['一般公共预算支出'] = df['一般公共预算支出'].pct_change()
    train['关口出口'] = df['关口出口'].pct_change()
    train['农业总产值'] = df['农业总产值'].pct_change()
    train['工业总产值'] = df['工业总产值'].pct_change()
    train['建筑业年末从业人员'] = df['建筑业年末从业人员'].pct_change()
    train['建筑业总产值'] = df['建筑业总产值'].pct_change()
    train['批发和零售业、住宿和餐饮业从业人员'] = df['批发和零售业、住宿和餐饮业从业人员'].pct_change()
    train['社会消费品零售总额'] = df['社会消费品零售总额'].pct_change()
    train['职工工资总额'] = df['职工工资总额'].pct_change()
    train['人口流动'] = df.apply(lambda row: subs(row['常住人口'], row['年末户籍人口']), axis=1)
    train = train.dropna(axis=0)
    train.to_csv("./datasets/gdp/rateTrain.csv", index=None)


def subs(x1, x2):
    return abs(x1 - x2)


def make_train_datasets():
    # 上海市生产总值	住户存款	投资总额	一般公共预算收入	关口出口	农业总产值	工业总产值
    # 建筑业总产值（亿元）	社会消费品零售总额	迁入人口	职工工资总额 

    save = []
    with open('./datasets/gdp/gdp_newTrain.csv', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            if "年份" in line:
                continue
            data = str2float(line).split("\t")
            print(len(data), data)
            rows = OrderedDict()
            rows['年份'] = data[0]
            rows['上海市生产总值'] = data[1]
            rows['住户存款'] = data[2]
            rows['投资总额'] = data[3]
            rows['一般公共预算收入'] = data[4]
            rows['关口出口'] = data[5]
            rows['农业总产值'] = data[6]
            rows['工业总产值'] = data[7]
            rows['建筑业总产值'] = data[8]
            rows['社会消费品零售总额'] = data[9]
            rows['职工工资总额'] = data[10]
            rows['常住人口'] = data[11]
            rows['年末户籍人口'] = data[12]
            save.append(rows)

    df = pd.DataFrame(save, dtype=np.float32)
    train = pd.DataFrame()
    train['年份'] = df['年份']
    train['上海市生产总值'] = df['上海市生产总值'].pct_change()
    train['住户存款'] = df['住户存款'].pct_change()
    train['投资总额'] = df['投资总额'].pct_change()
    train['一般公共预算收入'] = df['一般公共预算收入'].pct_change()
    train['关口出口'] = df['关口出口'].pct_change()
    train['农业总产值'] = df['农业总产值'].pct_change()
    train['工业总产值'] = df['工业总产值'].pct_change()
    train['建筑业总产值'] = df['建筑业总产值'].pct_change()
    train['社会消费品零售总额'] = df['社会消费品零售总额'].pct_change()
    train['职工工资总额'] = df['职工工资总额'].pct_change()
    train['人口流动'] = df.apply(lambda row: subs(row['常住人口'], row['年末户籍人口']), axis=1)
    train = train.dropna(axis=0)
    train.to_csv("./datasets/gdp/rateTrain_2.csv", index=None)


def gdp_regresion():
    data = pd.read_csv("./datasets/gdp/gdp051311.csv")
    del data['年份']
    # del data['一般公共预算收入']
    # del data['职工工资总额']
    # del data['建筑业总产值']
    # del data['农业总产值']
    # del data['社会消费品零售总额']
    # del data['住户存款']
    # del data['投资总额']
    # del data['关口出口']
    # del data['工业总产值']
    # del data['一般公共预算支出']
    y = data['上海市生产总值']
    del data['上海市生产总值']
    x = data
    k = 0
    # VIF
    vif = pd.DataFrame()
    vif['vif'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif['features'] = x.columns
    vif = vif.sort_values(by='vif', ascending=False)
    vif.to_csv("./datasets/gdp/gdp_vif.csv", index=None)

    X = sm.add_constant(x)
    est = sm.OLS(y, X).fit()
    summ = est.summary().as_csv()
    f_summ = open('./datasets/gdp/gdp_results_{}.csv'.format(k), 'w', encoding='utf-8')
    print(summ, end='\n', sep='', file=f_summ)
    params = est.params
    params.to_csv("./datasets/gdp/params_gdp_{}.csv".format(k))
    bse = est.bse
    bse.to_csv("./datasets/gdp/bse_gdp_{}.csv".format(k))


def make_renkou_train_datasets():
    # 常住人口,人口密度,总户数,平均每户人口,年末户籍人口,男性,女性,出生人数,死亡人数,迁入人口,迁出人口,准予登记结婚,初婚,再婚,离婚件数
    save = []
    with open('./datasets/gdp/renkou.csv', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            if '年份' in line:
                continue
            data = str2float(line).split(",")
            rows = OrderedDict()
            rows['年份'] = data[0]
            rows['常住人口'] = data[1]
            rows['人口密度'] = data[2]
            rows['总户数'] = data[3]
            rows['平均每户人口'] = data[4]
            rows['年末户籍人口'] = data[5]
            rows['男性'] = data[6]
            rows['女性'] = data[7]
            rows['出生人数'] = data[8]
            rows['死亡人数'] = data[9]
            rows['迁入人口'] = data[10]
            rows['迁出人口'] = data[11]
            rows['准予登记结婚'] = data[12]
            rows['初婚'] = data[13]
            rows['再婚'] = data[14]
            rows['离婚件数'] = data[15]
            save.append(rows)

    df = pd.DataFrame(save, dtype=np.float32)
    df.to_csv("./datasets/gdp/renkou_rateTrain.csv", index=None)


def regresion_renkouo():
    data = pd.read_csv("./datasets/gdp/renkou_rateTrain.csv")
    data_corr = data.corr()
    data_corr.to_csv("./datasets/gdp/corr_renkou.csv", index=None)
    del data['年份']
    y = data['迁入人口']
    del data['迁入人口']
    x = data
    X = sm.add_constant(x)
    est = sm.OLS(y, X).fit()
    summ = est.summary().as_csv()
    f_summ = open('./datasets/gdp/renkou_results.csv', 'w', encoding='utf-8')
    print(summ, end='\n', sep='', file=f_summ)
    params = est.params
    params.to_csv("./datasets/gdp/params_renkou.csv")


if __name__ == '__main__':
    ms = ['make_train_datasets', 'gdp_regresion', 'make_renkou_train_datasets', 'regresion_renkouo']

    method = 'gdp_regresion'

    if method == 'make_train_datasets_big':
        make_train_datasets_big()

    if method == 'gdp_regresion':
        gdp_regresion()

    if method == 'make_renkou_train_datasets':
        make_renkou_train_datasets()

    if method == 'regresion_renkouo':
        regresion_renkouo()
