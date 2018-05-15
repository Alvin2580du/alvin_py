import numpy as np
import pandas as pd
from collections import OrderedDict
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC

import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings

warnings.filterwarnings("ignore")


def str2float(inputs):
    return inputs.replace(" ", "").replace(" ", "").replace("\xa0", "").replace("\ufeff", "").replace("\n", "").replace(
        '?', "")


def subs(x1, x2):
    return abs(x1 - x2)


def make_gdp():
    df = pd.read_csv("./datasets/gdp/gdp051311.csv")
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
    del train['年份']
    del train['一般公共预算收入']
    del train['建筑业总产值']
    del train['农业总产值']
    del train['社会消费品零售总额']
    del train['住户存款']
    del train['投资总额']
    del train['关口出口']
    del train['工业总产值']
    del train['一般公共预算支出']
    del train['非税收收入']
    del train['批发和零售业、住宿和餐饮业从业人员']
    y = train['上海市生产总值']
    del train['上海市生产总值']
    x = train
    k = 0
    # VIF
    vif = pd.DataFrame()
    vif['vif'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif['features'] = x.columns
    vif = vif.sort_values(by='vif', ascending=False)
    vif.to_csv("./datasets/gdp/gdp_vif.csv", index=None)
    # 添加常数项
    X = sm.add_constant(x)
    # 最小二乘回归
    est = sm.OLS(y, X).fit()
    # 检验
    summ = est.summary().as_csv()
    f_summ = open('./datasets/gdp/gdp_results_{}.csv'.format(k), 'w', encoding='utf-8')
    print(summ, end='\n', sep='', file=f_summ)
    # 系数
    params = est.params
    params.to_csv("./datasets/gdp/params_gdp_{}.csv".format(k))


def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')


def make_renkou():
    # 年份,常住人口,年末户籍人口,废水排放总量,废水化学需氧量排放总量 ,污水处理厂座数,垃圾产生量,
    # 清运粪便,住宅投资额,上海市生产总值,一般公共预算支出,普通高等学校,获博士学位人数,获硕士学位人数,职工平均工资

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
            rows['年末户籍人口'] = data[2]
            rows['废水排放总量'] = data[3]
            rows['废水化学需氧量排放总量'] = data[4]
            rows['污水处理厂座数'] = data[5]
            rows['垃圾产生量'] = data[6]
            rows['清运粪便'] = data[7]
            rows['住宅投资额'] = data[8]
            rows['上海市生产总值'] = data[9]
            rows['一般公共预算支出'] = data[10]
            rows['普通高等学校'] = data[11]
            rows['获博士学位人数'] = data[12]
            rows['获硕士学位人数'] = data[13]
            rows['职工平均工资'] = data[14]
            save.append(rows)

    X = pd.DataFrame(save, dtype=np.float32)

    X['迁入人口'] = X.apply(lambda row: subs(row['常住人口'], row['年末户籍人口']), axis=1)
    del X['常住人口']
    del X['年末户籍人口']

    print(X.shape)
    y = X['迁入人口']
    del X['迁入人口']
    del X['年份']
    t1 = time.time()

    lassocv = LassoCV(eps=0.0001, n_alphas=10, alphas=None, fit_intercept=True,
                      normalize=False, precompute='auto', max_iter=1000, tol=1e-4,
                      copy_X=True, cv=None, verbose=False, n_jobs=1,
                      positive=False, random_state=None, selection='cyclic').fit(X, y)
    coefs = lassocv.coef_
    new_reg_data = X.loc[:, coefs != 0]
    ydf = pd.DataFrame(y)
    results = pd.concat([new_reg_data, ydf], axis=1)
    # 选择后的文件 保存到datasets 目录下面
    results.to_csv("./datasets/gdp/lasso_select.csv", index=None)
    print(results.shape)

    coefs_df = pd.DataFrame(coefs).T
    coefs_df.columns = X.columns
    coefs_df.to_csv("./datasets/gdp/renkou_coefs.csv")


if __name__ == '__main__':
    ms = ['make_gdp', 'make_renkou']

    method = 'make_renkou'

    if method == 'make_gdp':
        make_gdp()

    if method == 'make_renkou':
        make_renkou()
