import pandas as pd
import numpy as np


def question_1():
    total_invest = pd.read_csv("FinalExamData_Returns.txt", sep='\t', usecols=['Asset_X', 'Asset_Y'])
    print(total_invest.mean()*12)
    print(total_invest.std()*12)

    def question1_1(name='Asset_X'):
        total_invest = pd.read_csv("FinalExamData_Returns.txt", sep='\t', usecols=['Asset_X', 'Asset_Y', 'Market'])

        SR = total_invest[name].mean() / total_invest[name].std()
        print("Sharpe Rate:{}".format(SR))
        ASR = np.sqrt(12) * SR

        return total_invest[name].mean(), total_invest[name].std(), ASR

    for n in ['Asset_X', 'Asset_Y']:
        x, y, z = question1_1(n)
        print("annualized Sharpe ratio:{}, mean:{}, std:{}".format(x, y, z))

    print(total_invest.cov())
import statsmodels.api as sm
from statsmodels import regression
def linreg(x, y):
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    print(model.t_test([1, 0]))
    x = x[:, 1]
    print("R-squared: :")
    print(model.summary())
    return model.params[0], model.params[1]


def questions_2():
    total_invest = pd.read_csv("FinalExamData_Returns.txt", sep='\t', usecols=['Asset_X', 'Asset_Y', 'Market'])

    X = total_invest['Asset_X'].values
    Y = total_invest['Market'].values

    alpha, beta = linreg(X, Y)
    print("alpha: {}, beta: {}".format(alpha, beta))
questions_2()

def question_4():
    import scipy.stats

    four = pd.read_csv("FinalExamData_Uniforms.txt", sep='\t', usecols=['Uniform'])

    bernoulli_sample = []
    # part a

    for x in four.values:
        v = x[0]
        if v <= 0.5:
            bernoulli_sample.append(1)
        else:
            bernoulli_sample.append(0)
    df = pd.DataFrame(bernoulli_sample)
    print("sample sum of bernouli random variables : \n")
    print(df.sum().values[0])
    # part b

    for x in four.values:
        v = x[0]
        if v <= 0.25:
            bernoulli_sample.append(1)
        else:
            bernoulli_sample.append(0)
    df = pd.DataFrame(bernoulli_sample)
    print("mean on either side of 0.85 : {}".format(df.mean().values[0]))

    # part c
    k = 100
    print(scipy.stats.norm.cdf(2.49, loc=2.5, scale=np.sqrt(1.875 / k)))
    print(2 * scipy.stats.norm.cdf(2.49, loc=2.5, scale=np.sqrt(1.875 / k)))
    k = 1000
    print(2 * scipy.stats.norm.cdf(2.49, loc=2.5, scale=np.sqrt(1.875 / k)))
    k = 10000
    print(2 * scipy.stats.norm.cdf(2.49, loc=2.5, scale=np.sqrt(1.875 / k)))
    k = 100000
    print(2 * scipy.stats.norm.cdf(2.49, loc=2.5, scale=np.sqrt(1.875 / k)))


import math


# 计算置信区间的函数
def calc(r, n):
    if n < r:
        print('r cannot be greater than n.')
        return
    if math.floor(r) < r:
        print('r must be an integer value.')
        return
    if math.floor(n) < n:
        print('n must be an integer value.')
        return
    p = round((r / n) * 10000) / 10000
    print('p', p)

    q = 1 - p
    p = round(p * 10000) / 10000
    print('p', p)
    z = 1.95996
    zsq = z * z

    # l95a
    num = (2 * n * p) + zsq - (z * math.sqrt(zsq + (4 * n * p * q)))
    denom = 2 * (n + zsq)
    l95a = num / denom
    if p == 0:
        l95a = 0
    l95a = round(l95a * 10000) / 10000

    # u95a
    num = (2 * n * p) + zsq + (z * math.sqrt(zsq + (4 * n * p * q)))
    denom = 2 * (n + zsq)
    u95a = num / denom
    print('u95a', u95a)
    if p == 1:
        u95a = 1
    u95a = round(u95a * 10000) / 10000
    print('no continuity correction', l95a, '-', u95a)

    # l95b
    num = (2 * n * p) + zsq - 1 - (z * math.sqrt(zsq - 2 - (1 / n) + 4 * p * ((n * q) + 1)))
    denom = 2 * (n + zsq)
    l95b = num / denom
    if p == 0:
        l95b = 0
    l95b = round(l95b * 100000000) / 100000000

    # u95b
    num = (2 * n * p) + zsq + 1 + (z * math.sqrt(zsq + 2 - (1 / n) + 4 * p * ((n * q) - 1)))
    denom = 2 * (n + zsq)
    u95b = num / denom
    if p == 1:
        u95b = 1
    u95b = round(u95b * 100000000) / 100000000
    print('including continuity correction', l95b, '-', u95b)

    return


# 测试k = 12 n = 40

calc(12, 40)
print('THE END')


