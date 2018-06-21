"""

1）由于土壤呼吸速率，决定于土壤C库及其主要参数f1，f2，k1，k2，k3；
2）我能通过文献得知这些参数的范围；
3）通过贝叶斯构建这些参数的后验分布；
4）通过MCMC在后验分布中取样；
5）构造稳定的马尔科夫链做参数估计。


结果要求：

    1）得到稳定的马尔可夫链，即GR=1；
    2）输出参数的frequency直方图；
    3）计算参数的95%致信区间；
    4）计算参数的最大似然估计值或均值；
    5）计算参数相关性；
    6）利用以上得到的参数值计算240天和9960天的累计C排放

"""
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
import pandas as pd
import seaborn as sns

Ct0 = 11.7

zti = [1.104,
       0.817,
       0.417010242,
       0.93365,
       0.6753,
       1.192,
       0.205,
       0.201,
       0.47,
       0.34,
       0.21,
       0.408338512,
       0.281694997,
       0.232370631,
       0.155051482,
       0.131270575,
       0.16151326,
       0.183046266,
       ]


def cauchy(theta):
    # 从柯西分布p中采样数据
    y = 1.0 / (1.0 + theta ** 2)
    return y


def sdnorm(z):
    # 从正态分布中采样
    """
    输入z，返回标准正态分布对应的概率 
    """
    return np.exp(-z * z / 2.) / np.sqrt(2 * np.pi)


def q(x_star, x):
    # normal distribution
    mu = x
    sigma = 10
    return 1 / (math.pi * 2) ** 0.5 / sigma * np.exp(-(x_star - mu) ** 2 / 2 / sigma ** 2)


def p(x):  # target distribution
    return 0.3 * np.exp(-0.2 * x ** 2) + 0.7 * np.exp(-0.2 * (x - 10) ** 2)


def compute_intervals(x):
    x1 = np.array(x)
    mean = x1.mean()
    std = x1.std()
    interval = stats.t.interval(0.95, len(x) - 1, mean, std)
    res = "[{:0.5f},{:0.5f}]".format(interval[0], interval[1])
    return res


def normal_mle(input):
    # 极大似然估计
    x = np.array(input)
    u = np.mean(x)
    return np.sqrt(np.dot(x - u, (x - u).T) / x.shape[0])


def Metropolis_Hastings():
    T = len(zti)
    sigma = 1
    thetamin = 0.0
    thetamax1 = 0.2  # f1
    thetamax2 = 1  # f2
    thetamax3 = 0.7  # k1
    thetamax4 = 0.01  # k2
    thetamax5 = 0.00001  # k3
    thetamax_dict = {"f1": 0.2, 'f2': 1, 'k1': 0.7, 'k2': 0.01, 'k3': 0.00001}
    max_thetas = [thetamax1,
                  # thetamax2,
                  # thetamax3,
                  # thetamax4,
                  # thetamax5
                  ]
    theta_list = []

    for thetamax in max_thetas:
        theta = [0.0] * (T + 1)
        theta[0] = random.uniform(thetamin, thetamax)
        t = 0
        accepted = 0.0
        while t < T:
            t = t + 1
            theta_star = norm.rvs(loc=theta[t - 1], scale=sigma, size=1, random_state=None)
            # 从已知正态分布q中生成候选状态
            alpha = min([1., sdnorm(theta_star[0]) / sdnorm(theta[t - 1])])  # acceptance probability
            x1 = p(theta_star[0]) * q(theta[t - 1], theta_star[0])
            x2 = p(theta[t - 1]) / q(theta_star[0], theta[t - 1])
            alpha = min(1.0, x1 / x2)

            u = random.uniform(0, 1)
            if u <= alpha:  # 接受
                theta[t] = theta_star[0]
                accepted = accepted + 1.0  # monitor acceptance
            else:
                theta[t] = theta[t - 1]
        theta_list.append(theta)
        print("Length: {},{}".format(len(theta), theta[0]))
        print("{} Acceptance rate = {}".format(thetamax, str(accepted / T)))

        # theta 估计
        v = np.exp(np.sum(list(map(lambda x: np.square(x[0] - x[1]) / 2 * np.var(x[0]), zip(zti, theta)))))
        print("v:{}".format(v))
        print("= "*20)

        plt.figure()
        plt.plot(range(T + 1), theta, 'g-')
        plt.savefig("trace_image_{}.png".format(thetamax))
        plt.close()

        plt.figure()
        plt.hist(theta, 50, facecolor='red', alpha=0.5, density=True)
        plt.savefig("hist_image_{}.png".format(thetamax))
        plt.close()

        plt.figure()
        sns.distplot(theta, hist=False)
        plt.savefig("density_image_{}.png".format(thetamax))
        plt.close()
        print(compute_intervals(theta))
        print(np.mean(np.array(theta)))
        print("- " * 20)

    print("- " * 20)
    # res = np.corrcoef(theta_list)
    # df = pd.DataFrame(res)
    # df.columns = thetamax_dict.keys()
    # df.to_csv("./datasets/相关系数矩阵.csv", index=None)


