import numpy as np
import math
from statsmodels.tsa.stattools import adfuller
import pandas as pd

history_data = np.loadtxt('populations.txt')


def stationaritytest_(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]
    # 此函数返回的是p值


def best_diff(df, maxdiff=8):
    p_set = {}
    for i in range(0, maxdiff):
        temp = df.copy()  # 每次循环前，重置
        if i == 0:
            temp['diff'] = temp[temp.columns[1]]
        else:
            temp['diff'] = temp[temp.columns[1]].diff(i)
            temp = temp.drop(temp.iloc[:i].index)  # 差分后，前几行的数据会变成nan，所以删掉
        pvalue = stationaritytest_(temp['diff'])
        p_set[i] = pvalue
        p_df = pd.DataFrame.from_dict(p_set, orient="index")
        p_df.columns = ['p_value']
    i = 0
    while i < len(p_df):
        if p_df['p_value'][i] < 0.01:
            bestdiff = i
            break
        i += 1
    return bestdiff


def fun1():
    # history_data = [724.57, 746.62, 778.27, 800.8, 827.75, 871.1, 912.37, 954.28, 995.01, 1037.2]
    history_data = np.loadtxt('populations.txt')
    history_data2list = history_data.tolist()
    print(history_data2list)
    n = len(history_data)
    X0 = np.array(history_data)

    # 累加生成
    history_data_agg = [sum(history_data[0:i + 1]) for i in range(n)]
    X1 = np.array(history_data_agg)

    # 计算数据矩阵B和数据向量Y
    B = np.zeros([n - 1, 2])
    Y = np.zeros([n - 1, 1])
    for i in range(0, n - 1):
        B[i][0] = -0.5 * (X1[i] + X1[i + 1])
        B[i][1] = 1
        Y[i][0] = X0[i + 1]

    # 计算GM(1,1)微分方程的参数a和u
    # A = np.zeros([2,1])
    A = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)  # np.linalg.inv矩阵求逆, dot两个数组的点积
    a = A[0][0]
    u = A[1][0]

    # 建立灰色预测模型
    XX0 = np.zeros(n)
    XX0[0] = X0[0]
    for i in range(1, n):
        XX0[i] = (X0[0] - u / a) * (1 - math.exp(a)) * math.exp(-a * i)

    # 模型精度的后验差检验
    e = 0  # 求残差平均值
    for i in range(0, n):
        e += (X0[i] - XX0[i])
    e /= n

    # 求历史数据平均值
    aver = 0
    for i in range(0, n):
        aver += X0[i]
    aver /= n

    # 求历史数据方差
    s12 = 0
    for i in range(0, n):
        s12 += (X0[i] - aver) ** 2
    s12 /= n

    # 求残差方差
    s22 = 0
    for i in range(0, n):
        s22 += ((X0[i] - XX0[i]) - e) ** 2
    s22 /= n

    # 求后验差比值
    C = s22 / s12

    # 求小误差概率
    cout = 0
    for i in range(0, n):
        if abs((X0[i] - XX0[i]) - e) < 0.6754 * math.sqrt(s12):
            cout = cout + 1
        else:
            cout = cout
    P = cout / n

    if C < 0.35 and P > 0.95:
        # 预测精度为一级
        m = 5  # 请输入需要预测的年数  [ 394.00901644  398.77168963  403.59193271  408.47044156  413.4079205 ]
        f = np.zeros(m)
        for i in range(0, m):
            f[i] = (X0[0] - u / a) * (1 - math.exp(a)) * math.exp(-a * (i + n))
            print(f)
    else:
        print('灰色预测法不适用')


def gm11(x, n):
    x1 = x.cumsum()
    z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0  # 紧邻均值
    z1 = z1.reshape((len(z1), 1))
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Y = x[1:].reshape((len(x) - 1, 1))
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)  # 计算参数
    return (x[0] - b / a) * np.exp(-a * (n - 1)) - (x[0] - b / a) * np.exp(-a * (n - 2))


# dt = [323.3, 326.5, 330.0, 337.2, 340.3, 343.2, 344.9, 347.0, 349.3, 352.5, 355.3, 358.3, 360.6, 362.2, 365.4, 367.8,
#       366.6, 371.1, 376.3]
# print(gm11(np.array(dt), 5)) # 338.544202603

fun1()
