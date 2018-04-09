import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, Lars, lars_path
from sklearn.neural_network import MLPRegressor

da = pd.read_csv("./datasets/libai.csv")
da.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
              'x11', 'x12', 'x13', 'y']
y = da['y']
del da['y']
X = da
print(X.shape, y.shape)


def lasso_features_selection():
    lassocv = LassoCV()
    lassocv.fit(X, y)
    mask = lassocv.coef_ != 0
    print("mask:{}".format(mask), type(mask), len(mask))
    new_reg_data = X.loc[:, mask]
    print(new_reg_data.shape)
    print(type(new_reg_data), new_reg_data)
    ydf = pd.DataFrame(y)
    df = pd.concat([new_reg_data, ydf], axis=1)
    if not os.path.exists("./datasets"):
        os.makedirs("./datasets")

    df.to_csv("./datasets/lasso_select.csv", index=None)
    # 选择后的文件 保存到datasets 目录下面


def nn():
    # 神经网络模型
    from sklearn.model_selection import train_test_split
    data = pd.read_csv("./datasets/lasso_select.csv")
    y = data['y']
    del data['y']
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # 下面的参数都可以调，尝试修改不同的值，看看预测的结果会不会变好。
    mlp = MLPRegressor(hidden_layer_sizes=(200000,), activation="tanh",
                       solver='adam', alpha=0.01,
                       batch_size='auto', learning_rate="constant",
                       learning_rate_init=0.01,
                       power_t=0.5, max_iter=1000, shuffle=True,
                       random_state=None, tol=1e-4,
                       verbose=False, warm_start=False, momentum=0.9,
                       nesterovs_momentum=True, early_stopping=False,
                       validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-8)
    mlp.fit(X_train, y_train)
    loss = mlp.loss_
    print(loss, len(str(loss).split(".")[0]))
    # 这里主要看loss的值，越小越好
    # coefs = mlp.coefs_
    # inter = mlp.intercepts_
    # n_iter = mlp.n_iter_
    # nlayers = mlp.n_layers_
    # noutputs = mlp.n_outputs_
    # print(loss, coefs, n_iter, inter, nlayers, noutputs)
    y_pred = mlp.predict(X_test)

    predicts = pd.DataFrame()
    predicts['true'] = y_test
    predicts['pred'] = y_pred
    predicts.to_csv("./datasets/predicts_total_values.csv", index=None)
    # 这里在datasets目录下，保存了一份预测的结果文件。


def huiseyuce(history_data):
    # 灰色预测模型
    history_data, h_test = history_data[0:12], history_data[12:]
    X0 = np.array(history_data)
    n = len(history_data)

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
        m = 4  # 请输入需要预测的年数  [ 394.00901644  398.77168963  403.59193271  408.47044156  413.4079205 ]
        f = np.zeros(m)
        for i in range(0, m):
            f[i] = (X0[0] - u / a) * (1 - math.exp(a)) * math.exp(-a * (i + n))

        print(f)
        print(h_test.values)
        df = pd.DataFrame()
        df['y'] = h_test.values
        df['y_pred'] = f
        df.to_csv("./datasets/huiseyuce_results.csv", index=None)

    else:
        print('灰色预测法不适用')


def build_huiseyuce():

    data = pd.read_csv("./datasets/lasso_select.csv", usecols=['x6', 'x11'])
    x6 = data['x6']
    x11 = data['x11']
    huiseyuce(x6)
    huiseyuce(x11)


if __name__ == "__main__":
    method = 'lasso_features_selection'
    # 选择lasso_features_selection, nn, huiseyuce中的一个，执行下面的代码

    if method == 'lasso_features_selection':
        lasso_features_selection()
    if method == 'nn':
        nn()
    if method == 'huiseyuce':
        huiseyuce()
