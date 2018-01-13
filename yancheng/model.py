from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.externals import joblib
import random
from tqdm import tqdm

from pyduyp.logger.log import log
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from pyduyp.utils.utils import list_reverse_pop

scalar = StandardScaler()


def linear_model():
    data = pd.read_csv("./datasets/results/data_train.csv")
    lm = LinearRegression()
    x, y = data['week'].values, data['cnt'].values
    train_x = np.array(x[: int(len(x) * 0.7)]).reshape(722, 1)
    train_y = np.array(y[: int(len(x) * 0.7)]).reshape(722, )
    train_x = preprocessing.scale(train_x)
    train_y = preprocessing.scale(train_y)
    lm.fit(train_x, train_y)
    joblib.dump(lm, "./datasets/results/linear_model.m")

    data = pd.read_csv("./datasets/test_A_20171225.txt", sep="\t")
    test_x = data['day_of_week'].values.reshape(len(data), 1)
    mean_y, std_y = np.mean(test_x), np.std(test_x)

    test_x_scaled = preprocessing.scale(test_x)
    p = lm.predict(test_x_scaled)
    print(p)
    p_real = p * std_y + mean_y
    print(p_real)


def inear_model_t():
    lm = joblib.load("./datasets/results/linear_model.m")
    data = pd.read_csv("./datasets/test_A_20171225.txt", sep="\t")
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

    print(predicts)
    out['predict'] = predicts
    out.to_csv("./datasets/results/predicts_A.csv", index=None)


if __name__ == "__main__":
    GP()
