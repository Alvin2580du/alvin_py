import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.linear_model import MultiTaskLasso, Lasso

import pandas as pd
from pyduyp.logger.log import log


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        global indices
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def randomforest():
    pos = pd.read_csv("Order_predicts/datasets/results/train/action_pos_features.csv")
    neg = pd.read_csv("Order_predicts/datasets/results/train/action_neg_features.csv")
    data = pd.concat([pos, neg])
    data = data.fillna(-1).replace(np.inf, 100)
    data = shuffle(data)
    del data['id']
    Y = data['label']
    del data['label']
    X = data
    names = data.columns
    rf = RandomForestRegressor()
    rf.fit(X, Y)

    res = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)

    for x in res:
        print(x[0], x[1])


def train_models(model_name='lasso'):
    pos = pd.read_csv("Order_predicts/datasets/results/train/action_pos_features.csv")
    neg = pd.read_csv("Order_predicts/datasets/results/train/action_neg_features.csv")
    data = pd.concat([pos, neg])
    data = shuffle(data)

    del data['16_tmode']
    del data['10_t9']
    del data['28_tmode']
    del data['27_atmedian']
    del data['29_atptp']
    del data['continent']
    del data['province']
    del data['country']
    del data['city']
    del data['age']
    del data['id']

    y = data['label']
    del data['label']
    X = data.fillna(-1).replace(np.inf, 100)
    scaler = preprocessing.StandardScaler().fit(X)
    scaler.transform(X)
    X.to_csv("Order_predicts/datasets/results/scale_x.csv", index=None)
    data_scaled = preprocessing.scale(X)
    log.info("data shape: {}".format(data_scaled.shape))
    epoch = 10
    batch_size = 2000
    for e in range(epoch):
        i = 0
        for train_x, train_y in minibatches(data_scaled, y, batch_size=batch_size, shuffle=False):
            if model_name == 'svc':
                clf_weights = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
                                      coef0=0.0, shrinking=True, probability=False,
                                      tol=1e-3, cache_size=200, class_weight={1: 10},
                                      verbose=False, max_iter=-1, decision_function_shape='ovr',
                                      random_state=None)
            if model_name == 'svr':

                clf_weights = svm.SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                                      tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                                      cache_size=200, verbose=False, max_iter=-1)
            if model_name == 'lasso':

                clf_weights = Lasso(alpha=1.0, fit_intercept=True, normalize=False,
                                    precompute=False, copy_X=True, max_iter=1000,
                                    tol=1e-4, warm_start=False, positive=False,
                                    random_state=None, selection='cyclic')
            if model_name == 'logistic':

                clf_weights = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0,
                                                 fit_intercept=True, intercept_scaling=1, class_weight={1: 10},
                                                 random_state=None, solver='liblinear', max_iter=100,
                                                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
            clf_weights.fit(train_x, train_y)
            i += 1
            if i % 15 == 0:
                joblib.dump(clf_weights,
                            'Order_predicts/datasets/results/models/{}_{}_{}.model'.format(model_name, e, i))
                log.info(" Save ")


def modeltest(model_name='svm'):
    pos = pd.read_csv("Order_predicts/datasets/results/test/action_pos_features.csv")
    neg = pd.read_csv("Order_predicts/datasets/results/test/action_neg_features.csv")
    data = pd.concat([pos, neg])
    data = shuffle(data)
    ids = data['id']
    data = data.fillna(-1).replace(np.inf, 100)
    del data['16_tmode']
    del data['10_t9']
    del data['28_tmode']
    del data['27_atmedian']
    del data['29_atptp']
    del data['continent']
    del data['province']
    del data['country']
    del data['city']
    del data['age']

    data = data.fillna(-1).replace(np.inf, 100)
    for i in ids:
        batch_x = data[data['id'].isin([i])]
        del batch_x['id']
        clf_weights = joblib.load("Order_predicts/datasets/results/models/{}_1_15.model".format(model_name))
        p = clf_weights.predict(batch_x.values)
        print(p)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        exit(1)
    method = sys.argv[1]

    if method == 'train':
        train_models(model_name='lasso')
    if method == "test":
        modeltest(model_name='lasso')
