import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

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


def train_svm():
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
    data.to_csv("Order_predicts/datasets/results/train.csv", index=None)
    exit(1)
    y = data['label']
    del data['label']
    X = data.fillna(-1).replace(np.inf, 100)
    log.info("data shape: {}".format(X.shape))
    epoch = 10
    batch_size = 2000
    for e in range(epoch):
        i = 0
        for train_x, train_y in minibatches(X, y, batch_size=batch_size, shuffle=False):
            clf_weights = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
                                  coef0=0.0, shrinking=True, probability=False,
                                  tol=1e-3, cache_size=200, class_weight={1: 10},
                                  verbose=False, max_iter=-1, decision_function_shape='ovr',
                                  random_state=None)

            clf_weights.fit(train_x, train_y)
            i += 1
            if i % 15 == 0:
                joblib.dump(clf_weights, 'Order_predicts/datasets/results/models/svm_{}_{}.model'.format(e, i))
                log.info(" Save ")


def svmtest():
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

    y = data['label']
    del data['id']
    del data['label']
    X = data.fillna(-1).replace(np.inf, 100)
    batch_size = 100
    i = 0
    limit = 1
    for train_x, train_y in minibatches(X, y, batch_size=batch_size):
        clf_weights = joblib.load("Order_predicts/datasets/results/models/svm_1_15.model")
        p = clf_weights.predict(train_x)
        print("predicts:{}".format(p))
        print("- " * 20)
        print(train_y.values.tolist())
        print("- " * 20)
        print("- " * 20)

        print()
        i += 1
        if i > limit:
            break


def logistic():
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
    # data.to_csv("Order_predicts/datasets/results/train.csv", index=None)
    # exit(1)
    y = data['label']
    del data['label']
    X = data.fillna(-1).replace(np.inf, 100)
    log.info("data shape: {}".format(X.shape))
    epoch = 10
    batch_size = 2000
    for e in range(epoch):
        i = 0
        for train_x, train_y in minibatches(X, y, batch_size=batch_size, shuffle=False):
            los = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight={1: 10},
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
            los.fit(train_x, train_y)

            i += 1
            if i % 15 == 0:
                joblib.dump(los, 'Order_predicts/datasets/results/models/logistic_{}_{}.model'.format(e, i))
                log.info(" Save ")


def logistictest():
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
        clf_weights = joblib.load("Order_predicts/datasets/results/models/logistic_1_15.model")
        p = clf_weights.predict(batch_x.values)
        print("predicts:{}".format(p))
        print("- " * 20)


# logistic()
logistictest()