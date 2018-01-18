import numpy as np

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.externals import joblib

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


def train_svm():
    data = pd.read_csv("./datasets/results/train/train.csv")
    data = shuffle(data)

    y = data['label']
    del data['id']
    del data['label']
    X = data.fillna(-1).replace(np.inf, 100)
    X = X.reindex(range(len(X)))
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
                joblib.dump(clf_weights, './datasets/results/models/svm_{}_{}.model'.format(e, i))
                log.info(" Save ")


def svmtest():
    data = pd.read_csv("./datasets/results/train/train.csv")
    data = shuffle(data)
    y = data['label']
    del data['id']
    del data['label']
    X = data.fillna(-1).replace(np.inf, 100)
    batch_size = 100
    i = 0
    limit = 1
    for train_x, train_y in minibatches(X, y, batch_size=batch_size):
        clf_weights = joblib.load("./datasets/results/models/svm_2_15.model")
        p = clf_weights.predict(train_x)
        print(p)
        print("- "*20)
        print(train_y.values.tolist())
        i += 1
        if i > limit:
            break

svmtest()