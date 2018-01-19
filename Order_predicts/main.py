import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

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


def train_models(model_name, epoch=5, batch_size=100):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    log.info("{}, {}".format(X_train.shape, X_test.shape))

    for e in range(epoch):
        i = 0
        for train_x, train_y in minibatches(X_train, y_train, batch_size=batch_size, shuffle=False):
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
            if model_name == 'nn':
                # learning_rate: {'constant', 'invscaling', 'adaptive'}
                clf_weights = MLPRegressor(hidden_layer_sizes=(100,), activation="logistic",
                                           solver='adam', alpha=0.0001,
                                           batch_size='auto', learning_rate="constant",
                                           learning_rate_init=0.001,
                                           power_t=0.5, max_iter=200, shuffle=True,
                                           random_state=None, tol=1e-4,
                                           verbose=False, warm_start=False, momentum=0.9,
                                           nesterovs_momentum=True, early_stopping=False,
                                           validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                                           epsilon=1e-8)
            if model_name == 'rf':
                clf_weights = RandomForestClassifier(n_estimators=10, criterion="gini",
                                                     max_depth=None, min_samples_split=2,
                                                     min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                                     max_features="auto", max_leaf_nodes=None,
                                                     min_impurity_decrease=0., min_impurity_split=None,
                                                     bootstrap=True, oob_score=False, n_jobs=1, random_state=None,
                                                     verbose=0, warm_start=False, class_weight=None)
            if model_name == 'adaboost':
                clf_weights = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.,
                                                 algorithm='SAMME.R', random_state=None)

            # build
            clf_weights.fit(train_x, train_y)

            i += 1
            if i % 5 == 0:
                avgscores = cross_val_score(clf_weights, train_x, train_y).mean()
                log.info("训练集得分平均值：　{}".format(avgscores))
                joblib.dump(clf_weights,
                            'Order_predicts/datasets/results/models/{}_{}_{}.model'.format(model_name, e, i))
                log.info(" Save ")

            if i % 30 == 0:
                scores = clf_weights.score(X_test, y_test)
                log.info("验证得分：　{}".format(scores))


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
        clf_weights = joblib.load("Order_predicts/datasets/results/models/{}_3_70.model".format(model_name))
        p = clf_weights.predict(batch_x.values)
        print(p)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        exit(1)
    method = sys.argv[1]

    if method == 'train':
        train_models(model_name='rf')
    if method == "test":
        modeltest(model_name='rf')
