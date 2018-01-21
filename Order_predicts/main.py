import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.linear_model import Lasso, RANSACRegressor, SGDRegressor, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

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
    # 　特征选择
    pos = pd.read_csv("Order_predicts/datasets/results/train/action_pos_features.csv")
    posfillna = pos.fillna(pos.median()).replace(np.inf, 100)
    neg = pd.read_csv("Order_predicts/datasets/results/train/action_neg_features.csv")
    negfillna = neg.fillna(neg.median()).replace(np.inf, 100)
    data = pd.concat([posfillna, negfillna])
    data = shuffle(data)
    data.to_csv("Order_predicts/datasets/results/train.csv", index=None)
    log.info("train data save succes ...")
    del data['id']
    Y = data['label']
    del data['label']
    X = data
    names = data.columns
    rf = RandomForestRegressor(n_estimators=10, criterion="mse", max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="log2",
                               max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None,
                               bootstrap=True, oob_score=False, n_jobs=1, random_state=1, verbose=0,
                               warm_start=False)
    rf.fit(X, Y)

    res = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)

    for x in res:
        log.info("{}: {}".format(x[0], x[1]))


def train_models(model_name, epoch=5, batch_size=100):
    log.info("current model:{}".format(model_name))
    pos = pd.read_csv("Order_predicts/datasets/results/train/action_pos_features.csv")
    posfillna = pos.fillna(pos.median()).replace(np.inf, 100)
    neg = pd.read_csv("Order_predicts/datasets/results/train/action_neg_features.csv")
    negfillna = neg.fillna(neg.median()).replace(np.inf, 100)
    data = pd.concat([posfillna, negfillna])
    data = shuffle(data)
    del data['id']
    y = data['label']
    del data['label']
    scaler = preprocessing.StandardScaler().fit(data)
    X = scaler.transform(data)
    pd.DataFrame(X).to_csv("Order_predicts/datasets/results/scale_x.csv", index=None)
    data_scaled = preprocessing.scale(X)
    log.info("data shape: {}".format(data_scaled.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    log.info("{}, {}".format(X_train.shape, X_test.shape))

    i = 0
    for e in range(epoch):
        for train_x, train_y in minibatches(X_train, y_train, batch_size=batch_size, shuffle=False):
            if model_name == 'svc':
                clf_weights = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
                                      coef0=0.0, shrinking=True, probability=False,
                                      tol=1e-3, cache_size=200, class_weight={1: 10},
                                      verbose=False, max_iter=-1, decision_function_shape='ovr',
                                      random_state=0)
            elif model_name == 'svr':
                clf_weights = svm.SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                                      tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                                      cache_size=200, verbose=False, max_iter=-1)
            elif model_name == 'lasso':
                clf_weights = Lasso(alpha=1.0, fit_intercept=True, normalize=False,
                                    precompute=False, copy_X=True, max_iter=1000,
                                    tol=1e-4, warm_start=False, positive=False,
                                    random_state=0, selection='cyclic')
            elif model_name == 'logistic':
                clf_weights = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0,
                                                 fit_intercept=True, intercept_scaling=1, class_weight={0: 0.1, 1: 0.9},
                                                 random_state=0, solver='newton-cg', max_iter=100,
                                                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
            elif model_name == 'mlpr':
                # learning_rate: {'constant', 'invscaling', 'adaptive'}
                clf_weights = MLPRegressor(hidden_layer_sizes=(100,), activation="logistic",
                                           solver='adam', alpha=0.0001,
                                           batch_size='auto', learning_rate="constant",
                                           learning_rate_init=0.001,
                                           power_t=0.5, max_iter=200, shuffle=True,
                                           random_state=0, tol=1e-4,
                                           verbose=False, warm_start=False, momentum=0.9,
                                           nesterovs_momentum=True, early_stopping=False,
                                           validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                                           epsilon=1e-8)
            elif model_name == 'rf':
                clf_weights = RandomForestClassifier(n_estimators=20, criterion="entropy",
                                                     max_depth=None, min_samples_split=2,
                                                     min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                                     max_features="auto", max_leaf_nodes=None,
                                                     min_impurity_decrease=0., min_impurity_split=None,
                                                     bootstrap=True, oob_score=False, n_jobs=1, random_state=0,
                                                     verbose=0, warm_start=False, class_weight={0: 0.1, 1: 0.9})
            elif model_name == 'adaboost':
                base_estimator = RandomForestClassifier(n_estimators=20, criterion="entropy",
                                                        max_depth=None, min_samples_split=2,
                                                        min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                                        max_features="auto", max_leaf_nodes=None,
                                                        min_impurity_decrease=0., min_impurity_split=None,
                                                        bootstrap=True, oob_score=False, n_jobs=1, random_state=0,
                                                        verbose=0, warm_start=False, class_weight={0: 0.1, 1: 0.9})
                base_estimator1 = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0,
                                                     fit_intercept=True, intercept_scaling=1,
                                                     class_weight={0: 0.1, 1: 0.9},
                                                     random_state=0, solver='newton-cg', max_iter=100,
                                                     multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
                clf_weights = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, learning_rate=0.6666,
                                                 algorithm='SAMME.R', random_state=0)

            elif model_name == 'gbr':
                clf_weights = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100,
                                                        subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                                                        min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                                        max_depth=3, min_impurity_decrease=0.,
                                                        min_impurity_split=None, init=None, random_state=0,
                                                        max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                                                        warm_start=False, presort='auto')
            elif model_name == 'qda':
                clf_weights = QuadraticDiscriminantAnalysis(priors=None, reg_param=0., store_covariance=False,
                                                            tol=1.0e-4, store_covariances=None)
            elif model_name == 'lda':
                clf_weights = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None,
                                                         n_components=None, store_covariance=False, tol=1e-4)
            elif model_name == 'n_n':
                clf_weights = NearestNeighbors(n_neighbors=5, radius=1.0,
                                               algorithm='auto', leaf_size=30, metric='minkowski',
                                               p=2, metric_params=None, n_jobs=1)
            elif model_name == 'gnb':
                clf_weights = GaussianNB(priors=None)
            elif model_name == 'bnb':
                clf_weights = BernoulliNB(alpha=1.0, binarize=.0, fit_prior=True, class_prior=None)
            elif model_name == 'dcc':
                clf_weights = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None,
                                                     min_samples_split=2, min_samples_leaf=1,
                                                     min_weight_fraction_leaf=0., max_features=None,
                                                     random_state=0, max_leaf_nodes=None,
                                                     min_impurity_decrease=0., min_impurity_split=None,
                                                     class_weight=None, presort=False)
            elif model_name == 'dcr':
                clf_weights = DecisionTreeRegressor(criterion="mse", splitter="best", max_depth=None,
                                                    min_samples_split=2, min_samples_leaf=1,
                                                    min_weight_fraction_leaf=0., max_features=None,
                                                    random_state=0, max_leaf_nodes=None,
                                                    min_impurity_decrease=0., min_impurity_split=None, presort=False)
            elif model_name == 'RAN':
                base_estimator = LinearRegression()                                                                                                                                                                                                                                         
                clf_weights = RANSACRegressor(base_estimator=base_estimator, min_samples=None,
                                              residual_threshold=None, is_data_valid=None,
                                              is_model_valid=None, max_trials=100, max_skips=np.inf,
                                              stop_n_inliers=np.inf, stop_score=np.inf,
                                              stop_probability=0.99, residual_metric=None,
                                              loss='absolute_loss', random_state=0)
            elif model_name == 'adar':
                clf_weights = AdaBoostRegressor(base_estimator=None, n_estimators=50,
                                                learning_rate=1., loss='linear', random_state=None)

            else:  # model_name == 'SGDR':
                clf_weights = SGDRegressor(loss="squared_loss", penalty="l2", alpha=0.0001,
                                           l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None,
                                           shuffle=True, verbose=0, epsilon=0.1,
                                           random_state=None, learning_rate="invscaling", eta0=0.01,
                                           power_t=0.25, warm_start=False, average=False, n_iter=None)

            # build
            clf_weights.fit(train_x, train_y)
            i += 1

            if i % 20 == 0:
                mse = mean_squared_error(y_test, clf_weights.predict(X_test))
                log.info("均方误差：{}".format(mse))
                avgscores = cross_val_score(clf_weights, train_x, train_y).mean()
                log.info("{}/{} 训练集得分平均值：　{}".format(e, i, avgscores))
                model_path = os.path.join("Order_predicts/datasets/results/models", '{}'.format(model_name))
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                joblib.dump(clf_weights, os.path.join(model_path, "{}_{}.model".format(e, i)))
                log.info(" Save ")

            if i % 50 == 0:
                scores = clf_weights.score(X_test, y_test)
                log.info("验证得分：　{}".format(scores))


def modeltest(model_name):
    neg = pd.read_csv("Order_predicts/datasets/results/test/action_neg_features.csv")
    negfillna = neg.fillna(neg.median()).replace(np.inf, 100)
    data = shuffle(negfillna)
    ids = data['id'].values.tolist()
    df_push = pd.DataFrame()
    linenumber = 0
    clf_weights = joblib.load("Order_predicts/datasets/results/models/{}/42_760.model".format(model_name))

    for i in tqdm(ids):
        batch_x = data[data['id'].isin([i])]
        del batch_x['id']
        p = clf_weights.predict(batch_x.values)
        prob = clf_weights.predict_proba(batch_x.values)[0][1]
        # log.info("{}, {:0.8f}".format(p[0], prob))
        df_push.loc[linenumber, 'userid'] = int(i)
        df_push.loc[linenumber, 'orderType'] = "{:0.8f}".format(prob)
        linenumber += 1

    df_push.to_csv("Order_predicts/datasets/results_push.csv", index=None)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        exit(1)
    method = sys.argv[1]

    if method == 'select':
        randomforest()
    if method == 'train':
        m_names = ['svm', 'svr', 'lasso', 'mlpr', 'rf', 'adaboost', 'gbr', 'qda',
                   'lda', 'n_n', 'gnb', 'bnb', 'dcc', 'RAN', 'SGDR']
        log.info("Total number models: {}".format(len(m_names)))

        train_models(model_name='logistic', epoch=50, batch_size=2000)
    if method == "test":
        modeltest(model_name='logistic')
