from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from tqdm import trange
from sklearn.externals import joblib
import matplotlib.pyplot as plt


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


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def train():
    data = pd.read_csv("./datasets/other/train_clean.csv")
    data = shuffle(data)
    y = data['label']
    del data['label']
    X = data

    for e in trange(1, 100):
        # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        for batch_x, batch_y in minibatches(inputs=X, targets=y, batch_size=1000):
            wclf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
                           coef0=0.0, shrinking=True, probability=False,
                           tol=1e-3, cache_size=200, class_weight={1: 10},
                           verbose=False, max_iter=-1, decision_function_shape='ovr',
                           random_state=None)

            wclf.fit(batch_x, batch_y)

            if e % 10 == 0:
                clf = svm.SVC(kernel='rbf', C=1.0)
                clf.fit(X, y)
                # plot separating hyperplanes and samples
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
                plt.legend()

                # plot the decision functions for both classifiers
                ax = plt.gca()
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                # create grid to evaluate model
                xx = np.linspace(xlim[0], xlim[1], 30)
                yy = np.linspace(ylim[0], ylim[1], 30)
                YY, XX = np.meshgrid(yy, xx)
                xy = np.vstack([XX.ravel(), YY.ravel()]).T

                # get the separating hyperplane
                Z = clf.decision_function(xy).reshape(XX.shape)

                # plot decision boundary and margins
                a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

                # get the separating hyperplane for weighted classes
                Z = wclf.decision_function(xy).reshape(XX.shape)

                # plot decision boundary and margins for weighted classes
                b = ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

                plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
                           loc="upper right")
                plt.savefig("./datasets/other/{}.png".format(e))

                joblib.dump(wclf, "./datasets/other/svm_{}.model".format(e))


def svm_test():
    lm = joblib.load("./datasets/svm_0.model")
    da1 = pd.read_csv("./datasets/results/test/neg_features.csv")
    da2 = pd.read_csv("./datasets/results/test/pos_features.csv")
    print(da1.shape)
    print(da2.shape)
    da1 = clean_dataset(da1)
    print(da1.shape)
    da2 = clean_dataset(da2)
    test = pd.concat([da1, da2])
    del test['0_id']
    del test['10_have_order']

    print(test.shape)
    p = lm.predict(test)
    df = pd.Series(p)
    df.to_csv("./datasets/results/p.csv", index=None)


svm_test()

