import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

da = pd.read_csv("./datasets/lassodata.csv")
da.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
              'x11', 'x12', 'x13', 'y']
y = da['y']
del da['y']
X = da
print(X.shape, y.shape)


def f1():
    clf = LassoCV()
    sfm = SelectFromModel(clf, threshold=0.25)
    sfm.fit(X, y)

    while n_features > 2:
        sfm.threshold += 0.1
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]

        plt.title("Features selected from Boston using SelectFromModel with "
                  "threshold %0.3f." % sfm.threshold)
        feature1 = X_transform[:, 0]
        feature2 = X_transform[:, 1]
        plt.plot(feature1, feature2, 'r.')
        plt.xlabel("Feature number 1")
        plt.ylabel("Feature number 2")
        plt.ylim([np.min(feature2), np.max(feature2)])
        plt.show()


def f2():
    lassocv = LassoCV()
    mask = lassocv.coef_ != 0
    new_reg_data = X[:, mask]
    print(new_reg_data.shape)


def f3():
    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.datasets import load_boston
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LassoCV

    # Load the boston dataset.
    boston = load_boston()
    X, y = boston['data'], boston['target']
    print(X.shape, y.shape)
    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    clf = LassoCV()

    # Set a minimum threshold of 0.25
    sfm = SelectFromModel(clf, threshold=0.25)
    sfm.fit(X, y)
    n_features = sfm.transform(X).shape[1]

    while n_features > 2:
        sfm.threshold += 0.1
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]

    plt.title(
        "Features selected from Boston using SelectFromModel with "
        "threshold %0.3f." % sfm.threshold)
    feature1 = X_transform[:, 0]
    feature2 = X_transform[:, 1]
    plt.plot(feature1, feature2, 'r.')
    plt.xlabel("Feature number 1")
    plt.ylabel("Feature number 2")
    plt.ylim([np.min(feature2), np.max(feature2)])
    plt.show()

f3()