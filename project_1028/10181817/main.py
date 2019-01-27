import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# libraries for model selection and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score, \
    accuracy_score

# libraries for model fitting
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# import data
voice = pd.read_csv("voice.csv")
y = voice.label == 'male'
X = voice.drop(['label'], axis=1)

voice.head(5)

# fix the random seed for reproducibility
np.random.seed(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

classifier_names = ["Logistic Regression",
                    "Nearest Neighbors",
                    "Linear SVM",
                    "RBF SVM",
                    "Random Forest",
                    "AdaBoost"]

classifiers = [LogisticRegression(C=1., solver='lbfgs'),
               KNeighborsClassifier(3),
               SVC(kernel="linear", C=0.025, probability=True),
               SVC(gamma=2, C=1, probability=True),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               AdaBoostClassifier()]

pred_prob = {}
pred = {}

# fit individual classifiers
for name, classifier in zip(classifier_names, classifiers):
    np.random.seed(100)
    classifier.fit(X_train, y_train)
    pred_prob[name] = classifier.predict_proba(X_test)[:, 1]
    pred[name] = np.where(pred_prob[name] >= 0.5, 1, 0)

# 1. confusion matrix
cm = confusion_matrix(y_test, pred['Logistic Regression'])
print("confusion matrix of Logistic Regression models:\n")
print(cm)

"""
[[645 127]
 [ 38 774]]
812 772
"""


def roc_point(y_test, y_pred):
    y_test, y_pred = y_test.values.tolist(), list(y_pred)
    N, P = 0, 0
    y_test_new = []
    for x in y_test:
        if x:
            N += 1
            y_test_new.append(1)
        else:
            y_test_new.append(0)
            P += 1
    FP, TP = 0, 0
    R = []
    pprev = -np.inf
    i = 1
    n = len(y_test_new)

    while i <= n:
        if y_pred[i-1] != pprev:
            R.append((FP / N, TP / P))
            pprev = y_pred[i]
        if y_test_new[i-1] == 1:
            TP += 1
        else:
            FP += 1
        i += 1
    return R

r = roc_point(y_test, pred['Logistic Regression'])
print(r)