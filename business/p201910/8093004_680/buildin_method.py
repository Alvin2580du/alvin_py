from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, zero_one_loss
import numpy as np


def loaddata():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    with open('./data/wdbc_data.csv', 'r') as fr:
        lines = fr.readlines()
        num = -1
        for line in lines:
            line_spl = line.split(',')
            y_ = 1.0
            if line_spl[1] == 'B':
                y_ = -1.0
            tmp_ = [float(i) for i in line_spl[2:]]
            num += 1
            if num < 300:  # 前300 作为训练集
                train_y.append(y_)
                train_x.append(tmp_)
            else:
                test_y.append(y_)
                test_x.append(tmp_)
    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = loaddata()
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME", n_estimators=200, learning_rate=0.8)
bdt.fit(train_x, train_y)
y_pred = bdt.predict(test_x)
print(accuracy_score(y_true=test_y, y_pred=y_pred))



