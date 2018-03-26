import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error  # 回归模型使用
import csv

df = pd.read_csv('pfm_train _digitization2.csv')
y = df['Attrition']
del df['Attrition']
fea = ['Department', 'BusinessTravel', 'JobLevel', 'Gender', 'WorkLifeBalance', 'PerformanceRating']
for f in fea:
    del df[f]

X = df
names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
print("{},{},{},{}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_std = scaler.transform(X_train)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=10)


def randomforest():
    deletecolumns = []
    for tree in [50, 100, 200, 300, 400, 500]:
        rf = RandomForestClassifier(n_estimators=tree,
                                    criterion="entropy",
                                    max_depth=None,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.,
                                    max_features="auto",
                                    max_leaf_nodes=None,
                                    min_impurity_decrease=0.,
                                    min_impurity_split=None,
                                    bootstrap=True,
                                    oob_score=False,
                                    n_jobs=1,
                                    random_state=None,
                                    verbose=0,
                                    warm_start=False,
                                    class_weight=None)
        rf.fit(X_train, y_train)
        joblib.dump(rf, "rf.model")
        y_pred = rf.predict(X_test)
        predicts = pd.DataFrame()
        predicts['true'] = y_test
        predicts['pred'] = y_pred
        predicts.to_csv("predicts_{}.csv".format(tree), index=None)
        acc_val = abs(y_test - y_pred)
        k = 0
        for x in acc_val:
            if x == 0:
                k += 1
            else:
                k += 0
        acc = k / len(y_test)
        print("acc:{}".format(acc))
        ma = confusion_matrix(y_test, y_pred)
        score = rf.score(X_test, y_test)
        results = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)
        for score in results:
            if score[0] < 0.02:
                name = score[1]
                deletecolumns.append(name)
    fw = open("./datasets/deletecolumns.csv", 'w', encoding='utf-8')
    for x in deletecolumns:
        fw.writelines(x + "\n")


def train():
    rf_param_1 = {'n_estimators': range(1, 100)}
    rf_grid_1 = GridSearchCV(RandomForestClassifier(n_estimators=50,
                                                    criterion="gini",
                                                    max_depth=None,
                                                    min_samples_split=2,
                                                    min_samples_leaf=1,
                                                    min_weight_fraction_leaf=0.,
                                                    max_features="auto",
                                                    max_leaf_nodes=None,
                                                    min_impurity_decrease=0.,
                                                    min_impurity_split=None,
                                                    bootstrap=True,
                                                    oob_score=False,
                                                    n_jobs=1,
                                                    random_state=None,
                                                    verbose=0,
                                                    warm_start=False,
                                                    class_weight=None), param_grid=rf_param_1, cv=cv)
    rf_grid_1.fit(X_train_std, y_train)
    print('Parameter with best score:')
    print(rf_grid_1.best_params_)
    print('Cross validation score:', rf_grid_1.best_score_)


def train_1():
    rf_param_2 = {'max_depth': range(3, 29, 1)}
    rf_grid_2 = GridSearchCV(RandomForestClassifier(n_estimators=29, random_state=10), param_grid=rf_param_2, cv=cv)
    rf_grid_2.fit(X_train_std, y_train)
    print('Parameter with best score:')
    print(rf_grid_2.best_params_)
    print('Cross validation score:', rf_grid_2.best_score_)


def train_2():
    rf_param_4 = {'max_features': range(3, 15, 1)}
    rf_grid_4 = GridSearchCV(RandomForestClassifier(n_estimators=29, random_state=13, max_depth=13),
                             param_grid=rf_param_4, cv=cv)
    rf_grid_4.fit(X_train_std, y_train)
    print('Parameter with best score:')
    print(rf_grid_4.best_params_)
    print('Cross validation score:', rf_grid_4.best_score_)


def storFile(data, fileName):
    data = list(map(lambda x: [x], data))
    with open(fileName, 'w', newline='') as f:
        mywrite = csv.writer(f)
        for i in data:
            mywrite.writerow(i)


def rftest():
    rf_param_2 = {'max_depth': range(3, 29, 1)}
    rf_grid_2 = GridSearchCV(RandomForestClassifier(n_estimators=29, random_state=10), param_grid=rf_param_2, cv=cv)
    rf_grid_2.fit(X_train_std, y_train)
    print('Parameter with best score:')
    print(rf_grid_2.best_params_)
    print('Cross validation score:', rf_grid_2.best_score_)
    best_rf = rf_grid_2.best_estimator_
    print('Test score:', best_rf.score(X_test_std, y_test))
    # best_rf.fit(X_test_std, y_test)
    print(best_rf.predict(X_test_std))

    df_1 = pd.read_csv('pfm_test_digitization2.csv')
    std_df_1 = stdsc.transform(df_1)
    answer = best_rf.predict(std_df_1)

    storFile(answer, 'pfm_test10_n=29_f=11_d=16_n_splits=5_entropy.csv')
    print(answer)

    features = X.columns
    feature_importances = best_rf.feature_importances_

    features_df = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})
    features_df = features_df.sort_values('Importance Score', inplace=True, ascending=False)
    print(features_df)


# features = pd.read_csv("./datasets/deletecolumns.csv").values
# fea = [i for j in features for i in j]
# print(set(fea))
train()
