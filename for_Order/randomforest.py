import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing

df = pd.read_csv('pfm_train _digitization2.csv')
y = df['Attrition']
del df['Attrition']
fea = ['Department', 'BusinessTravel', 'JobLevel', 'Gender', 'WorkLifeBalance', 'PerformanceRating']
for f in fea:
    del df[f]

X = df
names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
print("训练集：{},标签：{}, 测试集：{}, 标签：{}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
# 训练集标准化
scale_method = 'z-score'
if scale_method == 'z-score':
    # 归一化到0均值，1标准差
    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    X_train_std = scaler.transform(X_train)
    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_test)
    X_test_std = scaler.transform(X_test)
if scale_method == 'one-zero':
    # 归一化到[0,1]之间
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_std = min_max_scaler.fit_transform(X_train)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_test_std = min_max_scaler.fit_transform(X_test)

if scale_method == 'oneandneg-one':
    # 归一化到[-1,1]之间

    max_abs_scaler = preprocessing.MaxAbsScaler()
    x_maxabs = max_abs_scaler.fit_transform(X_train)


def train(step='train'):
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=10)

    rf_param_1 = {'n_estimators': range(1, 100), 'max_depth': range(3, 29, 1), 'max_features': range(3, 15, 1)}
    rf_grid_1 = GridSearchCV(RandomForestClassifier(n_estimators=10,
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
    if step == 'test':
        best_rf = rf_grid_1.best_estimator_
        print('Test score:', best_rf.score(X_test_std, y_test))
        print(best_rf.predict(X_test_std))
        features = X.columns
        feature_importances = best_rf.feature_importances_
        features_df = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})
        features_df = features_df.sort_values('Importance Score', inplace=True, ascending=False)
        print(features_df)


train()
