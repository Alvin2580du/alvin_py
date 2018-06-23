from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Imputer, OneHotEncoder, scale
from sklearn.ensemble import AdaBoostClassifier  # 集成学习
from sklearn.feature_selection import RFE
import pandas as pd

# iris数据集
iris = load_iris()

x = iris.data
y = iris.target
# 预处理,标准化
x_scale = scale(x)

try:

    # one-hot
    enc = OneHotEncoder()
    y_onehot = enc.fit_transform(y.reshape(1, -1))
    imp = Imputer(missing_values='NAN', strategy='mean', axis=0)
    imp.fit(x_scale)
    outfile = imp.transform(x_scale)
except:
    outfile = x_scale
    y_onehot = y

# 划分训练集和测试集
test_rate = 0.3
x_train, x_test, y_train, y_test = train_test_split(outfile, y_onehot, test_size=test_rate)

# KNeighborsClassifier，K近邻算法
nflod_1 = 5
cv_1 = ShuffleSplit(n_splits=nflod_1, test_size=test_rate, random_state=1)
rf_param_1 = {'n_neighbors': [1, 2, 3, 4, 5]}
my_classifier_1 = GridSearchCV(KNeighborsClassifier(n_neighbors=5,
                                                    weights='uniform', algorithm='auto', leaf_size=30,
                                                    p=2, metric='minkowski', metric_params=None, n_jobs=1),
                               param_grid=rf_param_1, cv=cv_1)

my_classifier_1.fit(x_train, y_train)
predictions_1 = my_classifier_1.predict(x_test)
result_1 = pd.DataFrame()
result_1['p'] = predictions_1
result_1['t'] = y_test
result_1.to_csv("result_1.csv", index=None)
print("结果1保存成功")
# LogisticRegression，逻辑回归算法
nflod_2 = 5
cv_2 = ShuffleSplit(n_splits=nflod_2, test_size=test_rate, random_state=1)
rf_param_2 = {'penalty': ['l1', 'l2']}
my_classifier_2 = GridSearchCV(LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0,
                                                  fit_intercept=True, intercept_scaling=1, class_weight=None,
                                                  random_state=None, solver='liblinear', max_iter=100,
                                                  multi_class='ovr', verbose=0, warm_start=False, n_jobs=1),
                               param_grid=rf_param_2, cv=cv_2)

my_classifier_2.fit(x_train, y_train)
predictions_2 = my_classifier_2.predict(x_test)

result_2 = pd.DataFrame()
result_2['p'] = predictions_2
result_2['t'] = y_test
result_2.to_csv("result_2.csv", index=None)
print("结果2保存成功")

# DecisionTreeClassifier， 决策树算法
nflod_3 = 5
cv_3 = ShuffleSplit(n_splits=nflod_2, test_size=test_rate, random_state=1)
rf_param_3 = {'criterion': ['entropy', 'gini']}
my_classifier_3 = GridSearchCV(DecisionTreeClassifier(criterion="gini",
                                                      splitter="best",
                                                      max_depth=None,
                                                      min_samples_split=2,
                                                      min_samples_leaf=1,
                                                      min_weight_fraction_leaf=0.,
                                                      max_features=None,
                                                      random_state=None,
                                                      max_leaf_nodes=None,
                                                      min_impurity_decrease=0.,
                                                      min_impurity_split=None,
                                                      class_weight=None,
                                                      presort=False),
                               param_grid=rf_param_3, cv=cv_3)

my_classifier_3.fit(x_train, y_train)
predictions_3 = my_classifier_3.predict(x_test)

result_3 = pd.DataFrame()
result_3['p'] = predictions_3
result_3['t'] = y_test
result_3.to_csv("result_3.csv", index=None)
print("结果3保存成功")

# 特征选择和集成学习
my_classifier_4 = AdaBoostClassifier()
rfe = RFE(my_classifier_4, n_features_to_select=3)
rfe.fit(x, y)
predictions_4 = rfe.predict(x_test)

result_4 = pd.DataFrame()
result_4['p'] = predictions_4
result_4['t'] = y_test
result_4.to_csv("result_4.csv", index=None)
print("结果4保存成功")

names = iris["feature_names"]
print("选择的特征是：\n")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
