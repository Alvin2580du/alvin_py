import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.externals import joblib


# Age,Attrition,BusinessTravel,Department,DistanceFromHome,Education,EducationField,EnvironmentSatisfaction,Gender,
# JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,MonthlyIncome,NumCompaniesWorked,OverTime,
# PercentSalaryHike,PerformanceRating,RelationshipSatisfaction,StockOptionLevel,TotalWorkingYears,
# TrainingTimesLastYear,WorkLifeBalance,YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager

def trainrf_withsklearn():
    test_rate = 0.5
    random_state = 1
    nflod = 5
    df = pd.read_csv('pfm_train _digitization2.csv')
    df = shuffle(df)
    y = df['Attrition']
    del df['Attrition']
    fea = ['Department', 'BusinessTravel', 'JobLevel', 'Gender', 'WorkLifeBalance', 'PerformanceRating', 'Education',
           'YearsSinceLastPromotion', 'MonthlyIncome', 'MaritalStatus', 'RelationshipSatisfaction', 'JobRole']
    for f in fea:
        del df[f]

    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_rate, random_state=random_state)

    # 训练集标准化
    scale_method = 'z-score'  # z-score 一般来说 效果较好

    if scale_method == 'z-score':
        # 归一化到0均值，1标准差
        scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
        X_train_std = scaler.transform(X_train)

    if scale_method == 'one-zero':
        # 归一化到[0,1]之间
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_std = min_max_scaler.fit_transform(X_train)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_test_std = min_max_scaler.fit_transform(X_test)

    if scale_method == 'oneandneg-one':
        # 归一化到[-1,1]之间
        max_abs_scaler = preprocessing.MaxAbsScaler()
        X_train_std = max_abs_scaler.fit_transform(X_train)
        max_abs_scaler = preprocessing.MaxAbsScaler()
        X_test_std = max_abs_scaler.fit_transform(X_test)

    cv = ShuffleSplit(n_splits=nflod, test_size=test_rate, random_state=random_state)
    rf_param_1 = {'max_features': range(1, X_train.shape[1], 1)}
    rf_param_2 = {'n_estimators': range(1, 200)}
    # rf_param_1 = {'n_estimators': range(1, 100), 'max_depth': range(3, 29, 1), 'max_features': range(3, 15, 1)}
    rf_grid_1 = GridSearchCV(RandomForestClassifier(n_estimators=10,
                                                    criterion="gini",
                                                    max_depth=18,
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
                                                    class_weight=None), param_grid=rf_param_2, cv=cv)
    rf_grid_1.fit(X_train_std, y_train)
    print('Parameter with best score:')
    print(rf_grid_1.best_params_)
    print('Cross validation score:', rf_grid_1.best_score_)
    joblib.dump(rf_grid_1, 'rf.model')


def rftest():
    best_rf = joblib.load("rf.model")
    X_test = pd.read_csv('pfm_test_digitization2.csv')
    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_test)
    X_test_std = scaler.transform(X_test)
    answer = best_rf.predict(X_test_std)

    import csv
    def storFile(data, fileName):
        data = list(map(lambda x: [x], data))
        with open(fileName, 'w', newline='') as f:
            mywrite = csv.writer(f)
            for i in data:
                mywrite.writerow(i)

    storFile(answer, 'pmf_results.csv')
    print(answer)


trainrf_withsklearn()
