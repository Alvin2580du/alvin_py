from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn import metrics


import matplotlib.pyplot as plt
""" 需求

    Objective 1: 
    Develop two predictive models using all 561 features. 
    For each model, report: 
    1）Accuracy
    2）Confusion matrix
    3）Comment on the differences in accuracy between models (if applicable)
    
    Objective 2:
    For one of the chosen models
    (a) Vary the number of features used in the prediction (e.g. from 100 to 561), 
    and compute the resulting accuracy.
    (b) Determine the number of feature required to obtain 8-%, 90% accuracy.


"""


def convert(inputs):
    d = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING"
    }
    d1 = {v: k for k, v in d.items()}
    res = d1[inputs]
    return res


def data_loader():
    train = pd.read_csv("./data/DataSet_HAR.csv")
    del train['subject']
    train['Activity_label'] = train["Activity"].apply(convert)  #
    train_y = train['Activity_label']
    del train['Activity']
    del train['Activity_label']
    test_x = pd.read_csv("./data/X_test.txt", sep=' ', header=None)
    del test_x[0]
    test_x.columns = train.columns.tolist()
    test_y = pd.read_csv("./data/y_test.txt", sep=' ', header=None)
    test_y.columns = ['Activity']
    test_x = test_x.fillna(axis=1, method='ffill')  # 填充缺失值
    return train, test_x, train_y, test_y['Activity']


def my_confusion_matrix(y_true, y_pred):
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print("混淆矩阵：(left labels: y_true, up labels: y_pred):")
    print("labels\t", labels)
    print(conf_mat)


def chage2class(inputs):
    if inputs < 1:
        return 0
    elif inputs < 2:
        return 1
    elif inputs < 3:
        return 2
    elif inputs < 4:
        return 3
    elif inputs < 5:
        return 4
    elif inputs < 6:
        return 5
    else:
        return 6


def build_one():
    x_train, x_test, y_train, y_test = data_loader()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # random forest
    moedl_rf = RandomForestClassifier(criterion='entropy', n_estimators=200,
                                      min_samples_leaf=1, max_depth=10, random_state=0)
    moedl_rf.fit(x_train, y_train)
    y_pred = moedl_rf.predict(x_test)
    rf_scores = accuracy_score(y_test, y_pred)
    print("rf_scores:{}".format(rf_scores))
    my_confusion_matrix(y_test, y_pred)
    # lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(x_train, y_train)
    y_pred = lasso.predict(x_test)
    y_preds = []
    for one in y_pred:
        y_preds.append(chage2class(one))
    scores_lasso = accuracy_score(y_test, y_preds)
    print("scores_lasso:{}".format(scores_lasso))
    my_confusion_matrix(y_test, y_preds)


def plot_roc(y_true, y_pred, file_name):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(file_name)


def build_two():
    # 选择随机森林作为选择特征的模型
    x_train, x_test, y_train, y_test = data_loader()
    rf_classifier = RandomForestClassifier(criterion='entropy', n_estimators=200,
                                           min_samples_leaf=1, max_depth=10, random_state=0)
    rf_classifier.fit(x_train, y_train)
    names = x_train.columns.tolist()
    results = sorted(zip(map(lambda x: round(x, 4), rf_classifier.feature_importances_), names), reverse=True)
    res = []
    for score in results:
        rows = {"name": score[1], "score": score[0]}
        res.append(rows)

    for num in range(100, 561):
        save = []
        for x in results[:num]:
            save.append(x[1])
        new_train = x_train[save]
        new_test = x_test[save]
        moedl_rf = RandomForestClassifier(criterion='entropy', n_estimators=200,
                                          min_samples_leaf=1, max_depth=10, random_state=0)
        moedl_rf.fit(new_train, y_train)
        y_pred = moedl_rf.predict(new_test)
        rf_scores = accuracy_score(y_test, y_pred)
        plot_roc(y_pred=y_pred, y_true=y_test, file_name='{}.png'.format(num))

        print("number of features:{}, rf_scores:{}".format(num, rf_scores))
        my_confusion_matrix(y_test, y_pred)
        print("- * -" * 20)


if __name__ == '__main__':
    method = 'two'

    if method == 'one':
        build_one()

    if method == 'two':
        build_two()

