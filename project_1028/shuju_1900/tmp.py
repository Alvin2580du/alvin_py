from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def my_confusion_matrix(y_true, y_pred):
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print("混淆矩阵：(left labels: y_true, up labels: y_pred):")
    print("labels\t", labels)
    print(conf_mat)


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


def report_results(rf_classifier, x_train, y_train, x_test, y_test, file_name):
    y_train_ = rf_classifier.predict(x_train)
    acc = accuracy_score(y_train, y_train_)
    print("训练集：")
    print("accuracy:{:0.5f}".format(acc))
    scores = rf_classifier.score(x_train, y_train)
    print("scores:{:0.5f}".format(scores))
    print("= =" * 20)

    y_test_ = rf_classifier.predict(x_test)
    scores = accuracy_score(y_test, y_test_)
    precision = metrics.precision_score(y_test, rf_classifier.predict(x_test), average='macro')  # 精确率
    recall = metrics.recall_score(y_test, rf_classifier.predict(x_test), average='macro')  # 召回率
    f1_scores = metrics.f1_score(y_test, rf_classifier.predict(x_test), average='macro')  # f1 值
    print("测试集：")
    print("scores:{:0.5f}".format(scores))
    print("precision:{:0.5f}".format(precision))
    print("recall:{:0.5f}".format(recall))
    print("f1_scores:{:0.5f}".format(f1_scores))
    y_pred = rf_classifier.predict(x_test)
    my_confusion_matrix(y_test, y_pred)
    plot_roc(y_test, y_pred, file_name='{}.png'.format(file_name))


def str2int(a):
    if a == '配对':
        return 0
    if a == '舞弊':
        return 1


def preprocess():
    df1 = pd.read_excel("trainDataShuffle.xlsx")
    df2 = pd.read_excel("trainDataShuffle.xlsx")
    df = pd.concat([df1, df2])
    print(df.shape)
    del df['年份']
    del df['代码']
    del df['名称']

    # 标准化
    rate_v = df[['董事中外部董事比例', '内部人控制制度', '董事会会议频度', '管理层持股比例', '高管薪酬（对数）', '监事会规模',
                 '监事会持股比例', '监事会会议频度', '股东大会会议次数', '国有控股比例', '第一大股东持股比例', '股权集中度',
                 'Z指数', '关联交易影响程度', '净资产收益率', '资产收益率', '销售净利率', '流动比率', '速动比率', '利息保障倍数',
                 '资产负债率', '财务杠杆', '应收账款周转率', '存货周转率', '固定资产周转率', '总资产周转率', '固定资产/营业收入',
                 '营业收入增长比', '总资产增长率', '利润总额增长率', '企业自由现金流量', '每股经营活动产生的现金流量净额',
                 '经营活动产生的现金流量净额／流动负债']]
    rate_v = (rate_v - rate_v.mean()) / (rate_v.std())  # 标准化

    # 处理哑变量
    Dummy_v_list = ['董事长兼总经理（两职兼任）', '董事长变更', '是否设立审计委员会', '是否国有控股', '审计意见',
                    '会计师事务所更换', '大规模事务所', '避免ST或退市ST']
    hh = []
    for i in Dummy_v_list:
        result = pd.get_dummies(df[i], prefix=i)
        hh.append(result)
    Dummy_v = pd.concat(hh, axis=1)

    # 合并
    _X = pd.concat([rate_v, Dummy_v], axis=1)
    _X.to_excel("处理后的数据.xlsx", index=None)
    # 标签01化
    _y = df['标签']
    _y = pd.Series(map(str2int, _y))
    _y.to_excel("处理后的标签.xlsx", index=None)
    test_rate = 0.3
    x_train, x_test, y_train, y_test = train_test_split(_X, _y, test_size=test_rate, random_state=0)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test


def other(num):
    x_train, x_test, y_train, y_test = preprocess()
    rf_classifier = RandomForestClassifier(criterion='entropy', n_estimators=200,
                                           min_samples_leaf=1,
                                           max_depth=10, random_state=0)
    rf_classifier.fit(x_train.values, y_train)
    names = x_train.columns.tolist()
    results = sorted(zip(map(lambda x: round(x, 4), rf_classifier.feature_importances_), names), reverse=True)
    print("选择的特征是：\n")
    save = []
    for x in results[:num]:
        save.append(x[1])
    print(save)
    new_train = x_train[save]
    new_test = x_test[save]
    print(new_train.shape, new_test.shape)
    logis_classifier = LogisticRegression()
    logis_classifier.fit(new_train, y_train)
    report_results(logis_classifier, new_train, y_train, new_test, y_test, file_name='特征选择-逻辑回归')

    detree_classifier = DecisionTreeClassifier(max_depth=3)
    detree_classifier.fit(new_train, y_train)
    report_results(detree_classifier, new_train, y_train, new_test, y_test, file_name='特征选择-决策树')


if __name__ == "__main__":
    method = 'other'
    if method == 'other':
        other(num=17)
