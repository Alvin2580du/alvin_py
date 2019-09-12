import pandas as pd
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

from sklearn.model_selection import train_test_split
from sklearn import metrics
import os

warnings.filterwarnings("ignore")

"""
题目三文件中，提供的5487 条样本数据，特征变量：x1-x1763。其中gid是主键，apply_time 是申请日期，y 是好坏人表示（1 是坏人，0 是好人）。
1.统计样本中各个特征变量的最大值、最小值、均值、分位数。可以用一个函数实现。
2. 统计每个申请月的bad 人数、good 人数、总人数、bad_rate。
3. 对数值型数据，进行等频、等宽分箱。并统计每个变量分箱后每个箱子里的bad 人数、good 人数、总人数、bad 人数占比，
good 人数占比，累计bad 人数占比，累计good 人数占比，总人数占比，bad_rate、woe、KS、IV。
4.划分训练测试集建立风险预测模型以便于识别好坏人，给出最终建立模型训练集和测试集的KS、AUC。建模方法（评分卡、XGBoost、神经网络）。

"""
time_features = "gid, apply_time, x3, x4, x8, x9, x13, x14, x18, x19, x23, x24, x28, x29, x33, x34, x38, x39, x43, " \
                "x44, x48, x49, " \
                "x53, x54, x58, x59, x63, x64, x68, x69, x73, x74, x78, x79, x83, x84, x88, x89, x93, x94, x98, " \
                "x99, x103, x104, x108, x109, x113, x114, x118, x119, x123, x124, x128, x129, x133, x134, x138, " \
                "x139, x143, x144, x148, x149, x153, x154, x158, x159, x163, x164, x168, x169, x173, x174, x178, " \
                "x179, x183, x184, x188, x189, x193, x194, x198, x199, x203, x204, x208, x209, x213, x214, x217, " \
                "x218, x221, x222, x225, x226, x229, x230, x233, x234, x237, x238, x241, x242, x245, x246, x249, " \
                "x250, x253, x254, x257, x258, x261, x262, x265, x266, x269, x270, x273, x274, x277, x278, x281, " \
                "x282, x285, x286, x289, x290, x293, x294, x297, x298, x301, x302, x305, x306, x309, x310, x313, " \
                "x314, x317, x318, x321, x322, x325, x326, x329, x330, x333, x334, x337, x338, x341, x342, x345, " \
                "x346, x349, x350, x353, x354, x358, x359, x363, x364, x368, x369, x373, x374, x378, x379, x383, " \
                "x384, x387, x388, x391, x392, x395, x396, x399, x400, x403, x404, x408, x409, x413, x414, x418, " \
                "x419, x423, x424, x428, x429, x433, x434, x437, x438, x441, x442, x445, x446, x449, x450 "

time_features = [i.strip() for i in time_features.split(', ')]
print("非数值型变量个数", len(time_features))


def question_one():
    data = pd.read_excel("题目三.xlsx")

    save = []

    for col in data.columns:
        if 'gid' in col:
            continue
        if 'apply_time' in col:
            continue
        if 'y' in col:
            continue
        rows = OrderedDict()

        label = data['{}'.format(col)].dropna().values
        max_ = np.max(label)
        min_ = np.min(label)
        try:
            mean_ = np.mean(label)
            percentiles = np.array([2.5, 25, 50, 75, 97.5])
            percentile_ = np.percentile(label, q=percentiles)
        except:
            mean_ = ''
            percentile_ = ''

        rows['max'] = max_
        rows['min'] = min_
        rows['mean'] = mean_
        rows['percentile'] = percentile_
        save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("统计.xlsx", index=None)
    print(df.shape)


def get_month(inputs):
    return inputs[:7]


def build_q2():
    data = pd.read_excel("题目三.xlsx")

    data['month'] = data['apply_time'].apply(get_month)
    save = []

    for x, y in data.groupby(by='month'):
        print(x)
        bad = [i for i in y['y'] if i == 0]
        good = [i for i in y['y'] if i == 1]
        total = y.shape[0]
        bad_rate = len(bad) / total
        rows = OrderedDict()
        rows['bad'] = len(bad)
        rows['good'] = len(good)
        rows['totoal'] = total
        rows['bad_rate'] = bad_rate
        rows['month'] = x
        save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("q2_results.xlsx", index=None)
    print(df.shape)


#
# 3. 对数值型数据，进行等频、等宽分箱。并统计每个变量分箱后每个箱子里的bad 人数、good 人数、总人数、bad 人数占比，
# good 人数占比，累计bad 人数占比，累计good 人数占比，总人数占比，bad_rate、woe、KS、IV。


def bin_frequency(x, y, n=5, col='x1'):  # x为待分箱的变量，y为target变量.n为分箱数量
    total = y.count()  # 计算总样本数
    bad = y.sum()  # 计算坏样本数
    good = y.count() - y.sum()  # 计算好样本数
    d1 = pd.DataFrame({'x': x, 'y': y, 'bucket': pd.qcut(x, n, duplicates='drop')})  # 用pd.cut实现等频分箱
    d2 = d1.groupby('bucket', as_index=True)  # 按照分箱结果进行分组聚合
    d3 = pd.DataFrame(d2.x.min(), columns=['min_bin'])
    d3['min_bin'] = d2.x.min()  # 箱体的左边界
    d3['max_bin'] = d2.x.max()  # 箱体的右边界
    d3['bad'] = d2.y.sum()  # 每个箱体中坏样本的数量
    d3['good'] = d2.y.count() - d2.y.sum()  # 每个箱体中坏样本的数量
    d3['total'] = d2.y.count()  # 每个箱体的总样本数
    d3['bad_rate'] = d3['bad'] / d3['total']  # 每个箱体中坏样本所占总样本数的比例
    d3['good_rate'] = (d2.y.count() - d2.y.sum()) / d3['total']  # 每个箱体中好样本所占总样本数的比例
    d3['bad_rate_cum'] = d3['bad_rate'].cumsum()
    d3['good_rate_cum'] = d3['good_rate'].cumsum()
    d3['badattr'] = d3['bad'] / bad  # 每个箱体中坏样本所占坏样本总数的比例
    d3['goodattr'] = (d3['total'] - d3['bad']) / good  # 每个箱体中好样本所占好样本总数的比例
    d3['woe'] = np.log(d3['goodattr'] / d3['badattr'])  # 计算每个箱体的woe值
    d3['iv'] = (d3['goodattr'] - d3['badattr']) * d3['woe']  # 计算变量的iv值
    d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True)  # 对箱体从大到小进行排序
    if not os.path.exists('./bin_frequency'):
        os.makedirs('./bin_frequency')

    d4.to_excel("./bin_frequency/{}.xlsx".format(col), index=None)


def build_freq():
    data = pd.read_excel('题目三.xlsx')
    for col in tqdm(data.columns):
        if col in time_features:
            continue
        try:
            df = pd.DataFrame()
            df['y'] = data['y']
            df[col] = data[col]
            df = df.drop_duplicates()
            bin_frequency(df[col], y=df['y'], col=col)
        except:
            print('col 无法等频分箱', col)


def bin_distince(x, y, n=10, col='x1'):  # x为待分箱的变量，y为target变量.n为分箱数量
    total = y.count()  # 计算总样本数
    bad = y.sum()  # 计算坏样本数
    good = y.count() - y.sum()  # 计算好样本数
    d1 = pd.DataFrame({'x': x, 'y': y, 'bucket': pd.cut(x, n)})  # 利用pd.cut实现等距分箱
    d2 = d1.groupby('bucket', as_index=True)  # 按照分箱结果进行分组聚合
    d3 = pd.DataFrame(d2.x.min(), columns=['min_bin'])
    d3['min_bin'] = d2.x.min()  # 箱体的左边界
    d3['max_bin'] = d2.x.max()  # 箱体的右边界
    d3['bad'] = d2.y.sum()  # 每个箱体中坏样本的数量
    d3['good'] = d2.y.count() - d2.y.sum()  # 每个箱体中坏样本的数量
    d3['total'] = d2.y.count()  # 每个箱体的总样本数
    d3['bad_rate'] = d3['bad'] / d3['total']  # 每个箱体中坏样本所占总样本数的比例
    d3['good_rate'] = (d2.y.count() - d2.y.sum()) / d3['total']  # 每个箱体中好样本所占总样本数的比例
    d3['bad_rate_cum'] = d3['bad_rate'].cumsum()
    d3['good_rate_cum'] = d3['good_rate'].cumsum()
    d3['badattr'] = d3['bad'] / bad  # 每个箱体中坏样本所占坏样本总数的比例
    d3['goodattr'] = (d3['total'] - d3['bad']) / good  # 每个箱体中好样本所占好样本总数的比例
    d3['woe'] = np.log(d3['goodattr'] / d3['badattr'])  # 计算每个箱体的woe值
    d3['iv'] = (d3['goodattr'] - d3['badattr']) * d3['woe']  # 计算变量的iv值
    d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True)  # 对箱体从大到小进行排序
    if not os.path.exists('./bin_distince'):
        os.makedirs('./bin_distince')
    d4.to_excel("./bin_distince/{}.xlsx".format(col), index=None)


def build_dis():
    data = pd.read_excel('题目三.xlsx')
    for col in tqdm(data.columns):
        if col in time_features:
            continue
        df = pd.DataFrame()
        df['y'] = data['y']
        df[col] = data[col]
        df = df.drop_duplicates()
        bin_distince(df[col], y=df['y'], n=5, col=col)


# 4.划分训练测试集建立风险预测模型以便于识别好坏人，给出最终建立模型训练集和测试集的KS、AUC。建模方法（评分卡、XGBoost、神经网络）。
def build_xgboost_mlp():
    import os
    # os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/graphviz-2.38/release/bin/'

    data = pd.read_excel('题目三.xlsx')
    data = data.fillna(-1)
    label = data['y']
    del data['y']
    for tf in time_features:
        del data[tf]

    X_tr, X_te, y_tr, y_te = train_test_split(data, label, test_size=0.3)
    print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)

    model = XGBClassifier(learning_rate=0.1,
                          n_estimators=1000,  # 树的个数--1000棵树建立xgboost
                          max_depth=6,  # 树的深度
                          min_child_weight=1,  # 叶子节点最小权重
                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                          subsample=0.8,  # 随机选择80%样本建立决策树
                          colsample_btree=0.8,  # 随机选择80%特征建立决策树
                          objective='multi:softmax',  # 指定损失函数
                          scale_pos_weight=1,  # 解决样本个数不平衡的问题
                          random_state=27,  # 随机数
                          num_class=2
                          )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="mlogloss", early_stopping_rounds=10, verbose=True)
    ### plot feature importance
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_importance(model, height=0.5, ax=ax, max_num_features=64)
    plt.show()
    y_pred = model.predict(X_te)

    ### model evaluate
    accuracy = metrics.accuracy_score(y_te, y_pred)
    print("xgboost accuarcy: %.2f%%" % (accuracy * 100.0))

    auc_roc = metrics.roc_auc_score(y_te, y_pred)
    print('XGboost AUC', auc_roc)

    fpr, tpr, thresholds = metrics.roc_curve(np.array(y_te), y_pred)
    print('XGboost KS:', max(tpr - fpr))

    # xgb.plot_tree(bst, num_trees=2)
    # plt.show()

    print()
    print()

    print('---------------开始训练神经网络---------------')
    clf = MLPClassifier(solver='sgd', activation='identity', max_iter=10,
                        alpha=1e-5, hidden_layer_sizes=(100, 50),
                        random_state=1, verbose=True)
    clf.fit(X_tr, y_tr)
    # 预测
    y_pred = clf.predict(X_te)
    ### model evaluate
    accuracy = metrics.accuracy_score(y_te, y_pred)
    print("NN accuarcy: %.2f%%" % (accuracy * 100.0))

    auc_roc = metrics.roc_auc_score(y_te, y_pred)
    print('NN AUC', auc_roc)

    y_predict_proba = clf.predict_proba(X_te)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y_te), y_predict_proba)
    print('NN KS:', max(tpr - fpr))


if __name__ == '__main__':
    method = 'build_xgboost_mlp'

    if method == 'question_one':
        question_one()

    if method == 'build_q2':
        build_q2()

    if method == 'build_freq':
        build_freq()

    if method == 'build_dis':
        build_dis()

    if method == 'build_xgboost_mlp':
        build_xgboost_mlp()
