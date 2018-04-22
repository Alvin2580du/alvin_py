# encoding:utf-8
import pandas as pd
import os
import xlrd
from tqdm import tqdm
import re
import jieba
import numpy as np
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn import tree
from sklearn.svm import SVC, LinearSVC
import importlib
import sys
import datetime
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


importlib.reload(sys)

"""
F:功能性
NF：非功能性


"""

sw = pd.read_csv("./datasets/stopwords_zh.csv", lineterminator="\n").values.tolist()
sw2list = [j for i in sw for j in i]


def name2class(name='Faceu激萌'):
    path = './datasets/appdataNew'
    res = {}
    for (root, dirs, files) in os.walk(path):
        for file in files:
            fullPath = os.path.join(root, file).split("\\")
            names = str(fullPath[-1]).replace(".csv", "").replace('.xls.csv', "")
            classs = fullPath[-2]
            res[names] = classs
    try:
        return res[name]
    except:
        return "娱乐"


def get_time(name='Faceu激萌', banben='3.0.4'):
    path = "./datasets/appdata"
    for (root, dirs, files) in os.walk(path):
        for dirc in tqdm(files):
            save_name = dirc.replace(".xlsx", "").replace(".xls", "")
            fullPath = os.path.join(root, dirc)
            data = xlrd.open_workbook(fullPath)
            table = data.sheets()[0]
            nrows = table.nrows
            for i in range(nrows):
                if i == 0:
                    continue
                content = table.row_values(i)
                time = content[1]
                id = content[0]
                if save_name == name and id == banben:
                    return time


def cuts(inputs):
    cut = jieba.lcut(inputs)
    save = []
    for x in cut:
        if x in sw2list:
            continue
        save.append(x)
    return " ".join(save)


def replaces(inputs):
    string = re.sub("[\.\!\/_,$%^*；(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", inputs)
    return re.sub("[0-9]", "", string.replace(" ", ""))


def label_data(inputs):
    try:
        if "NF" in inputs:
            return "1"
        elif "O" in inputs:
            return "0"
        elif "F" in inputs:
            return "-1"
        else:
            return "nolabel"
    except:
        return "nolabel"


def make_data(path="./datasets/appdata"):
    save = []
    for (root, dirs, files) in os.walk(path):
        for dirc in tqdm(files):
            fullPath = os.path.join(root, dirc)
            save_name = str(fullPath.split("\\")[-1]).replace(".xlsx.csv", "")
            try:
                data = pd.read_csv(fullPath, encoding='gbk', usecols=['版本', '日期', '更新内容'], sep=',')
            except:
                data = pd.read_csv(fullPath, encoding='gbk', usecols=['版本', '更新', '更新内容'], sep=',')
            for one in data.values:
                reslist = str(one[2]).split("\n")
                for res in reslist:
                    if len("{}".format(res)) < 10:
                        continue
                    if "BUG反馈QQ群" in res:
                        continue
                    if "======" in res:
                        continue
                    if "近期更新" in res:
                        continue
                    if "最新更新" in res:
                        continue
                    if "本次更新" in res:
                        continue
                    rows = {"appname": save_name, "id": one[0], 'date': one[1], "msg": res, 'label': label_data(one[2])}
                    save.append(rows)

    df = pd.DataFrame(save)
    save_file = './datasets/AppTrain.csv'
    df.to_csv(save_file, index=None)


def wordtovec_jueceshu():
    data = pd.read_csv("./datasets/AppTrain.csv")
    data = shuffle(data)
    train_x = data[~data['label'].isin(['nolabel'])]['msg']
    train_y = data[~data['label'].isin(['nolabel'])]['label']
    predicts_data = data[data['label'].isin(['nolabel'])]['msg']
    predicts_data_all = data[data['label'].isin(['nolabel'])]
    train_data_all = data[~data['label'].isin(['nolabel'])]
    del predicts_data_all['label']
    train = train_x.values.tolist()
    print(train)
    labels = train_y.values.tolist()
    print(set(labels))
    tv = TfidfVectorizer()
    fea_train = tv.fit_transform(train)
    print(fea_train.shape)
    fea_test = tv.transform(predicts_data)
    clf = tree.DecisionTreeClassifier()
    clf.fit(fea_train, np.array(labels))
    pred = clf.predict(fea_test)
    df = predicts_data_all.copy()
    df['label'] = pred
    save = pd.concat([df, train_data_all], axis=0)
    save.to_csv("./预测_jueceshu.csv", index=None)


def wordtovec_SVM():
    data = pd.read_csv("./datasets/AppTrain.csv")
    train_x = data[~data['label'].isin(['nolabel'])]['msg']
    train_y = data[~data['label'].isin(['nolabel'])]['label']
    print(train_y)
    predicts_data = data[data['label'].isin(['nolabel'])]['msg']
    predicts_data_all = data[data['label'].isin(['nolabel'])]
    train_data_all = data[~data['label'].isin(['nolabel'])]
    del predicts_data_all['label']
    train = train_x.values.tolist()
    labels = train_y.values.tolist()
    print(set(labels))
    tv = TfidfVectorizer()
    fea_train = tv.fit_transform(train)
    fea_test = tv.transform(predicts_data)
    clf = LinearSVC()
    clf.fit(fea_train, np.array(labels))
    pred = clf.predict(fea_test)
    df = predicts_data_all.copy()
    df['label'] = pred
    save = pd.concat([df, train_data_all], axis=0)
    save.to_csv("./预测_支持向量机.csv", index=None)


def groupbyclass_Juceshu():
    data = pd.read_csv("./预测_jueceshu.csv")
    data['class'] = data['appname'].apply(name2class)
    datagroup = data.groupby(by='class')
    for i, j in datagroup:
        j.to_csv("./datasets/{}_tree.csv".format(i), index=None)


def groupbyclasssvm():
    data = pd.read_csv("./预测_支持向量机.csv")
    data['class'] = data['appname'].apply(name2class)
    datagroup = data.groupby(by='class')
    for i, j in datagroup:
        j.to_csv("./datasets/{}_svm.csv".format(i), index=None)


def get_jidu(inputs='6月6日2014年'):
    try:
        time1 = datetime.datetime.strptime(inputs, "%m月%d日%Y年")
        if 1 <= time1.month <= 3:
            return "{}年第一季度".format(time1.year)

        if 4 <= time1.month <= 6:
            return "{}年第二季度".format(time1.year)

        if 7 <= time1.month <= 9:
            return "{}年第三季度".format(time1.year)

        if 10 <= time1.month <= 12:
            return "{}年第四季度".format(time1.year)
    except:
        return None


def plots(filename='娱乐_tree.csv'):
    data = pd.read_csv("./datasets/{}".format(filename), usecols=['label', 'date'])
    data['jidu'] = data['date'].apply(get_jidu)
    del data['date']
    datagroup = data.groupby(by='jidu')
    res = []
    for x, y in datagroup:
        class_one = len(y[y['label'].isin(['1'])]['label'].values)
        class_zero = len(y[y['label'].isin(['0'])]['label'].values)
        class_neg_one = len(y[y['label'].isin(['-1'])]['label'].values)
        rows = {'季度': x, 'class_one': class_one, 'class_zero': class_zero, 'class_neg_one': class_neg_one}
        res.append(rows)
    df = pd.DataFrame(res)
    df.to_csv("./datasets/plots/{}".format(filename), index=None)


def get_plot_data():
    files = ['娱乐_tree.csv', '生活_tree.csv', '社交_tree.csv', '娱乐_svm.csv', '生活_svm.csv', '社交_svm.csv']
    for file in files:
        plots(file)

if __name__ == "__main__":
    method = "get_plot_data"

    if method == 'make_data':
        print("make_data")
        make_data()

    if method == 'wordtovec_jueceshu':
        print("wordtovec_jueceshu")
        wordtovec_jueceshu()

    if method == 'wordtovec_SVM':
        print("wordtovec_SVM")
        wordtovec_SVM()

    if method == 'groupbyclass_Juceshu':
        print("groupbyclass_Juceshu")
        groupbyclass_Juceshu()

    if method == 'groupbyclasssvm':
        print("groupbyclasssvm")
        groupbyclasssvm()

    if method == 'get_plot_data':
        print("get_plot_data")
        get_plot_data()
