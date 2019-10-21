import pandas as pd
import os
import math
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter, OrderedDict
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

stoplist = stopwords.words('english')

"""
C.最后设计代码，提取自建的致股东信小型语料库词表中的候选词，将得到的候选词与基础情感词典逐个扫描，判断是否为已有情感词，
如果属于已有情感词则结束，否则，将候选词与预先选定的基础词典做SO-PMI计算，获取情感倾向点互信息值，
判断候选词是否应该归入基础情感词典，否则，应舍弃。

基于评价系统，对已经建立的情感词典从-5—+5进行分类和赋值

④利用python对赋值的情感词进行情感极性计算。

⑤利用机器学习的SVM（支持向量机）技术、NB（朴素贝叶斯）、K-means等方法对相同的

PMI, 是互信息(NMI)中的一种特例, 而互信息,是源于信息论中的一个概念,主要用于衡量2个信号的关联程度.至于PMI,是在文本处理中,
用于计算两个词语之间的关联程度.比起传统的相似度计算, pmi的好处在于,从统计的角度发现词语共现的情况来分析出词语间是否存在语义相关 ,
 或者主题相关的情况.
 
基本思想是：选用一组褒义词（Pwords）跟一组贬义词（Nwords）作为基准词。若把一个词语word1跟Pwords的点间互信息减
去word1跟Nwords的点间互信息会得到一个差值，就可以根据该差值判断词语word1的情感倾向。

"""


def get_data():
    save = []

    for file in os.listdir("./China08"):
        file_name = os.path.join('./China08', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                rows = {}
                rows['content'] = line
                rows['country'] = 'China'
                rows['label'] = '08'
                save.append(rows)

    for file in os.listdir("./China09"):
        file_name = os.path.join('./China09', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                rows = {}
                rows['content'] = line
                rows['country'] = 'China'
                rows['label'] = '09'
                save.append(rows)

    for file in os.listdir("./China10"):
        file_name = os.path.join('./China10', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                rows = {}
                rows['content'] = line
                rows['country'] = 'China'
                rows['label'] = '09'
                save.append(rows)

    for file in os.listdir("./China18"):
        file_name = os.path.join('./China18', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                rows = {}
                rows['content'] = line
                rows['country'] = 'China'
                rows['label'] = '18'
                save.append(rows)

    for file in os.listdir("./USA08"):
        file_name = os.path.join('./USA08', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                rows = {}
                rows['content'] = line
                rows['country'] = 'USA'
                rows['label'] = '08'
                save.append(rows)

    for file in os.listdir("./USA09"):
        file_name = os.path.join('./USA09', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                rows = {}
                rows['content'] = line
                rows['country'] = 'USA'
                rows['label'] = '09'
                save.append(rows)
    for file in os.listdir("./USA10"):
        file_name = os.path.join('./USA10', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                rows = {}
                rows['content'] = line
                rows['country'] = 'USA'
                rows['label'] = '10'
                save.append(rows)

    for file in os.listdir("./USA18"):
        file_name = os.path.join('./USA18', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                rows = {}
                rows['content'] = line
                rows['country'] = 'USA'
                rows['label'] = '18'
                save.append(rows)
    df = pd.DataFrame(save)
    df.to_excel("data.xlsx", index=None)
    print(df.shape)


def get_data_docs():
    save = []

    for file in os.listdir("./China08"):
        file_name = os.path.join('./China08', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            rows = {}
            rows['content'] = " ".join(lines)
            rows['country'] = 'China'
            rows['label'] = '08'
            save.append(rows)

    for file in os.listdir("./China09"):
        file_name = os.path.join('./China09', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            rows = {}
            rows['content'] = " ".join(lines)
            rows['country'] = 'China'
            rows['label'] = '09'
            save.append(rows)

    for file in os.listdir("./China10"):
        file_name = os.path.join('./China10', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            rows = {}
            rows['content'] = " ".join(lines)
            rows['country'] = 'China'
            rows['label'] = '10'
            save.append(rows)

    for file in os.listdir("./China18"):
        file_name = os.path.join('./China18', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            rows = {}
            rows['content'] = " ".join(lines)
            rows['country'] = 'China'
            rows['label'] = '18'
            save.append(rows)
    for file in os.listdir("./USA08"):
        file_name = os.path.join('./USA08', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            rows = {}
            rows['content'] = " ".join(lines)
            rows['country'] = 'USA'
            rows['label'] = '08'
            save.append(rows)

    for file in os.listdir("./USA09"):
        file_name = os.path.join('./USA09', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            rows = {}
            rows['content'] = " ".join(lines)
            rows['country'] = 'USA'
            rows['label'] = '09'
            save.append(rows)

    for file in os.listdir("./USA10"):
        file_name = os.path.join('./USA10', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            rows = {}
            rows['content'] = " ".join(lines)
            rows['country'] = 'USA'
            rows['label'] = '10'
            save.append(rows)

    for file in os.listdir("./USA18"):
        file_name = os.path.join('./USA18', file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            rows = {}
            rows['content'] = " ".join(lines)
            rows['country'] = 'USA'
            rows['label'] = '18'
            save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("data_docs.xlsx", index=None)


def get_seed():
    seed_dict = pd.read_excel("seed dictionary.xlsx")

    pos_dict = []
    neg_dict = []

    for seeds, y in seed_dict.iterrows():
        words = y['word']
        value = y['value']
        if value < 0:
            neg_dict.append(words)
        else:
            pos_dict.append(words)

    return pos_dict, neg_dict


def build_pmi():
    data = pd.read_excel("data_docs.xlsx")
    n_docs = data.shape[0]

    for country, y0 in data.groupby(by='country'):
        stop_single = []

        for content, y in y0.iterrows():
            cont = y['content'].split()
            for i in cont:
                if i not in stoplist:
                    stop_single.append(i)

        count_stop_single = {}
        for x, y in Counter(stop_single).most_common(100000):
            tags = pos_tag(word_tokenize(x))[0][1]
            if tags == 'RB' or tags == "VB" or tags == "JJ":
                count_stop_single[x.replace("”", "").replace(",", "")] = y

        pos_dict, neg_dict = get_seed()
        print(len(pos_dict), len(neg_dict))

        com = defaultdict(lambda: defaultdict(int))

        for content, y in data.iterrows():
            terms_only = y['content'].split()
            for i in range(len(terms_only) - 1):
                for j in range(i + 1, len(terms_only)):
                    w1, w2 = sorted([terms_only[i], terms_only[j]])
                    if w1 != w2:
                        com[w1][w2] += 1

        p_t = {}
        p_t_com = defaultdict(lambda: defaultdict(int))

        for term, n in count_stop_single.items():
            if n > 5:
                p_t[term] = n / n_docs
                for t2 in com[term]:
                    p_t_com[term][t2] = com[term][t2] / n_docs

        pmi = defaultdict(lambda: defaultdict(int))
        for t1 in p_t:
            for t2 in com[t1]:
                try:
                    denom = p_t[t1] * p_t[t2]
                    pmi[t1][t2] = math.log2(p_t_com[t1][t2] / denom)
                except:
                    continue

        SO_save = []
        for term, n in p_t.items():
            semantic_orientation = {}
            positive_assoc = sum(pmi[term][tx] for tx in pos_dict)
            negative_assoc = sum(pmi[term][tx] for tx in neg_dict)
            semantic_orientation["term"] = term
            semantic_orientation["so_pmi"] = positive_assoc - negative_assoc
            semantic_orientation["pos"] = positive_assoc
            semantic_orientation["neg"] = negative_assoc
            SO_save.append(semantic_orientation)

        df = pd.DataFrame(SO_save).sort_values(by='so_pmi')
        df.to_excel("semantic_orientation-{}.xlsx".format(country))
        print(df.shape)


def get_senti(inputs):
    res = 0
    cates = []
    for i in inputs:
        if i in pos_dict or neg_dict:
            try:
                cate = dict_[dict_['word'].isin([i])]['main category'].values.tolist()[0]
                scores = dict_[dict_['word'].isin([i])]['value'].values.tolist()[0]
                res += scores
                cates.append(cate)
            except:
                continue
    rows = {}
    for name, times in Counter(cates).most_common(10):
        rows[name] = times
    if res > 0:
        return '积极', rows
    elif res == 0:
        return '中性', rows
    else:
        return '消极', rows


def build_senti(data_name='data.xlsx'):
    data = pd.read_excel(data_name)
    save = []
    for c, country in data.groupby(by='country'):
        for docs, y in country.iterrows():
            content = y['content'].split()
            res, cates = get_senti(content)
            rows = OrderedDict()
            rows['content'] = y['content']
            rows['senti'] = res
            rows['country'] = c
            rows['label'] = y['label']
            if "attitude" in cates.keys():
                rows['attitude'] = cates['attitude']
            else:
                rows['attitude'] = 0

            if "engagement" in cates.keys():
                rows['engagement'] = cates['engagement']
            else:
                rows['engagement'] = 0

            if "graduation" in cates.keys():
                rows['graduation'] = cates['graduation']
            else:
                rows['graduation'] = 0

            save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("情感分析结果.xlsx", index=None)
    print(df.shape)


def trans(inputs):
    if inputs == '积极':
        return 0
    elif inputs == '中性':
        return 1
    else:
        return 2


def get_roc(test_y, y_score, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc


if __name__ == '__main__':

    method = 'machineLearning'

    if method == 'get_data':
        # 合并数据，每一行一个样本
        get_data()

    if method == 'get_data_docs':
        # 合并数据，每个文档一个样本
        get_data_docs()

    if method == 'build_pmi':
        # 计算PMI，得到候选词
        build_pmi()

    if method == 'senti':
        # 基于词典的情感分析
        dict_ = pd.read_excel("seed dictionary3.xlsx")
        pos_dict = dict_[dict_['value'] > 0]['word'].values.tolist()
        neg_dict = dict_[dict_['value'] < 0]['word'].values.tolist()
        build_senti()

    if method == 'machineLearning':
        # 基于机器学习的情感分析
        data = pd.read_excel("情感分析结果.xlsx")
        data = shuffle(data)
        data['senti2vec'] = data['senti'].apply(trans)
        n_classes = 3

        for country, lines in data.groupby(by='country'):
            X_, y = lines['content'].values, lines['senti2vec'].values
            x_train,  x_test, y_train, y_test = train_test_split(X_, y)
            print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

            con = CountVectorizer(binary=True)
            train_x = con.fit_transform(x_train)
            test_x = con.transform(x_test)
            train_y = label_binarize(y_train, classes=[0, 1, 2])
            test_y = label_binarize(y_test, classes=[0, 1, 2])

            # 朴素贝叶斯
            print('- -'*10)
            print('{}-朴素贝叶斯'.format(country))
            nb_clf = OneVsRestClassifier(GaussianNB())
            nb_clf.fit(train_x.toarray(), train_y)
            nb_score = nb_clf.predict(test_x.toarray())
            nb_roc_auc = get_roc(test_y, nb_score, n_classes)
            print(nb_roc_auc)
            reports_nb = classification_report(test_y, nb_score, labels=[0, 1, 2])
            print(reports_nb)
            print('- -'*10)

            # 支持向量机
            print('{}-支持向量机'.format(country))
            svm_clf = OneVsRestClassifier(LinearSVC(random_state=0))
            svm_score = svm_clf.fit(train_x, train_y).decision_function(test_x)
            svm_roc_auc = get_roc(test_y, svm_score, n_classes)
            print(svm_roc_auc)
            y_pred = svm_clf.predict(test_x)
            reports_svm = classification_report(test_y, y_pred, labels=[0, 1, 2])
            print(reports_svm)
            print('- -'*10)
            # 逻辑回归
            print('{}-逻辑回归'.format(country))
            lr_clf = OneVsRestClassifier(LogisticRegression(random_state=0))
            lr_score = lr_clf.fit(train_x, train_y).decision_function(test_x)
            lr_roc_auc = get_roc(test_y, lr_score, n_classes)
            print(lr_roc_auc)
            y_pred = lr_clf.predict(test_x)
            reports_lr = classification_report(test_y, y_pred, labels=[0, 1, 2])
            print(reports_lr)
            print('- -'*10)

