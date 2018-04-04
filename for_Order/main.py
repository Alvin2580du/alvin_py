#!/usr/bin/python
# -*- coding: utf-8 -*-
from nltk.tokenize import TweetTokenizer
import pandas as pd
from collections import Counter
from nltk.text import Text
import re
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import os
from sklearn import metrics
import sys
stemmer = SnowballStemmer("english")

if sys.version_info.major == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')


def nltkcut(inputs):
    tknzr = TweetTokenizer()
    res = tknzr.tokenize(inputs)
    return " ".join(res).encode("utf-8")


def get_traindata():
    # 合并数据集，保存到train.csv
    data = pd.read_csv("./datasets/2016_12_05_trumptwitterall.csv", usecols=['tweet'])
    data1 = pd.read_csv("./datasets/2017_01_28_trump_tweets.csv", usecols=['tweet'])
    df = pd.concat([data, data1])
    df.to_csv("./datasets/tweets_train.csv", index=None)


def get_second_question():
    # 使用TweetTokenizer，分割句子，保存到train_tokenize.csv文件，
    #  第二题
    data = pd.read_csv("./datasets/tweets_train.csv")
    fw = open("train_tokenize.csv", 'w')
    for one in data.values:
        tweets = one[0]
        tweetscut = nltkcut(tweets)
        fw.writelines(tweetscut)


def get_third_question():
    # 提取@后面的昵称，打印出来结果
    # 第三题
    num = 20
    data = pd.read_csv("./datasets/tweets_train.csv")
    save = []
    for one in data.values:
        p = '@\w+:'
        names = re.compile(p).findall(one[0])
        if names:
            for n in names:
                save.append(n.replace(":", ""))
    for i, j in Counter(save).most_common(num):
        print("最多的昵称为：")
        print("昵称: {}, 频率：{}".format(i, j))


def get_vec(inputs):
    out = []
    for x in inputs:
        vec = x[1] * x[0]  # tfidf = tf * idf
        out.append(vec)
    length = len(out)
    res = out + np.zeros((30 - length,)).tolist()
    return res


def get_four_question_and_b():
    # kmeans 聚类, 第四题和第四题的4.b。
    clusters_number = 10
    documents = []
    data = pd.read_csv("./datasets/tweets_train.csv")

    for one in data.values:
        documents.append(one[0])
    texts = [[word for word in document.lower().split()] for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]
    X = []
    for doc in tqdm(corpus_tfidf):
        res = get_vec(doc)
        X.append(res)
    calinski_harabaz_list = []
    for iters in range(1, 5):  # 迭代次数1,2,3,4, 这里可以设置更大的值
        k_means = KMeans(n_clusters=clusters_number, init='k-means++', n_init=10,
                         max_iter=iters, tol=1e-4, precompute_distances='auto',
                         verbose=0, random_state=None, copy_x=True,
                         n_jobs=1, algorithm='auto')
        k_means.fit(X)
        clusters = k_means.labels_.tolist()
        datacp = data.copy()
        datacp['label'] = clusters
        labels = k_means.labels_
        # 类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高,
        # calinski_harabaz_score 计算的是类别间的协方差。
        calinski_harabaz = metrics.calinski_harabaz_score(X, labels)
        ch = {iters: calinski_harabaz}
        calinski_harabaz_list.append(ch)
        datacpgroup = datacp.groupby(by='label')
        for i, j in datacpgroup:
            if not os.path.exists("./data"):
                os.makedirs("./data")
            j.to_csv("./data/{}_{}.csv".format(iters, i), index=None)
    # What was the variance observed on each run? the results is below, saved in calinski_harabaz_list.csv
    df = pd.Series(calinski_harabaz_list)
    df.to_csv("calinski_harabaz_list.csv", index=None)


def get_four_question_of_a():
    # 第四题的  4.a
    for file in os.listdir("./best"):
        filename = os.path.join('./best', file)
        data = pd.read_csv(filename, usecols=['tweet'])
        p = '@\w+:'
        p1 = '#\w+'
        allnames = []
        all_words = []
        hashtags = []
        for one in data.values:
            onesp = one[0].split()
            for x in onesp:
                all_words.append(x)

            names = re.compile(p).findall(one[0])
            if names:
                for n in names:
                    allnames.append(n.replace(":", ""))

            hashtag = re.compile(p1).findall(one[0])
            if hashtag:
                for n in hashtag:
                    hashtags.append(n.replace(":", ""))
        most_words = []
        for i, j in Counter(all_words).most_common(100):
            res = "{},{}".format(i, j)
            most_words.append(res)
        df_words = pd.DataFrame(most_words)
        df_words.to_csv("most_words_{}".format(file), index=None, header=None, sep=',')

        hashtags_list = []
        for i, j in Counter(hashtags).most_common(100):
            res = "{},{}".format(i, j)
            hashtags_list.append(res)
        df_hashtags = pd.DataFrame(hashtags_list)
        df_hashtags.to_csv("df_hashtags_{}".format(file), index=None, header=None, sep=',')

        df = pd.DataFrame(allnames)
        df.to_csv("handles_{}".format(file), index=None, header=None)


def get_five_question(num=100, window_size=2):
    # 第五题
    data = pd.read_csv("./datasets/tweets_train.csv")
    totals = []
    for one in data.values:
        tweets = one[0]
        tweetscut = tweets.encode("utf-8").lower().split()
        for x in tweetscut:
            totals.append(x)
    text = Text(totals)
    text.collocations(num=num, window_size=window_size)
    # 把这里的输出结果可以复制到txt，保存即可。下面是我做的输出


if __name__ == "__main__":
    method = 'get_traindata'  # 改变这里的method的名称，执行下面不同的方法，分别对应的不同的题号。

    if method == 'get_traindata':
        get_traindata()

    if method == 'get_second_question':
        get_second_question()

    if method == 'get_third_question':
        get_third_question()

    if method == 'get_four_question':
        get_four_question_and_b()

    if method == 'get_four_question_of_a':
        get_four_question_of_a()

    if method == 'get_five_question':
        get_five_question()
