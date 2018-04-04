#!/usr/bin/python
# -*- coding: utf-8 -*-
from nltk.tokenize import TweetTokenizer
import pandas as pd
import string
from collections import Counter
from nltk.text import Text
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models
from sklearn.cluster import MiniBatchKMeans, KMeans
from tqdm import tqdm
import numpy as np
import os

stemmer = SnowballStemmer("english")

import sys

reload(sys)
sys.setdefaultencoding('utf8')


def nltkcut(inputs):
    tknzr = TweetTokenizer()
    res = tknzr.tokenize(inputs)
    return " ".join(res).encode("utf-8")


def get_traindata():
    data = pd.read_csv("2016_12_05_trumptwitterall.csv", usecols=['tweet'])
    data1 = pd.read_csv("2017_01_28_trump_tweets.csv", usecols=['tweet'])
    df = pd.concat([data, data1])
    df.to_csv("train.csv", index=None)


def get_second_question():
    data = pd.read_csv("train.csv")
    fw = open("train_tokenize.csv", 'w')
    for one in data.values:
        tweets = one[0]
        tweetscut = nltkcut(tweets)
        fw.writelines(tweetscut)


def get_third_question():
    data = pd.read_csv("train.csv")

    save = []
    for one in data.values:
        tweets = one[0]
        tweetscut = nltkcut(tweets)
        for x in tweetscut:
            if x in string.punctuation:
                save.append(x)

    for i, j in Counter(save).most_common(20):
        print(i, j)


def tokenize_and_stem(text):
    # 首先分句，接着分词，而标点也会作为词例存在
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def get_vec(inputs):
    out = []
    for x in inputs:
        vec = x[1] * x[0]
        out.append(vec)
    length = len(out)
    res = out + np.zeros((30 - length,)).tolist()
    return res


def numpy_fillna(data):
    lens = np.array([len(i) for i in data])
    mask = np.arange(30) < lens[:, None]
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


def get_four_question():
    documents = []
    data = pd.read_csv("train.csv")

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

    k_means = KMeans(init='k-means++', n_clusters=10, n_init=10)
    k_means.fit(X)
    clusters = k_means.labels_.tolist()
    datacp = data.copy()
    datacp['label'] = clusters

    datacpgroup = datacp.groupby(by='label')
    for i, j in datacpgroup:
        j.to_csv("./data/{}.csv".format(i), index=None)


def get_four_question_of_a():
    for file in os.listdir("./data"):
        filename = os.path.join('./data', file)
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
    data = pd.read_csv("train.csv")
    totals = []
    for one in data.values:
        tweets = one[0]
        tweetscut = tweets.encode("utf-8").lower().split()
        for x in tweetscut:
            totals.append(x)
    text = Text(totals)
    text.collocations(num=num, window_size=window_size)
    # 把这里的输出结果可以复制到txt，保存即可。
    """
    thank you.; donald trump; make america; thank you!; great again!;
    america great; looking forward; can't wait; crooked hillary; new york;
    celebrity apprentice; hillary clinton; #makeamericagreatagain
    #trump2016; ted cruz; please run; white house; last night; united
    states; global warming; miss universe; golf course; trump tower; thank
    you,; midas touch; think like; interview discussing; great job; trump
    national; – think; mr. trump; look forward; trump international;
    "donald trump; jeb bush; good luck.; president obama; last night.;
    wind turbines; blue monster; happy birthday; country needs; new
    hampshire; “donald trump; mr. trump,; old post; america needs; marco
    rubio; @realdonaldtrump please; tea party; bernie sanders; looks like;
    many people; south carolina; god bless; real estate; post office; las
    vegas; good luck!; think big; donald trump"; golf links; national
    doral; palm beach; never give; mitt romney; failing @nytimes; wall
    street; winston churchill; getting ready; albert einstein; 13th
    season; via @newsmax_media; stay tuned!; work hard; muslim
    brotherhood; joan rivers; good luck; press conference; years ago; golf
    club; goofy elizabeth; 7:00 a.m.; miss usa; mr. trump.; law
    enforcement; jon stewart; trump int'l; poll numbers; much better;
    obamacare website; saudi arabia; henry ford; gas prices; #trump2016
    #makeamericagreatagain; via @breitbartnews; tune in!; vince lombardi;
    tom brady; derek jeter; ronald reagan
"""


if __name__ == "__main__":
    method = 'get_most_freq_words'

    if method == 'get_second_question':
        get_second_question()

    if method == 'get_third_question':
        get_third_question()

    if method == 'get_five_question':
        get_five_question()

    if method == 'get_four_question':
        get_four_question()

    if method == 'get_most_freq_words':
        get_most_freq_words()
