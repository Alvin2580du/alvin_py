import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora, models


def xlsx2csv():
    data = pd.read_excel("mainbody_cut.xlsx")
    data.to_csv('mainbody_cut.csv', sep=' ', index=None)


def cluster():
    # 聚类分析
    data = pd.read_excel("mainbody_cut.xlsx")
    corpus = []
    limit = 10
    num = 0
    for one in data['mainbody_cut'].values:
        num += 1
        if num > limit:
            break
        corpus.append(one)

    countvec = CountVectorizer(min_df=5)
    resmtx = countvec.fit_transform(corpus)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(resmtx)
    clf = KMeans(n_clusters=5)
    clf.fit(tfidf)
    labels = clf.labels_
    df = data.head(limit)
    df['class'] = labels
    df.to_excel("聚类分析结果.xlsx", index=None)


def classifyWords(wordList):
    # (1) 情感词
    with open("BosonNLP_sentiment_score.txt", 'r', encoding='utf8') as f:
        senList = f.readlines()
    senDict = defaultdict()
    for s in senList:
        try:
            senDict[s.split(' ')[0]] = s.split(' ')[1].strip('\n')
        except Exception:
            pass
    # (2) 否定词
    with open("notDict.txt", 'r', encoding='utf-8') as f:
        notList = f.read().splitlines()
    # (3) 程度副词
    with open("degreeDict.txt", 'r', encoding='utf-8') as f:
        degreeList = f.read().splitlines()
    degreeDict = defaultdict()
    for index, d in enumerate(degreeList):
        if 3 <= index <= 71:
            degreeDict[d] = 2
        elif 74 <= index <= 115:
            degreeDict[d] = 1.25
        elif 118 <= index <= 154:
            degreeDict[d] = 1.2
        elif 157 <= index <= 185:
            degreeDict[d] = 0.8
        elif 188 <= index <= 199:
            degreeDict[d] = 0.5
        elif 202 <= index <= 231:
            degreeDict[d] = 1.5
        else:
            pass

    senWord = defaultdict()
    notWord = defaultdict()
    degreeWord = defaultdict()

    for index, word in enumerate(wordList):
        if word in senDict.keys() and word not in notList and word not in degreeDict.keys():
            senWord[index] = senDict[word]
        elif word in notList and word not in degreeDict.keys():
            notWord[index] = -1
        elif word in degreeDict.keys():
            degreeWord[index] = degreeDict[word]
    return senWord, notWord, degreeWord


def scoreSent(senWord, notWord, degreeWord, segResult):
    W = 1
    score = 0
    # 存所有情感词的位置的列表
    senLoc = list(senWord.keys())
    notLoc = list(notWord.keys())
    degreeLoc = list(degreeWord.keys())
    senloc = -1
    # notloc = -1
    # degreeloc = -1

    # 遍历句中所有单词segResult，i为单词绝对位置
    for i in range(0, len(segResult)):
        # 如果该词为情感词
        if i in senLoc:
            # loc为情感词位置列表的序号
            senloc += 1
            # 直接添加该情感词分数
            score += W * float(senWord[i])
            # print "score = %f" % score
            if senloc < len(senLoc) - 1:
                # 判断该情感词与下一情感词之间是否有否定词或程度副词
                # j为绝对位置
                for j in range(senLoc[senloc], senLoc[senloc + 1]):
                    # 如果有否定词
                    if j in notLoc:
                        W *= -1
                    # 如果有程度副词
                    elif j in degreeLoc:
                        W *= float(degreeWord[j])
        # i定位至下一个情感词
        if senloc < len(senLoc) - 1:
            i = senLoc[senloc + 1]
    return score


def get_wordDicts(filename):
    words = {}
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        num = 0
        for line in lines:
            newSent = line.strip().replace("\n", "").split()
            for x in newSent:
                if x not in words.keys():
                    words[x] = num
                    num += 1
    return words


def build_sentments(filename):
    # 情感分析
    wordsDicts = get_wordDicts(filename)
    senWord, notWord, degreeWord = classifyWords(wordsDicts)
    save = []
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        limit = 10
        num = 0
        for line in lines:
            num += 1
            if num > limit:
                break
            newSent = line.strip().replace("\n", "").split()
            scores = scoreSent(senWord, notWord, degreeWord, newSent)
            if scores >= 0:
                rows = {"sents": line.strip().replace("\n", ""), 'scores': 'pos'}
            else:
                rows = {"sents": line.strip().replace("\n", ""), 'scores': 'neg'}
            save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("情感分析结果.xlsx", index=None)
    print(df.shape)


def get_words_list(file_name):
    word_list = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            tmp_list = line.strip().replace("\n", "").split()
            word_list.append(tmp_list)
    return word_list


def build_lda():
    print("build_lda")
    raw_msg_file = 'mainbody_cut.csv'
    word_list = get_words_list(raw_msg_file)  # 列表，其中每个元素也是一个列表，即每行文字分词后形成的词语列表
    word_dict = corpora.Dictionary(word_list)  # 生成文档的词典，每个词与一个整型索引值对应
    corpus_list = [word_dict.doc2bow(text) for text in word_list]  # 词频统计，转化成空间向量格式
    print("corpus_list:", len(corpus_list))
    lda = models.ldamodel.LdaModel(corpus=corpus_list, id2word=word_dict, num_topics=20, alpha='auto')

    # 打印一下前二十名的主题规则，就是一些词的组合
    output_file = 'lda_output.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for pattern in lda.show_topics(num_topics=20):
            f.writelines("{},{}\n".format(pattern[0], pattern[1]))
    # 画图，主题数在10左右的文档有100个，大多数文档的主题都在7-16，如果把刻度分细或许可以了估计了
    tfidf = models.TfidfModel(corpus_list)
    corpus_tfidf = tfidf[corpus_list]
    thetas = [lda[c] for c in corpus_tfidf]
    plt.figure(dpi=300, figsize=(10, 8))
    plt.hist([len(t) for t in thetas], np.arange(100))
    plt.ylabel('Nr of documents')
    plt.xlabel('Nr of topics')
    plt.show()
    plt.savefig("lad.png")


if __name__ == '__main__':

    method = 'build_lda'

    if method == 'cluster':
        # 文本聚类
        cluster()

    if method == 'build_sentments':
        # 情感分析
        build_sentments(filename='mainbody_cut.csv')

    if method == 'build_lda':
        # LAD 主题模型
        build_lda()

