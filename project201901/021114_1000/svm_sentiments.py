# coding: utf-8
# 用gensim去做word2vec的处理，用sklearn当中的SVM进行建模
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
from tqdm import tqdm
import matplotlib.pyplot as plt
import wordcloud
from collections import Counter
from gensim import corpora, models


def readLines(filename):
    # 读文件的方法
    out = []
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            out.append(line.replace("\n", ""))
    return out


stopwords = readLines('stopwords.txt')


def sent2word(sentence):
    # 分词方法，输入句子，输出list
    segList = jieba.lcut(sentence)
    newSent = [i for i in segList if i not in stopwords]
    return newSent


#  载入数据，做预处理(分词)，切分训练集与测试集
def load_file_and_preprocessing():
    neg = pd.read_table('neg_train.txt', header=None)
    pos = pd.read_table('pos_train.txt', header=None)
    cw = lambda x: list(jieba.cut(x))

    # 新增一列 word ,存放分好词的评论，pos[0]代表表格第一列
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))), axis=0)
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    return x_train, x_test


# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算词向量
def get_train_vecs(x_train, x_test):
    n_dim = 300
    # 初始化模型和词表
    imdb_w2v = Word2Vec(x_train, size=n_dim, min_count=10)
    imdb_w2v = Word2Vec(size=300, window=5, min_count=10, workers=12)
    imdb_w2v.build_vocab(x_train)

    imdb_w2v.train(x_train, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])

    np.save('train_vecs.npy', train_vecs)
    print(train_vecs.shape)
    # 在测试集上训练
    imdb_w2v.train(x_test, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)

    imdb_w2v.save('w2v_model.pkl')
    # Build test tweet vectors then scale
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    # test_vecs = scale(test_vecs)
    np.save('test_vecs.npy', test_vecs)


def get_data():
    train_vecs = np.load('train_vecs.npy')
    y_train = np.load('y_train.npy')
    test_vecs = np.load('test_vecs.npy')
    y_test = np.load('y_test.npy')
    return train_vecs, y_train, test_vecs, y_test


# 训练svm模型
def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf = SVC(kernel='rbf', verbose=True)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, 'model.pkl')
    print(clf.score(test_vecs, y_test))


# 构建待预测句子的向量

def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('w2v_model.pkl')
    # imdb_w2v.train(words)
    train_vecs = build_sentence_vector(words, n_dim, imdb_w2v)
    # print train_vecs.shape
    return train_vecs


# 对单个句子进行情感判断
def svm_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    clf = joblib.load('model.pkl')
    result = clf.predict(words_vecs)
    if int(result[0]) == 1:
        return "positive"
    else:
        return "negative"


def build_svm():
    x_train, x_test = load_file_and_preprocessing()

    get_train_vecs(x_train, x_test)
    train_vecs, y_train, test_vecs, y_test = get_data()
    svm_train(train_vecs, y_train, test_vecs, y_test)

    fw = open("results.txt", 'w', encoding='utf-8')

    with open("comment_12,9_cut.txt", 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for string in tqdm(lines):
            res = svm_predict(string.replace("\n", ""))
            save = "{} {}".format(string.replace("\n", ""), res)
            fw.writelines(save + "\n")


def plot_word_cloud(file_name, savename):
    text = open(file_name, 'r', encoding='utf-8').read()
    wc = wordcloud.WordCloud(background_color="white", width=800, height=600,
                             max_font_size=50,
                             random_state=1,
                             max_words=1000,
                             font_path='msyh.ttf')
    wc.generate(text)
    plt.axis("off")
    plt.figure(dpi=600)
    wc.to_file(savename)


def cipin():
    text = open("comment_12,9_cut.txt", 'r', encoding='utf-8')
    lines = text.readlines()
    words = []
    for line in lines:
        line_sp = line.split()
        for one in line_sp:
            words.append(one)
    save = []
    for x, y in Counter(words).most_common(1000):
        rows = {'word': x, 'freq': y}
        save.append(rows)
        print(rows)
    df = pd.DataFrame(save)

    df.to_csv("wordfreq.csv", index=None)


def get_words_list(file_name):
    word_list = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            tmp_list = jieba.lcut(line.strip(), cut_all=False)
            word_list.append([term for term in tmp_list if str(term) not in stopwords and len(term) > 0])
    return word_list


def build_lda():
    print("build_lda ")
    raw_msg_file = 'comment_12,9.txt'
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


if __name__ == "__main__":
    method = 'build_svm'  # 修改这里，依次执行下面的method

    if method == 'build_svm':
        build_svm()

    if method == 'plot_word_cloud':
        plot_word_cloud("comment_12,9_cut.txt", 'wordcloud.png')

    if method == 'cipin':
        cipin()

    if method == 'build_lda':
        build_lda()
