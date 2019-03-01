# 用gensim去做word2vec的处理，用sklearn当中的SVM进行建模
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC

data = pd.read_excel("data.xlsx")


#  载入数据，做预处理(分词)，切分训练集与测试集
def load_file_and_preprocessing():
    neg = data[data['senti'].isin(['neg'])]
    pos = data[data['senti'].isin(['pos'])]
    neu = data[data['senti'].isin(['neu'])]

    cw = lambda x: list(jieba.cut(x))

    # 新增一列 word ,存放分好词的评论，pos[0]代表表格第一列
    pos['words'] = pos['text'].apply(cw)
    neg['words'] = neg['text'].apply(cw)
    neu['words'] = neu['text'].apply(cw)
    pos.to_excel('pos_cut.xlsx', index=None)
    neg.to_excel('neg_cut.xlsx', index=None)
    neu.to_excel('neu_cut.xlsx', index=None)

    y = np.concatenate((np.ones(len(pos)), -np.ones(len(neg)), np.zeros(len(neu))), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'], neu['words'])), y,
                                                        test_size=0.2)
    print(x_train.shape, x_test.shape)
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
    joblib.dump(clf, 'svm_model.pkl')
    print("svm 准确率：{}".format(clf.score(test_vecs, y_test)))


# 构建待预测句子的向量
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('w2v_model.pkl')
    train_vecs = build_sentence_vector(words, n_dim, imdb_w2v)
    return train_vecs


# 对单个句子进行情感判断
def svm_predict():
    string = '风险资本退出渠道不畅与风险资本市场波动性较大是导致我国风险投资对技术创新以及高新技术产业发展支撑作用不显著的主要原因'
    print(string)
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    clf = joblib.load('svm_model.pkl')
    result = clf.predict(words_vecs)
    if int(result[0]) == 1:
        return "positive"
    elif int(result[0]) == 0:
        return 'neu'
    else:
        return "negative"


def build_svm():
    x_train, x_test = load_file_and_preprocessing()
    get_train_vecs(x_train, x_test)
    train_vecs, y_train, test_vecs, y_test = get_data()
    svm_train(train_vecs, y_train, test_vecs, y_test)


# build_svm()
res = svm_predict()
print(res)
