from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import sys

fw = open('result.txt', 'w', encoding='utf-8')
sys.stdout = fw


# 朴素贝叶斯，KNN，BP神经网络，SVM


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
    n_dim = 200
    # 初始化模型和词表
    imdb_w2v = Word2Vec(size=n_dim, window=5, min_count=10, workers=12)
    imdb_w2v.build_vocab(x_train)

    imdb_w2v.train(x_train, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])
    train_vecs = preprocessing.scale(train_vecs)

    np.save('train_vecs.npy', train_vecs)
    # 在测试集上训练
    imdb_w2v.train(x_test, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
    imdb_w2v.save('w2v_model.pkl')
    # Build test tweet vectors then scale
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    test_vecs = preprocessing.scale(test_vecs)
    np.save('test_vecs.npy', test_vecs)
    return train_vecs, test_vecs


def build_models():
    data = pd.read_excel("contentCut_senti.xlsx")
    data = shuffle(data)

    #  载入数据，做预处理(分词)，切分训练集与测试集
    neg = data[data['senti'].isin(['消极'])]
    pos = data[data['senti'].isin(['积极'])]
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))), axis=0)
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['comment_cut'], neg['comment_cut'])), y,
                                                        test_size=0.3)
    train_vecs, test_vecs = get_train_vecs(x_train, x_test)

    # MLPClassifier， SVC, MultinomialNB, KNeighborsClassifier

    for model in ['SVC', 'MLPClassifier', 'KNeighborsClassifier', 'BernoulliNB']:
        print("- * -" * 8, "{}".format(model), '- * -' * 8)
        if model == 'SVC':
            clf = SVC(kernel='rbf', verbose=True)
        elif model == 'MLPClassifier':
            clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                                solver='adam', )
        elif model == 'KNeighborsClassifier':
            clf = KNeighborsClassifier(n_neighbors=5)
        elif model == 'BernoulliNB':
            clf = BernoulliNB(alpha=1.0)
        else:
            raise NotImplementedError("没有这种方法，只能是svm，MultinomialNB， knn， MultinomialNB 中的一个")

        clf.fit(train_vecs, y_train)
        joblib.dump(clf, '{}_model.pkl'.format(model))
        print("{} 准确率：{:0.4f}".format(model, clf.score(test_vecs, y_test)))
        y_pred = clf.predict(test_vecs)
        res = classification_report(y_test, y_pred)
        print("{} 模型的输出参数报告：".format(model))
        print(res)
        print("- * -" * 20)


build_models()
fw.close()
