import codecs
import jieba
from tensorflow.contrib import learn
import numpy as np
import collections
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding, Bidirectional
from keras.layers import LSTM
import pandas as pd
import os


def get_train_data():
    data = pd.read_excel("trainSet.xlsx")
    data['text'] = data['Title'] + data['Abstract']
    if not os.path.exists("./data"):
        os.makedirs("./data")

    for x, y in data.groupby(by='Category'):
        tmp = y['text']
        tmp.to_csv("./data/{}.csv".format(x), index=None)
        print(tmp.shape)


def build():
    datapaths = "./data/"

    class1_data = []
    y_1 = []

    class2_data = []
    y_2 = []

    class3_data = []
    y_3 = []

    class4_data = []
    y_4 = []

    class5_data = []
    y_5 = []

    class6_data = []
    y_6 = []

    class7_data = []
    y_7 = []

    print("#------------------------------------------------------#")
    print("加载数据集")
    with codecs.open(datapaths + "0.csv", "r", "utf-8") as f1,\
            codecs.open(datapaths + "1.csv", "r", "utf-8") as f2,\
            codecs.open(datapaths + "2.csv", "r", "utf-8") as f3,\
            codecs.open(datapaths + "3.csv", "r", "utf-8") as f4, \
            codecs.open(datapaths + "4.csv", "r", "utf-8") as f5, \
            codecs.open(datapaths + "5.csv", "r", "utf-8") as f6,\
            codecs.open(datapaths + "6.csv", "r", "utf-8") as f7:
        for line in f1:
            class1_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
            y_1.append([1, 0, 0, 0, 0, 0, 0])
        for line in f2:
            class2_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
            y_2.append([0, 1, 0, 0, 0, 0, 0])
        for line in f3:
            class3_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
            y_3.append([0, 0, 1, 0, 0, 0, 0])

        for line in f4:
            class4_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
            y_4.append([0, 0, 0, 1, 0, 0, 0])

        for line in f5:
            class5_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
            y_5.append([0, 0, 0, 0, 1, 0, 0])

        for line in f6:
            class6_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
            y_6.append([0, 0, 0, 0, 0, 1, 0])

        for line in f7:
            class7_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
            y_7.append([0, 0, 0, 0, 0, 0, 1])

    print("class1_data data: {}".format(len(class1_data)))
    print("class2_data data: {}".format(len(class2_data)))
    print("class3_data data: {}".format(len(class3_data)))
    print("class4_data data: {}".format(len(class4_data)))
    print("class5_data data: {}".format(len(class5_data)))
    print("class6_data data: {}".format(len(class6_data)))
    print("class7_data data: {}".format(len(class7_data)))

    x_text = class1_data + class2_data + class3_data + class4_data + class5_data + class6_data + class7_data

    y_label = y_1 + y_2 + y_3 + y_4 + y_5 + y_6 + y_7

    max_document_length = 100
    min_frequency = 2
    vocab = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency, tokenizer_fn=list)
    x = np.array(list(vocab.fit_transform(x_text)))
    vocab_dict = collections.OrderedDict(vocab.vocabulary_._mapping)

    with codecs.open(r"vocabulary.txt", "w", "utf-8") as f:
        for key, value in vocab_dict.items():
            f.write("{} {}\n".format(key, value))

    print("#----------------------------------------------------------#")
    print("数据混洗")
    np.random.seed(10)
    y = np.array(y_label)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    test_sample_percentage = 0.3
    test_sample_index = -1 * int(test_sample_percentage * float(len(y)))
    x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
    y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]

    print("读取預训练词向量矩阵")
    embedding_index = {}
    with codecs.open("F:\\BaiduNetdiskDownload\\sgns.wiki.bigram\\sgns.wiki.bigram", "r", "utf-8") as f:
        line = f.readline()
        nwords = int(line.strip().split(" ")[0])
        ndims = int(line.strip().split(" ")[1])
        for line in f:
            values = line.split()
            words = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embedding_index[words] = coefs

    print("預训练模型中Token总数：{} = {}".format(nwords, len(embedding_index)))
    print("預训练模型的维度：{}".format(ndims))
    print("#----------------------------------------------------------#")
    print("\n")

    print("#----------------------------------------------------------#")
    print("将vocabulary中的 index-word 对应关系映射到 index-word vector形式")

    embedding_matrix = []
    notfoundword = 0

    for word in vocab_dict.keys():
        if word in embedding_index.keys():
            embedding_matrix.append(embedding_index[word])
        else:
            notfoundword += 1
            embedding_matrix.append(np.random.uniform(-1, 1, size=ndims))

    embedding_matrix = np.array(embedding_matrix, dtype=np.float32)  # 必须使用 np.float32

    batch_size = 16
    embedding_dims = ndims
    dropout = 0.2
    num_classes = 7
    epochs = 10

    # 定义网络结构
    model = Sequential()
    model.add(Embedding(len(vocab_dict),
                        embedding_dims,
                        weights=[embedding_matrix],
                        input_length=max_document_length,
                        trainable=False))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation="sigmoid"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    # 训练得分和准确度
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("预测准确率2:{}".format(acc))
    score, acc = model.evaluate(x_train, y_train, batch_size=batch_size)

    print("预测准确率1:{}".format(acc))
    # 模型保存
    model.save(r"sentiment_analysis_lstm.h5")


if __name__ == '__main__':

    method = 'build'

    if method == 'get_train_data':
        get_train_data()

    if method == 'build':
        build()
