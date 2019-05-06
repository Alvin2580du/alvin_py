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
from keras.models import load_model


def get_train_data():
    data = pd.read_excel("./datasets/interaction_copy.xlsx")
    if not os.path.exists("./data"):
        os.makedirs("./data")
    for x, y in data.groupby(by='sentiment'):
        tmp = y['askcontent']
        tmp.to_csv("./data/{}.csv".format(int(x)), index=None)
        print(tmp.shape)


def build():
    datapaths = "./data/"

    neutral_data = []
    y_neutral = []
    negative_data = []
    y_negative = []

    print("#------------------------------------------------------#")
    print("加载数据集")
    with codecs.open(datapaths + "1.csv", "r", "utf-8") as f1, \
            codecs.open(datapaths + "0.csv", "r", "utf-8") as f2, \
            codecs.open(datapaths + "-1.csv", "r", "utf-8") as f3:
        for line in f1:
            neutral_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
            y_neutral.append([1, 0])
        for line in f2:
            neutral_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
            y_neutral.append([1, 0])
        for line in f3:
            negative_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
            y_negative.append([0, 1])

    # print("positive data:{}".format(len(positive_data)))
    print("neutral data:{}".format(len(neutral_data)))
    print("negative data:{}".format(len(negative_data)))

    x_text = neutral_data + negative_data
    y_label = y_neutral + y_negative
    print("#------------------------------------------------------#")
    print("\n")

    max_document_length = 100
    min_frequency = 2
    vocab = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency, tokenizer_fn=list)
    x = np.array(list(vocab.fit_transform(x_text)))
    vocab_dict = collections.OrderedDict(vocab.vocabulary_._mapping)

    with codecs.open(r"vocabulary.txt", "w", "utf-8") as f:
        for key, value in vocab_dict.items():
            f.write("{} {}\n".format(key, value))

    print("#----------------------------------------------------------#")
    print("\n")

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

    train_neutral_label = 0
    train_negative_label = 0

    test_neutral_label = 0
    test_negative_label = 0

    for i in range(len(y_train)):
        if y_train[i, 1] == 1:
            train_neutral_label += 1
        else:
            train_negative_label += 1

    for i in range(len(y_test)):
        if y_test[i, 1] == 1:
            test_neutral_label += 1
        else:
            test_negative_label += 1
    print("#----------------------------------------------------------#")
    print("训练集中 neutral 样本个数：{}".format(train_neutral_label))
    print("训练集中 negative 样本个数：{}".format(train_negative_label))
    print("#----------------------------------------------------------#")
    print("测试集中 neutral 样本个数：{}".format(test_neutral_label))
    print("测试集中 negative 样本个数：{}".format(test_negative_label))
    print("#----------------------------------------------------------#")
    print("\n")
    print("#----------------------------------------------------------#")
    print("读取预训练词向量矩阵")
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

    print("预训练模型中Token总数：{} = {}".format(nwords, len(embedding_index)))
    print("预训练模型的维度：{}".format(ndims))
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
    print("词汇表中未找到单词个数：{}".format(notfoundword))
    print("#----------------------------------------------------------#")
    print("\n")

    print("#---------------------------------------------------#")
    print("Build model .................")
    print("NN structure .......")
    print("Embedding layer --- Bi_LSTM layer --- Dense layer")
    print("#---------------------------------------------------#")
    print("\n")

    batch_size = 32
    max_sentence_length = max_document_length
    embedding_dims = ndims
    dropout = 0.2
    num_classes = 2
    epochs = 3

    # 定义网络结构
    model = Sequential()
    model.add(Embedding(len(vocab_dict),
                        embedding_dims,
                        weights=[embedding_matrix],
                        input_length=max_sentence_length,
                        trainable=False))
    model.add(Dropout(dropout))
    model.add((LSTM(64)))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation="sigmoid"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    # 训练得分和准确度
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("预测得分:{}".format(score))
    print("预测准确率:{}".format(acc))
    # 模型预测
    predictions = model.predict(x_test)
    print("测试集的预测结果，对每个类有一个得分/概率，取值大对应的类别")
    print(predictions)
    """
    预测得分:0.5989024954268144
    预测准确率:0.7876025523070107
    """
    # 模型预测类别
    predict_class = model.predict_classes(x_test)
    print("测试集的预测类别")
    print(predict_class)
    print("\n")
    # 模型保存
    model.save(r"sentiment_analysis_lstm.h5")
    # 模型总结
    print("输出模型总结")
    print(model.summary())
    # 模型的配置文件
    config = model.get_config()
    print("#---------------------------------------------------#")
    print("输出模型配置信息")
    print(config)
    print("#---------------------------------------------------#")
    print("\n")


def cw(inputs):
    try:
        inputs_cut = inputs.split()
        out = []
        for x in inputs_cut:
            try:
                x_cut = x.split("/")[0]
                out.append(x_cut)
            except:
                continue
        return " ".join(out)
    except:
        return inputs


if __name__ == '__main__':

    method = 'predicts'

    if method == 'get_train_data':
        get_train_data()

    if method == 'build':
        build()

    if method == 'predicts':
        model = load_model('sentiment_analysis_lstm.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        batch_size = 256

        def predicts(line):
            x_text = []
            for one in line:
                if isinstance(one, str):
                    one_cut = cw(one)
                    x_text.append(one_cut)
                else:
                    x_text.append("")
            max_document_length = 100
            min_frequency = 2
            vocab = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency, tokenizer_fn=list)
            x_test = np.array(list(vocab.fit_transform(x_text)))
            proba = model.predict(x_test, batch_size=batch_size)
            out_ = proba.argmax(axis=-1)
            out = []
            for i in out_:
                if i == 0:
                    out.append(0)
                else:
                    out.append(-1)
            return out

        data = pd.read_table("./datasets/interaction_copy.csv", chunksize=batch_size, sep=',')
        for df in data:
            lines = df['askcontent']
            res = predicts(lines)
            df['predicts'] = res
            df.to_csv("interaction_copy_predicts.csv", index=None, mode='a')

