from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import numpy as np
import jieba
from keras.models import load_model
import json
import sys
from keras.layers import Embedding, Bidirectional
import os.path
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
import tornado

jieba.load_userdict("userdict.txt")


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
    newSent = []
    for i in segList:
        if not i.replace(" ", ""):
            continue
        if i in stopwords:
            continue
        newSent.append(i)
    if len(newSent) > 1:
        return newSent
    else:
        return None


def train(trainSet):
    maxlen = 0
    word_freqs = collections.Counter()
    num_recs = 0
    labels = []
    with open(trainSet, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sentence, label = line.strip().split("\t")
            if label not in labels:
                labels.append(label)
            words = sent2word(sentence)
            if not words:
                continue
            if len(words) > maxlen:
                maxlen = len(words)
            for word in words:
                word_freqs[word] += 1
            num_recs += 1
    print('max_len ', maxlen)
    print('nb_words ', len(word_freqs))
    print("labels", labels)
    print("num_recs", num_recs)
    ## 准备数据
    MAX_FEATURES = 2000
    MAX_SENTENCE_LENGTH = 100
    vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
    print("vocab_size", vocab_size)
    word2index = {x[0]: i + 2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v: k for k, v in word2index.items()}
    json_str = json.dumps(index2word)
    with open('index2word.json', 'w') as json_file:
        json_file.write(json_str)

    X = np.empty(num_recs, dtype=list)
    y = np.zeros(num_recs)
    i = 0
    with open(trainSet, 'r', encoding='utf-8') as f:
        for line in f:
            sentence, label = line.strip().split("\t")
            words = sent2word(sentence)
            if not words:
                continue
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[i] = seqs
            y[i] = int(label)
            i += 1
    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
    print("## 数据划分")
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    print("## 网络构建")
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=MAX_SENTENCE_LENGTH))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("## 网络训练")
    model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, ytest), verbose=2)
    # 保存训练模型
    model.save(r"bilstm_epoch_{}.h5".format(NUM_EPOCHS))
    ## 预测
    score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))


# 对单个句子进行情感判断
def predicts_label(sentence):
    MAX_SENTENCE_LENGTH = 40
    index2word = json.load(open('index2word.json', 'r'))
    word2index = {v: k for k, v in index2word.items()}
    model = load_model('sentiment_analysis_lstm.h5')
    XX = np.empty(1, dtype=list)
    i = 0
    words = jieba.lcut(sentence)
    print(words)
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i += 1
    XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
    labels = [int(round(x[0])) for x in model.predict(XX)][0]
    if labels == 1:
        return '积极'
    else:
        return '消极'


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class UserHandler(tornado.web.RequestHandler):
    def post(self):
        message = self.get_argument("message")
        print("输入的句子是：{}".format(message))
        res = predicts_label(message)
        self.render("message.html", message="{}".format(message), results='{}'.format(res))


if __name__ == '__main__':

    method = 'server'

    if method == 'train':
        print("训练模型")
        trainset = 'trainSmall.txt'
        train(trainSet=trainset)

    if method == 'test':
        # string = "垃圾，还说质量保证，朋友说算了，叫我不要退货"
        string = '裤子收到了，质量非常好，这个价格买到这个料子的裤子简直太高兴了，下次还来这家店买，建议大家可以买，质量很好'
        labels = predicts_label(string)

    if method == 'server':
        ports = 8992
        define("port", default=ports, help="run on the given port", type=int)

        handlers = [
            (r"/", IndexHandler),
            (r"/user", UserHandler)
        ]
        template_path = os.path.join(os.path.dirname(__file__), "template")
        tornado.options.parse_command_line()
        app = tornado.web.Application(handlers, template_path)
        http_server = tornado.httpserver.HTTPServer(app)
        http_server.listen(options.port)
        tornado.ioloop.IOLoop.instance().start()
