from tensorflow.contrib.crf import crf_decode
import tensorflow as tf

import keras
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Input, LSTM, TimeDistributed, Dense, Bidirectional
from keras.models import Model, load_model

import numpy as np

assert keras.__version__ == '2.1.1', "请安装2.1.1版本的keras"
assert tf.__version__ == '1.4.0', "请安装1.4.0版本的tensorflow"


def create_custom_objects():
    """Returns the custom objects, needed for loading a persisted model."""
    instanceHolder = {'instance': None}

    class ClassWrapper(CRFLayer):
        def __init__(self, *args, **kwargs):
            instanceHolder['instance'] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instanceHolder['instance'], 'loss')
        return method(*args)

    return {'CRFLayer': ClassWrapper, 'loss': loss}


class CRFLayer(Layer):
    def __init__(self, transition_params=None, **kwargs):
        super(CRFLayer, self).__init__(**kwargs)
        self.transition_params = transition_params
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape[0]) == 3

        return input_shape[0]

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 2
        n_steps = input_shape[0][1]
        self.n_classes = input_shape[0][2]
        assert n_steps is None or n_steps >= 2

        self.transition_params = self.add_weight(shape=(self.n_classes, self.n_classes),
                                                 initializer='uniform',
                                                 name='transition')
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, n_steps, self.n_classes)),
                           InputSpec(dtype='int32', shape=(None, 1))]
        self.built = True

    def viterbi_decode(self, potentials, sequence_length):
        decode_tags, best_score = crf_decode(potentials, self.transition_params, sequence_length)
        return decode_tags

    def call(self, inputs, mask=None, **kwargs):
        inputs, sequence_lengths = inputs
        self.sequence_lengths = K.flatten(sequence_lengths)
        y_pred = self.viterbi_decode(inputs, self.sequence_lengths)
        nb_classes = self.input_spec[0].shape[2]
        y_pred_one_hot = K.one_hot(y_pred, nb_classes)

        return K.in_train_phase(inputs, y_pred_one_hot)

    def loss(self, y_true, y_pred):
        y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            y_pred, y_true, self.sequence_lengths, self.transition_params)
        loss = tf.reduce_mean(-log_likelihood)

        return loss

    def get_config(self):
        config = {
            'transition_params': K.eval(self.transition_params),
        }
        base_config = super(CRFLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def get_sents(datasets):
    sents = []
    tmp = []
    words = []
    with open(datasets, 'r', encoding='utf-8') as fr:
        while True:
            lines = fr.readline()
            if lines:
                if len(lines) > 2:
                    lines_sp = lines.split(" ")
                    w, label = lines_sp[0], lines_sp[1].replace("\n", "")
                    tmp.append((w, label))
                    if w not in words:
                        words.append(w)
                else:
                    sents.append(tmp)
                    tmp = []
            else:
                break
    return sents, words


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    print(len(inputs), len(targets), "*****************************************")
    assert len(inputs) == len(targets)
    if shuffle:
        global indices
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def traincrf(data_name, model_name='crfModel.h5'):
    n_classes = 4
    max_len = 75
    batch_size = 10000
    epoch = 50
    tags = ['S', 'B', 'I', 'E']
    sentences, words = get_sents(datasets=data_name)
    print(len(sentences), len(words))
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    vocab_size = len(words)

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=vocab_size - 1)

    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["E"])
    y = [to_categorical(i, num_classes=n_classes) for i in y]
    # 获得数据
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
    print(len(X_tr), len(y_tr), len(X_te), len(y_te))
    s = np.asarray([max_len] * batch_size, dtype='int32')

    # 建立模型
    word_ids = Input(batch_shape=(batch_size, max_len), dtype='int32')
    sequence_lengths = Input(batch_shape=[batch_size, 1], dtype='int32')
    word_embeddings = Embedding(vocab_size, n_classes)(word_ids)
    crf = CRFLayer()
    pred = crf(inputs=[word_embeddings, sequence_lengths])
    model = Model(inputs=[word_ids, sequence_lengths], outputs=[pred])
    model.compile(loss=crf.loss, optimizer='rmsprop', metrics=['accuracy'])  # metrics=['mae', 'acc']

    print(model.summary())
    for batch_x, batch_y in minibatches(X_tr, y_tr, batch_size=batch_size):
        model.fit([batch_x, s], np.array(batch_y), batch_size=batch_size, epochs=epoch)

    model.save(model_name)


def lstmCrf(data_name, model_name='LstmCrfModel.h5'):
    n_classes = 4
    max_len = 75
    batch_size = 200
    epoch = 10
    tags = ['S', 'B', 'I', 'E']
    sentences, words = get_sents(datasets=data_name)
    print(len(sentences), len(words))
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    vocab_size = len(words)

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=vocab_size - 1)

    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["E"])
    y = [to_categorical(i, num_classes=n_classes) for i in y]
    # 获得数据
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
    print(len(X_tr), len(y_tr), len(X_te), len(y_te))
    s = np.asarray([max_len] * batch_size, dtype='int32')

    # 建立模型

    word_ids = Input(batch_shape=(batch_size, max_len), dtype='int32')
    sequence_lengths = Input(batch_shape=[batch_size, 1], dtype='int32')
    word_embeddings = Embedding(vocab_size, n_classes)(word_ids)
    blstm = Bidirectional(LSTM(units=50, return_sequences=True))(word_embeddings)
    model = TimeDistributed(Dense(4, activation='tanh'))(blstm)
    crf = CRFLayer()
    pred = crf(inputs=[model, sequence_lengths])
    model = Model(inputs=[word_ids, sequence_lengths], outputs=[pred])
    model.compile(optimizer="rmsprop", loss=crf.loss, metrics=['accuracy'])

    print(model.summary())
    for batch_x, batch_y in minibatches(X_tr, y_tr, batch_size=batch_size):
        model.fit([batch_x, s], np.array(batch_y), batch_size=batch_size, epochs=epoch)

    model.save(model_name)


def crfTest(filename):
    model = load_model(filename, custom_objects=create_custom_objects())
    model.predict()


def max_in_dict(d):  # 定义一个求字典中最大值的函数
    key, value = d.items()[0]
    for i, j in d.items()[1:]:
        if j > value:
            key, value = i, j
    return key, value


def viterbi(nodes, trans):  # viterbi算法，跟前面的HMM一致
    paths = nodes[0]  # 初始化起始路径
    for l in range(1, len(nodes)):  # 遍历后面的节点
        paths_old, paths = paths, {}
        for n, ns in nodes[l].items():  # 当前时刻的所有节点
            max_path, max_score = '', -1e10
            for p, ps in paths_old.items():  # 截止至前一时刻的最优路径集合
                score = ns + ps + trans[p[-1] + n]  # 计算新分数
                if score > max_score:  # 如果新分数大于已有的最大分
                    max_path, max_score = p + n, score  # 更新路径
            paths[max_path] = max_score  # 储存到当前时刻所有节点的最优路径
    return max_in_dict(paths)


def cut(s, trans, filename):  # 分词函数，也跟前面的HMM基本一致
    model = load_model(filename, custom_objects=create_custom_objects())
    char2id = {}
    if not s:  # 空字符直接返回
        return []
    # 字序列转化为id序列。注意，经过我们前面对语料的预处理，字符集是没有空格的，
    # 所以这里简单将空格的id跟句号的id等同起来
    sent_ids = np.array([[char2id.get(c, 0) if c != ' ' else char2id[u'。'] for c in s]])
    probas = model.predict(sent_ids)[0]  # 模型预测
    nodes = [dict(zip('sbme', i)) for i in probas[:, :4]]  # 只取前4个
    nodes[0] = {i: j for i, j in nodes[0].items() if i in 'bs'}  # 首字标签只能是b或s
    nodes[-1] = {i: j for i, j in nodes[-1].items() if i in 'es'}  # 末字标签只能是e或s
    tags = viterbi(nodes, trans)[0]
    result = [s[0]]
    for i, j in zip(s[1:], tags[1:]):
        if j in 'bs':  # 词的开始
            result.append(i)
        else:  # 接着原来的词
            result[-1] += i
    return result


if __name__ == '__main__':

    method = 'traincrf'

    if method == 'traincrf':
        traincrf(data_name='./data/train.utf8')

    if method == 'lstmCrf':
        lstmCrf(data_name='./data/train.utf8')
