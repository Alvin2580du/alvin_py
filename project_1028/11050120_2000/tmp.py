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
        # 初始化权重矩阵
        self.transition_params = transition_params  # 一个 [num_tags，num_tags] 转换矩阵
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]  # InputSpec，每一个元素描述了对于输入的要求， ndim表示维度
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 2
        n_steps = input_shape[0][1]
        self.n_classes = input_shape[0][2]
        assert n_steps is None or n_steps >= 2
        # add_weight 给当前网络层添加权重
        self.transition_params = self.add_weight(shape=(self.n_classes, self.n_classes),
                                                 initializer='uniform', name='transition')
        # InputSpec 指定crf每层的输入的维度，数据类型，和大小
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, n_steps, self.n_classes)),
                           InputSpec(dtype='int32', shape=(None, 1))]
        self.build = True


    def viterbi_decode(self, potentials, sequence_length):
        """
        crf_decode(potentials,transition_params,sequence_length)  在tensorflow内解码
        参数:
            potentials: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor, 
            transition_params: 一个形状为[num_tags, num_tags] 的转移矩阵 
            sequence_length: 一个形状为[batch_size] 的 ,表示batch中每个序列的长度
        返回：
            decode_tags:一个形状为[batch_size, max_seq_len] 的tensor,类型是tf.int32.表示最好的序列标记. 
            best_score: 有个形状为[batch_size] 的tensor, 包含每个序列解码标签的分数.
        """
        decode_tags, best_score = crf_decode(potentials, self.transition_params, sequence_length)
        return decode_tags

    def call(self, inputs, mask=None, **kwargs):
        inputs, sequence_lengths = inputs
        # latten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
        self.sequence_lengths = K.flatten(sequence_lengths)
        y_pred = self.viterbi_decode(inputs, self.sequence_lengths)
        nb_classes = self.input_spec[0].shape[2]
        # one_hot(indices, nb_classes)
        # 输入为n维的整数张量，形如(batch_size, dim1, dim2, ...dim(n - 1))，
        # 输出为(n + 1) 维的one - hot编码，形如(batch_size, dim1, dim2, ... dim(n - 1), nb_classes)
        y_pred_one_hot = K.one_hot(y_pred, nb_classes)
        # in_train_phase(x, alt)
        # 如果处于训练模式，则选择x，否则选择alt，注意alt应该与x的shape相同
        return K.in_train_phase(inputs, y_pred_one_hot)

    def loss(self, y_true, y_pred):
        # argmax 在给定轴上求张量之最大元素下标，
        # cast 改变张量的数据类型，dtype只能是float16, float32或float64之一
        y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')
        # tf.contrib.crf.crf_log_likelihood 解释：
        # 在一个条件随机场里面计算标签序列的log-likelihood
        # 参数:
        #     inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,
        #               一般使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入.
        #     tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签.
        #     sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度.
        #     transition_params: 形状为[num_tags, num_tags] 的转移矩阵
        # 返回：
        #
        #     log_likelihood: 标量,log-likelihood
        #     transition_params: 形状为[num_tags, num_tags] 的转移矩阵
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