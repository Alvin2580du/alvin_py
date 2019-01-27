from tensorflow.contrib.crf import crf_decode
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, Input, LSTM, TimeDistributed, Dense, Bidirectional
from keras.models import Model

from sklearn.model_selection import train_test_split
import numpy as np

from utils import get_train_sents

assert keras.__version__ == '2.1.1', "请安装2.1.1版本的keras"
assert tf.__version__ == '1.4.0', "请安装1.4.0版本的tensorflow"


class CrfModel(Layer):
    def __init__(self, transition_params=None, **kwargs):
        super(CrfModel, self).__init__(**kwargs)
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
        crf_decode(potentials,transition_params,sequence_length) 
         在tensorflow内解码
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
        # cast 改变张量的数据类型
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
        #     log_likelihood: 标量,log-likelihood
        #     transition_params: 形状为[num_tags, num_tags] 的转移矩阵
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            y_pred, y_true, self.sequence_lengths, self.transition_params)
        loss = tf.reduce_mean(-log_likelihood)

        return loss

    def get_config(self):
        # eval 获取交换矩阵的值
        config = {
            'transition_params': K.eval(self.transition_params),
        }
        # get_config 返回包含模型配置信息的Python字典。模型也可以从它的config信息中重构回去
        base_config = super(CrfModel, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    # 数据批处理的函数
    print(len(inputs), len(targets), "*****************************************")
    assert len(inputs) == len(targets)  # 输入的训练数据和标签的长度要一致，不一致报错
    if shuffle:
        global indices
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:  # 是不是每次迭代之前要做随机扰乱的处理
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            # 使用slice函数做数据切片
            excerpt = slice(start_idx, start_idx + batch_size)
        # 把一个batch的数据返回
        yield inputs[excerpt], targets[excerpt]


n_classes = 4  # 类别数
max_len = 100  # 样本的最大长度，不够做padding处理
batch_size = 128  # 批处理大小
epoch = 100  # 每个batch的迭代次数
tags = ['S', 'B', 'I', 'E']  # 标签


def TrainCRF(data_name, model_name):
    sentences, words = get_train_sents(datasets=data_name)  # 调用get_sents函数，返回数据里面所有的句子和单词
    word2idx = {w: i + 1 for i, w in enumerate(words)}  # 单词和id对应起来
    tag2idx = {t: i for i, t in enumerate(tags)}  # 标签和id对应起来
    vocab_size = len(words)  # 单词的总数量

    X = [[word2idx[w[0]] for w in s] for s in sentences]  # 所有句子与id对应起来
    # pad_sequences:
    #               将长为nb_samples的序列（标量序列）转化为形如(nb_samples,nb_timesteps)2D numpy array。如果提供了参数maxlen，
    #               nb_timesteps=maxlen，否则其值为最长序列的长度。其他短于该长度的序列都会在后部填充0以达到该长度。
    #               长于nb_timesteps的序列将会被截断，以使其匹配目标长度。padding和截断发生的位置分别取决于padding和truncating
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=vocab_size - 1)
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["E"])
    # 转换为类别标签
    y = [to_categorical(i, num_classes=n_classes) for i in y]
    # 获得数据
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
    print(len(X_tr), len(y_tr), len(X_te), len(y_te))
    s = np.asarray([max_len] * batch_size, dtype='int32')

    # 建立模型
    word_ids = Input(batch_shape=(batch_size, max_len), dtype='int32')
    print(word_ids)
    sequence_lengths = Input(batch_shape=[batch_size, 1], dtype='int32')
    print(sequence_lengths)
    word_embeddings = Embedding(vocab_size, n_classes)(word_ids)
    print(word_embeddings)
    crf = CrfModel()
    pred = crf(inputs=[word_embeddings, sequence_lengths])
    print(pred)
    model = Model(inputs=[word_ids, sequence_lengths], outputs=[pred])
    print(model)
    model.compile(loss=crf.loss, optimizer='rmsprop', metrics=['accuracy'])   # rmsprop
    print(model.summary())
    k = 0
    for batch_x, batch_y in minibatches(X_tr, y_tr, batch_size=batch_size):
        model.fit([batch_x, s], np.array(batch_y), epochs=epoch, batch_size=batch_size)
        k += 1
        if k % 50 == 0:
            model.save("./models/{}_{}".format(k, model_name))
            print("saved")

    # 保存模型
    model.save(model_name)


def TrainLstmCrf(data_name, model_name):
    sentences, words = get_train_sents(datasets=data_name)
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
    print(sequence_lengths)
    word_embeddings = Embedding(vocab_size, n_classes)(word_ids)
    blstm = Bidirectional(LSTM(units=50, return_sequences=True))(word_embeddings)
    model = TimeDistributed(Dense(4, activation='tanh'))(blstm)
    crf = CrfModel()
    pred = crf(inputs=[model, sequence_lengths])
    model = Model(inputs=[word_ids, sequence_lengths], outputs=[pred])
    print("word_ids:{}".format(word_ids))
    print("sequence_lengths:{}".format(sequence_lengths))
    model.compile(optimizer="rmsprop", loss=crf.loss, metrics=['accuracy'])
    print(model.summary())

    k = 0
    for batch_x, batch_y in minibatches(X_tr, y_tr, batch_size=batch_size):
        model.fit([batch_x, s], np.array(batch_y), epochs=epoch, batch_size=batch_size)
        k += 1
        if k % 50 == 0:
            model.save("./models/{}_{}".format(k, model_name))
            print("saved")
    # 保存模型
    model.save(model_name)


if __name__ == '__main__':

    method = 'TrainLstmCrf'

    if method == 'get_sents':
        get_train_sents(datasets='./data/train.utf8')

    if method == 'TrainCRF':
        TrainCRF(data_name='./data/train.utf8', model_name='crf.model.h5')

    if method == 'TrainLstmCrf':
        TrainLstmCrf(data_name='./data/train.utf8', model_name='crflstm.model.h5')
