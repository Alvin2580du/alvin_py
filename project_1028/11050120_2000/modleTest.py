from keras.models import Model
from keras.layers import Embedding, Input, LSTM, TimeDistributed, Dense, Bidirectional
import numpy as np

from main import CrfModel
from utils import get_test_sents, viterbi_, get_train_sents

n_classes = 4  # 类别数
max_len = 100  # 样本的最大长度，不够做padding处理
batch_size = 128  # 批处理大小
epoch = 100  # 每个batch的迭代次数
tags = ['S', 'B', 'I', 'E']  # 标签


def cutTest(s, filename, batch_size, modelType='crf'):     # 分词函数
    # 建立模型
    word_ids = Input(batch_shape=(batch_size, max_len), dtype='int32')
    sequence_lengths = Input(batch_shape=[batch_size, 1], dtype='int32')
    word_embeddings = Embedding(vocab_size, n_classes)(word_ids)
    crf = CrfModel()
    if modelType == 'crf':
        pred = crf(inputs=[word_embeddings, sequence_lengths])
        model = Model(inputs=[word_ids, sequence_lengths], outputs=[pred])
        model.load_weights(filename)
    else:
        blstm = Bidirectional(LSTM(units=50, return_sequences=True))(word_embeddings)
        model = TimeDistributed(Dense(4, activation='tanh'))(blstm)
        pred = crf(inputs=[model, sequence_lengths])
        model = Model(inputs=[word_ids, sequence_lengths], outputs=[pred])
        model.load_weights(filename)

    if not s:  # 空字符直接返回
        return []
    sent_ids = []
    for c in s:
        if c != ' ':
            sent_ids.append(char2id.get(c, 0))
        else:
            sent_ids.append(char2id[u'。'])
    length = len(sent_ids)
    if length > max_len:
        X = sent_ids[:max_len]
    else:
        X = sent_ids + list(range(max_len - length))

    x_s = np.asarray([max_len] * batch_size, dtype='int32')
    probas = model.predict([np.array([X]), x_s])[0]  # 模型预测

    nodes = [dict(zip('SBIE', i)) for i in probas[:, :4]]  # 只取前4个
    nodes[0] = {i: j for i, j in nodes[0].items() if i in 'BS'}  # 首字标签只能是b或s
    nodes[-1] = {i: j for i, j in nodes[-1].items() if i in 'ES'}  # 末字标签只能是e或s

    id2tag = {0: 'S', 1: 'B', 2: 'I', 3: 'E'}  # 标签（sbme）与id之间的映射
    tag2id = {j: i for i, j in id2tag.items()}
    _ = model.get_weights()[-1][:4, :4]  # 从训练模型中取出最新得到的转移矩阵
    trans = {}
    for i in 'SBIE':
        for j in 'SBIE':
            trans[i + j] = _[tag2id[i], tag2id[j]]

    key, value = viterbi_(nodes, trans)
    result = [s[0]]
    for i, j in zip(s[1:], key[1:]):
        if j in 'BS':  # 词的开始
            result.append(i)
        else:  # 接着原来的词
            result[-1] += i
    return result


if __name__ == '__main__':

    # 把要测试的句子放到文件test.utf8中，执行下面的代码
    fw = open('predict.utf8', 'w', encoding='utf-8')
    data_name = './data/train.utf8'
    model_name = 'tmp_crflstm.model.h5'
    modelType = 'lstmcrf'
    sentences, words = get_train_sents(datasets=data_name)
    testsentences, testwords = get_test_sents(datasets='test.utf8')
    vocab_size = len(words)
    for sentence in testsentences:
        id2char = {i + 1: j for i, j in enumerate(words)}  # id到字的映射
        char2id = {j: i for i, j in id2char.items()}  # 字到id的映射
        res = cutTest(sentence, filename=model_name, batch_size=1, modelType=modelType)
        fw.writelines(" ".join(res)+'\n')
        print(" ".join(res))


