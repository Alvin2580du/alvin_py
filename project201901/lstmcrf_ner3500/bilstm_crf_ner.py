import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import os
import pickle
import numpy as np
import random
import sys

# 将tag转换成数字，是为了训练的时候使用
tag2label = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}


# 读取训练数据和测试数据，返回[([sent_],[tag_])]
def read_data(corpus_path):
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            if len(line.strip().split()) != 2:
                continue
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data


# 从训练数据中构建词典word2id
def build_vocab(vocab_path, corpus_path, min_count):
    data = read_data(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word)
    for word in low_freq_words:  # 要过滤掉低频词
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id  # 那些没有在词典中出现的词的id，模型训练好之后，肯定会遇到一些不在词典中的词
    word2id['<PAD>'] = 0
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


# 将句子中的词全部替换成其在词典中的id，在embedding层可以根据这个id得到这个词的embedding
# 对于那些没有出现在词典中的词，均认为其为<UNK>
def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        try:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:  # 对于没出现在词典中的词，就用<UNK>代替
                word = '<UNK>'
            sentence_id.append(word2id[word])
        except:
            continue
    return sentence_id


# 读取词典
def read_dict(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


# 随机初始化embedding，返回embedding矩阵，这个embedding可以在训练的更新，不需要强制传入已训练好的词向量，但是用已训练好的词向量肯定更快
def init_embedding(vocab, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


# 将原始数据进行分批，每一批包含的数据量是batch_size，
# 这里要将原始数据中的词换成其在word2id中的id，要把原始数据中的tag换成数字label
# 现在数据变成[([sent_id], [label])]
def gen_batch(data, batch_size, vocab, tag2label, shuffle=False):
    if shuffle:  # 是否混洗数据
        random.shuffle(data)
    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []
        seqs.append(sent_)
        labels.append(label_)
    if len(seqs) != 0:
        yield seqs, labels


# 从返回的标注序列中取出所有的实体
def get_entity(tag_seq, char_seq):
    length = len(char_seq)
    ENT = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER' or tag == 'B-LOC' or tag == 'B-ORG':
            if 'ent' in locals().keys():  # 变量ent如果存在那么就将其添加到实体中
                ENT.append(ent)
                del ent  # 将变量从命名空间中删除
            ent = char
            if i + 1 == length:
                ENT.append(ent)
        if tag == 'I-PER' or tag == 'I-LOC' or tag == 'I-ORG':
            if 'ent' in locals().keys():
                ent += char
            if i + 1 == length:
                if 'ent' in locals().keys():
                    ENT.append(ent)
        if tag not in ['I-PER', 'B-PER', 'I-LOC', 'B-LOC', 'I-ORG', 'B-ORG']:
            if 'ent' in locals().keys():
                ENT.append(ent)
                del ent
            continue
    return ENT


# 调用第三方perl程序进行评测
def conlleval(label_predict, label_path, metric_path):
    eval_perl = "./conlleval.pl"
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics


class BLC(object):
    def __init__(self, batch_size, epoch_num, hidden_dim, embeddings, dropout_keep, optimizer, lr, clip_grad,
                 tag2label, vocab, shuffle, model_path, summary_path, result_path, update_embedding=True):
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        self.dropout_keep_prob = dropout_keep
        self.optimizer = optimizer
        self.lr = lr
        self.clip_grad = clip_grad
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = shuffle
        self.model_path = model_path
        self.summary_path = summary_path
        self.result_path = result_path
        self.update_embedding = update_embedding

    # 创建图
    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer()
        self.biLSTM_layer()
        self.loss_layer()
        self.optimize()
        self.init()

    # 得到传递进来的真实的训练样本
    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    # 第一层，embedding层，根据传进来的词找到id，再根据id找到其对应的embedding，并返回
    def lookup_layer(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids, name="word_embeddings")
        # 在进入下一层之前先做一个dropout，防止过拟合
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    # 第二层，双向lstm层
    def biLSTM_layer(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                                inputs=self.word_embeddings,
                                                                                sequence_length=self.sequence_lengths,
                                                                                dtype=tf.float32)
            # 对正反输出的向量直接拼接
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)
        # 将lstm输出的向量转到了多个tag上的概率
        # 最终bi-lstm的输出是一个n*m的矩阵p，其中n表示词的个数，m表示tag的个数，其中pij表示词wi映射到tagj的非归一化概率
        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W", shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable(name="b", shape=[self.num_tags], initializer=tf.zeros_initializer(), dtype=tf.float32)
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    # 第三层，CRF层
    def loss_layer(self):
        # log_likelihood是对数似然函数，transition_params是转移概率矩阵
        log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.labels,
                                                                    sequence_lengths=self.sequence_lengths)
        self.loss = -tf.reduce_mean(log_likelihood)
        tf.summary.scalar("loss", self.loss)

    # 确定优化方法
    def optimize(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init(self):
        self.init = tf.global_variables_initializer()

    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    # 训练模型
    def train(self, train, dev):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(self.init)
            self.add_summary(sess)
            for epoch in range(self.epoch_num):
                print('第', epoch + 1, '轮训练')
                self.run(sess, train, dev, epoch, saver)

    # 进行一轮训练
    def run(self, sess, train, dev,  epoch, saver):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        print("num_batches : ", num_batches)
        # 将所有的训练数据分成多个batch，每次把一个batch送进网络中学习
        batches = gen_batch(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            sys.stdout.write('batch总数: {}, 当前batch: {}'.format(num_batches, step + 1) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            print("loss:{}, step_num_:{} step:{} ".format(loss_train, step_num_, step))
            self.file_writer.add_summary(summary, step_num)

            if step % 50 == 0:
                saver.save(sess, self.model_path, global_step=step_num)
                print("Save Suceess ")

    # mode为test的时候调用，运行训练好的模型为输入的句子进行命名实体识别
    def test(self, sess, sent):
        label_list = []
        # 生成多个batch
        for seqs, labels in gen_batch(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        # 将数字转换成tag
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def predict(self, sess, seqs):
        feed_dict, seq_len_list = self.get_feed(seqs, dropout=1.0)
        logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            # 调用维特比算法求最优标注序列
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        return label_list, seq_len_list

    # 准备输入的参数
    def get_feed(self, seqs, labels=None, lr=None, dropout=None):
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

if __name__ == '__main__':
    # train/test/build_vocab
    mode = 'test'
    # 参数
    args = {}
    args['train_data'] = 'data'  # 训练数据路径
    args['test_data'] = 'data'  # 测试数据路径
    args['batch_size'] = 64  # 每一批用来训练的样本数
    args['epoch'] = 10  # 迭代次数
    args['hidden_dim'] = 100  # lstm接受的数据的维度
    args['optimizer'] = 'Adam'  # 优化损失函数的方法
    args['lr'] = 0.001  # 学习率
    args['clip'] = 5.0  # 限定梯度更新的时候的阈值
    args['dropout'] = 0.5  # 保留率
    args['update_embedding'] = True  # 是否要对embedding进行更新，embedding初始化之后，这里设置成更新，就可以更新embedding
    args['embedding_dim'] = 100  # embedding的维度
    args['shuffle'] = True  # 是否每次在把数据送进lstm中训练时都混洗

    # 读取词典，把一个字映射到一个id，这个词典是从训练数据中得到的
    word2id = read_dict(os.path.join('.', args['train_data'], 'word2id.pkl'))
    # 随机初始化embedding
    embeddings = init_embedding(word2id, args['embedding_dim'])

    # 设置模型的输出路径
    model_path = 'BLCM'
    output_path = os.path.join('.', model_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    summary_path = os.path.join(output_path, "summaries")
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    result_path = os.path.join(output_path, "results")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if mode == 'build_vocab':
        build_vocab(vocab_path='./data/word2id.pkl', corpus_path='./data/train_data', min_count=1)

    # 训练模型
    if mode == 'train':
        # 读取数据
        train_path = os.path.join('.', args['train_data'], 'train_data')
        test_path = os.path.join('.', args['test_data'], 'test_data')
        train_data = read_data(train_path)
        test_data = read_data(test_path)
        # 创建模型并训练
        model = BLC(batch_size=args['batch_size'], epoch_num=args['epoch'], hidden_dim=args['hidden_dim'],
                    embeddings=embeddings,
                    dropout_keep=args['dropout'], optimizer=args['optimizer'], lr=args['lr'], clip_grad=args['clip'],
                    tag2label=tag2label, vocab=word2id, shuffle=args['shuffle'],
                    model_path=ckpt_prefix, summary_path=summary_path, result_path=result_path,
                    update_embedding=args['update_embedding'])
        model.build_graph()
        model.train(train_data, test_data)
    # 演示模型
    elif mode == 'test':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        model = BLC(batch_size=args['batch_size'], epoch_num=args['epoch'], hidden_dim=args['hidden_dim'],
                    embeddings=embeddings,
                    dropout_keep=args['dropout'], optimizer=args['optimizer'], lr=args['lr'], clip_grad=args['clip'],
                    tag2label=tag2label, vocab=word2id, shuffle=args['shuffle'],
                    model_path=ckpt_file, summary_path=summary_path, result_path=result_path,
                    update_embedding=args['update_embedding'])
        model.build_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, ckpt_file)
            while 1:
                print('输入待识别的句子: ')
                sent = input()
                if sent == '' or sent.isspace():
                    break
                else:
                    sent = list(sent.strip())
                    data = [(sent, ['O'] * len(sent))]
                    tag = model.test(sess, data)
                    print(tag)
                    ENT = get_entity(tag, sent)
                    print('ENT: {}\n'.format(ENT))
