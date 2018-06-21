# -*- coding:utf-8 -*-

import tensorflow as tf
import os, time
import random
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict, OrderedDict

# Parameters
# =================================================

tf.flags.DEFINE_integer('rnn_size', 64,
                        'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 4, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 30, 'Sequence length (default : 32)')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 30, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.002, 'learning rate')
tf.flags.DEFINE_float('decay_rate', 0.97, 'decay rate for rmsprop')
tf.flags.DEFINE_string('train_file', 'train_big.csv', 'train raw file')
tf.flags.DEFINE_string('test_file', 'test_big.csv', 'train raw file')
tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model save directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log directory')
tf.flags.DEFINE_string('init_from', None, 'continue training from saved model at this path')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

questions = pd.read_csv("./data/question.csv")


def get_sentences(questionid):
    # qid,words,chars
    words = questions[questions['qid'].isin([questionid])]['words'].values
    chars = questions[questions['qid'].isin([questionid])]['chars'].values
    return "{} {}".format(" ".join(words), " ".join(chars))


class InputHelper():

    def __init__(self, data_dir, input_file_name, batch_size, sequence_length, is_train=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.is_train = is_train

        vocab_file = os.path.join(data_dir, 'vocab.pkl')
        input_file_dir = os.path.join(data_dir, input_file_name)

        if not (os.path.exists(vocab_file)):
            print('readling train file')
            self.preprocess(input_file_dir, vocab_file)
        else:
            print('loading vocab file')
            self.load_vocab(vocab_file)

        if is_train:
            self.create_batches(input_file_dir)
            self.reset_batch()

    def preprocess(self, input_file, vocab_file, min_freq=2):

        token_freq = defaultdict(int)

        with open(input_file, 'r') as fr:
            while True:
                line = fr.readline()
                if "question" in line:
                    continue
                if line:
                    #  question1,question2,label
                    seq1, seq2, label = line.split(',')
                    seq = "{} {}".format(seq1, seq2)

                    for token in seq.split(' '):
                        token_freq[token] += 1
                else:
                    break

        token_list = [w for w in token_freq.keys() if token_freq[w] >= min_freq]
        token_list.append('<pad>')
        token_dict = {token: index for index, token in enumerate(token_list)}
        with open(vocab_file, 'wb') as f:
            pickle.dump(token_dict, f)

        self.token_dictionary = token_dict
        self.vocab_size = len(self.token_dictionary)

    def load_vocab(self, vocab_file):

        with open(vocab_file, 'rb') as f:
            self.token_dictionary = pickle.load(f)
            self.vocab_size = len(self.token_dictionary)

    def text_to_array(self, text, is_clip=True):

        seq_ids = [int(self.token_dictionary.get(token)) for token in text if
                   self.token_dictionary.get(token) is not None]
        if is_clip:
            seq_ids = seq_ids[:self.sequence_length]
        return seq_ids

    def padding_seq(self, seq_array, padding_index):

        for i in range(len(seq_array), self.sequence_length):
            seq_array.append(padding_index)

    def create_batches(self, text_file):
        x1 = []
        x2 = []
        y = []
        padding_index = self.vocab_size - 1
        with open(text_file, 'r') as fr:
            while True:
                line = fr.readline()
                if "question" in line:
                    continue
                if line:
                    seq1, seq2, label = line.split(',')
                    seq1_array = self.text_to_array(seq1.split(' '))
                    seq2_array = self.text_to_array(seq2.split(' '))
                    self.padding_seq(seq1_array, padding_index)
                    self.padding_seq(seq2_array, padding_index)
                    x1.append(seq1_array)
                    x2.append(seq2_array)
                    y.append(label)
                else:
                    break

        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(y)

        self.num_samples = len(y)
        self.num_batches = self.num_samples / self.batch_size
        indices = np.random.permutation(self.num_samples)
        self.x1 = x1[indices]
        self.x2 = x2[indices]
        self.y = y[indices]

    def next_batch(self):

        begin = self.pointer
        end = self.pointer + self.batch_size
        x1_batch = self.x1[begin:end]
        x2_batch = self.x2[begin:end]
        y_batch = self.y[begin:end]

        new_pointer = self.pointer + self.batch_size

        if new_pointer >= self.num_samples:
            self.eos = True
        else:
            self.pointer = new_pointer

        return x1_batch, x2_batch, y_batch

    def reset_batch(self):
        self.pointer = 0
        self.eos = False


class SiameseLSTM(object):

    def bi_lstm(self, rnn_size, layer_size, keep_prob):
        # forward rnn
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                           output_keep_prob=keep_prob)

        # backward rnn
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                           output_keep_prob=keep_prob)

        return lstm_fw_cell_m, lstm_bw_cell_m

    def weight_variables(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name=name)

    def bias_variables(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    def transform_inputs(self, inputs, rnn_size, sequence_length):
        inputs = tf.transpose(inputs, [1, 0, 2])
        inputs = tf.reshape(inputs, [-1, rnn_size])
        inputs = tf.split(inputs, sequence_length, 0)
        return inputs

    def contrastive_loss(self, Ew, y):
        l_1 = 0.25 * tf.square(1 - Ew)
        l_0 = tf.square(tf.maximum(Ew, 0))
        loss = tf.reduce_sum(y * l_1 + (1 - y) * l_0)
        return loss

    def __init__(self, rnn_size, layer_size, vocab_size, sequence_length, keep_prob, grad_clip):
        self.input_x1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_x2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_y')

        with tf.device('/cpu:0'):
            embedding = self.weight_variables([vocab_size, rnn_size], 'embedding')
            inputs_x1 = tf.nn.embedding_lookup(embedding, self.input_x1)
            inputs_x2 = tf.nn.embedding_lookup(embedding, self.input_x2)

        inputs_x1 = self.transform_inputs(inputs_x1, rnn_size, sequence_length)
        inputs_x2 = self.transform_inputs(inputs_x2, rnn_size, sequence_length)

        with tf.variable_scope('output'):
            bilstm_fw, bilstm_bw = self.bi_lstm(rnn_size, layer_size, keep_prob)
            outputs_x1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(bilstm_fw, bilstm_bw, inputs_x1,
                                                                       dtype=tf.float32)
            output_x1 = tf.reduce_mean(outputs_x1, 0)
            ## 开启变量重用的开关
            tf.get_variable_scope().reuse_variables()
            outputs_x2, _, _ = tf.contrib.rnn.static_bidirectional_rnn(bilstm_fw, bilstm_bw, inputs_x2,
                                                                       dtype=tf.float32)
            output_x2 = tf.reduce_mean(outputs_x2, 0)

        with tf.variable_scope('dense_layer'):
            fc_w1 = self.weight_variables([2 * rnn_size, 128], 'fc_w1')
            fc_w2 = self.weight_variables([2 * rnn_size, 128], 'fc_w2')

            fc_b1 = self.bias_variables([128], 'fc_b1')
            fc_b2 = self.bias_variables([128], 'fc_b2')

            self.logits_1 = tf.matmul(output_x1, fc_w1) + fc_b1
            self.logits_2 = tf.matmul(output_x2, fc_w2) + fc_b2

        print('fw(x1) shape: ', self.logits_1.shape)
        print('fw(x2) shape: ', self.logits_2.shape)

        # calc Energy 1,2 ..
        f_x1x2 = tf.reduce_sum(tf.multiply(self.logits_1, self.logits_2), 1)
        norm_fx1 = tf.sqrt(tf.reduce_sum(tf.square(self.logits_1), 1))
        norm_fx2 = tf.sqrt(tf.reduce_sum(tf.square(self.logits_2), 1))
        self.Ew = f_x1x2 / (norm_fx1 * norm_fx2)

        print('Ecos shape: ', self.Ew.shape)

        # contrastive loss
        self.y_data = tf.placeholder(tf.float32, shape=[None], name='y_data')
        self.cost = self.contrastive_loss(self.Ew, self.y_data)

        # train optimization
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


def train():
    train_data_loader = InputHelper(FLAGS.data_dir, FLAGS.train_file, FLAGS.batch_size,
                                    FLAGS.sequence_length, True)

    FLAGS.vocab_size = train_data_loader.vocab_size
    FLAGS.num_batches = train_data_loader.num_batches

    if FLAGS.init_from is not None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.init_from)

    model = SiameseLSTM(FLAGS.rnn_size, FLAGS.layer_size, FLAGS.vocab_size,
                        FLAGS.sequence_length, FLAGS.dropout_keep_prob, FLAGS.grad_clip)

    tf.summary.scalar('train_loss', model.cost)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if FLAGS.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(FLAGS.num_epochs):
            train_data_loader.reset_batch()
            b = 0
            while not train_data_loader.eos:
                b += 1
                start = time.time()
                x1_batch, x2_batch, y_batch = train_data_loader.next_batch()
                # random exchange x1_batch and x2_batch
                if random.random() > 0.5:
                    feed = {model.input_x1: x1_batch, model.input_x2: x2_batch, model.y_data: y_batch}
                else:
                    feed = {model.input_x1: x2_batch, model.input_x2: x1_batch, model.y_data: y_batch}
                train_loss, summary, _ = sess.run([model.cost, merged, model.train_op], feed_dict=feed)
                end = time.time()
                print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(e * FLAGS.num_batches + b,
                                                                                          FLAGS.num_epochs * FLAGS.num_batches,
                                                                                          e, train_loss, end - start))
                if e % 5 == 1:
                    checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=b)
                    print('model saved to {}'.format(checkpoint_path))

                if b % 20 == 0:
                    train_writer.add_summary(summary, e * FLAGS.num_batches + b)


def minibatches(inputs=None, batch_size=None):
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt]


token_freq = defaultdict(int)

with open("./data/test_data_big.csv", 'r') as fr:
    while True:
        line = fr.readline()
        if "question" in line:
            continue
        if line:
            #  question1,question2,label
            seq1, seq2 = line.split(',')
            seq = "{} {}".format(seq1, seq2)

            for token in seq.split(' '):
                token_freq[token] += 1
        else:
            break


def text_to_array(text, is_clip=True):
    min_freq = 2
    token_list = [w for w in token_freq.keys() if token_freq[w] >= min_freq]
    token_list.append('<pad>')
    token_dictionary = {token: index for index, token in enumerate(token_list)}
    seq_ids = [int(token_dictionary.get(token)) for token in text if token_dictionary.get(token) is not None]
    if is_clip:
        seq_ids = seq_ids[:FLAGS.sequence_length]
    return seq_ids


def bi_lstm(rnn_size, layer_size, keep_prob):
    # forward rnn
    with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
        lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
        lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                       output_keep_prob=keep_prob)

    # backward rnn
    with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
        lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
        lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                       output_keep_prob=keep_prob)

    return lstm_fw_cell_m, lstm_bw_cell_m


def weight_variables(shape, name):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name=name)


def bias_variables(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


def transform_inputs(inputs, rnn_size, sequence_length):
    inputs = tf.transpose(inputs, [1, 0, 2])
    inputs = tf.reshape(inputs, [-1, rnn_size])
    inputs = tf.split(inputs, sequence_length, 0)
    return inputs


def contrastive_loss(Ew, y):
    l_1 = 0.25 * tf.square(1 - Ew)
    l_0 = tf.square(tf.maximum(Ew, 0))
    loss = tf.reduce_sum(y * l_1 + (1 - y) * l_0)
    return loss


def padding_seq(seq_array, padding_index):
    sequence_length = 30
    for i in range(len(seq_array), sequence_length):
        seq_array.append(padding_index)
    return seq_array


def test():
    rnn_size = FLAGS.rnn_size
    sequence_length = FLAGS.sequence_length
    layer_size = FLAGS.layer_size
    keep_prob = FLAGS.dropout_keep_prob
    vocab_size = len(token_freq)
    padding_index = vocab_size - 1

    input_x1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
    input_x2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_y')

    with tf.device('/cpu:0'):
        embedding = weight_variables([vocab_size, rnn_size], 'embedding')
        inputs_x1 = tf.nn.embedding_lookup(embedding, input_x1)
        inputs_x2 = tf.nn.embedding_lookup(embedding, input_x2)

    inputs_x1 = transform_inputs(inputs_x1, rnn_size, sequence_length)
    inputs_x2 = transform_inputs(inputs_x2, rnn_size, sequence_length)

    with tf.variable_scope('output'):
        bilstm_fw, bilstm_bw = bi_lstm(rnn_size, layer_size, keep_prob)
        outputs_x1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(bilstm_fw, bilstm_bw, inputs_x1, dtype=tf.float32)
        output_x1 = tf.reduce_mean(outputs_x1, 0)
        tf.get_variable_scope().reuse_variables()
        outputs_x2, _, _ = tf.contrib.rnn.static_bidirectional_rnn(bilstm_fw, bilstm_bw, inputs_x2, dtype=tf.float32)
        output_x2 = tf.reduce_mean(outputs_x2, 0)

    with tf.variable_scope('dense_layer'):
        fc_w1 = weight_variables([2 * rnn_size, 128], 'fc_w1')
        fc_w2 = weight_variables([2 * rnn_size, 128], 'fc_w2')

        fc_b1 = bias_variables([128], 'fc_b1')
        fc_b2 = bias_variables([128], 'fc_b2')

        logits_1 = tf.matmul(output_x1, fc_w1) + fc_b1
        logits_2 = tf.matmul(output_x2, fc_w2) + fc_b2
    # calc Energy 1,2 ..
    f_x1x2 = tf.reduce_sum(tf.multiply(logits_1, logits_2), 1)
    norm_fx1 = tf.sqrt(tf.reduce_sum(tf.square(logits_1), 1))
    norm_fx2 = tf.sqrt(tf.reduce_sum(tf.square(logits_2), 1))
    Ew = f_x1x2 / (norm_fx1 * norm_fx2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("! model load success !")

    test_data = pd.read_csv("./data/test_data_big.csv")
    test_data_src = pd.read_csv("./data/test.csv")
    y = OrderedDict()
    k = 1
    for x_batch in minibatches(test_data, batch_size=128):
        x1, x2 = x_batch['question1'].values, x_batch['question2'].values
        x1_batch = []
        x2_batch = []

        for one in x1:
            text2arr = padding_seq(text_to_array(one.split(" ")), padding_index)
            x1_batch.append(text2arr)

        for two in x2:
            text2arr = padding_seq(text_to_array(two.split(" ")), padding_index)
            x2_batch.append(text2arr)

        x1_array = np.array(x1_batch)
        x2_array = np.array(x2_batch)
        if random.random() > 0.5:
            feed = {input_x1: x1_array, input_x2: x2_array}
        else:
            feed = {input_x1: x2_array, input_x2: x1_array}

        y_pred = sess.run(Ew, feed_dict=feed)

        for x in y_pred:
            if x > 0.996:
                y[k] = 1
                k += 1
            else:
                y[k] = 0
                k += 1

    df = pd.DataFrame(y, index=[0])
    df.to_csv("./data/y.csv", index=None)
    test_data_src['label'] = y.values()
    test_data_src.to_csv("./data/predicts.csv", index=None)


def build_test():
    test = pd.read_csv("./data/test.csv")
    print(test.shape)
    test_big = pd.DataFrame()
    test_big['q1'] = test['q1'].apply(get_sentences)
    test_big['q2'] = test['q2'].apply(get_sentences)
    test_big.to_csv("./data/test_big.csv", index=None)
    print(test_big.shape)
    print(test_big.head())


build_test()
