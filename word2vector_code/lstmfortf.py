from tensorflow.contrib import rnn
import tensorflow as tf

from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell


class Config:
    in_size = 32
    class_num = 2
    learning_rate = 0.001
    timesteps = 28
    num_classes = 2
    num_hidden = 128
    layer_number = 3
    keep_prob = 0.7
    hidden_size = 128
    batch_size = 16
    seq_len = 100


# X = tf.placeholder(tf.float32, [None, Config.in_size, Config.in_size])
# y = tf.placeholder(tf.float32, [None, Config.class_num])


def bilstm(char_vec):
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(Config.hidden_size, forget_bias=0.0, state_is_tuple=True)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(Config.hidden_size, forget_bias=0.0, state_is_tuple=True)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, char_vec, sequence_length=Config.seq_len,
                                                 dtype=tf.float32)
    output = tf.concat(2, outputs)
    return output


def multi_layer_lstm(inputs, label):
    def unit_lstm():
        lstm_cell = rnn.BasicLSTMCell(num_units=Config.hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=Config.keep_prob)
        return lstm_cell

    multilstm = []
    for layer in range(Config.layer_number):
        multilstm.append(unit_lstm())

    mlstm_cell = rnn.MultiRNNCell(multilstm, state_is_tuple=True)
    init_state = mlstm_cell.zero_state(Config.batch_size, dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=inputs, initial_state=init_state, time_major=False)
    h_state = outputs[:, -1, :]
    W = tf.Variable(tf.truncated_normal([Config.hidden_size, Config.class_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[Config.class_num]), dtype=tf.float32)
    logits = tf.nn.softmax(tf.matmul(h_state, W) + bias)
    loss_op = -tf.reduce_mean(label * tf.log(logits))
    train_op = tf.train.AdamOptimizer(Config.learning_rate).minimize(loss_op)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return h_state, logits, train_op, accuracy


def BiRNN(inputs, label):
    """

    :param inputs: 输入训练数据
    :param label:  训练标签
    :return: 预测值，损失函数，训练目标，准确率
    ex:python

    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    logits, loss_op, train_op, accuracy = BiRNN(inputs=X, label=Y)
    batch_x = ''
    batch_y = ''
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
    """
    weights = {'out': tf.Variable(tf.random_normal([2 * Config.num_hidden, Config.num_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([Config.num_classes]))}

    x = tf.unstack(inputs, Config.timesteps, 1)
    lstm_fw_cell = rnn.BasicLSTMCell(Config.num_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(Config.num_hidden, forget_bias=1.0)
    (outputs, output_state_fw, output_state_bw) = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                                               dtype=tf.float32)
    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']

    prediction = tf.nn.softmax(logits)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=Config.learning_rate).minimize(loss_op)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return logits, loss_op, train_op, accuracy


def RNN(inputs, label):
    weights = {'out': tf.Variable(tf.random_normal([Config.num_hidden, Config.num_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([Config.num_classes]))}

    x = tf.unstack(inputs, Config.timesteps, 1)
    lstm_cell = rnn.BasicLSTMCell(Config.num_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    prediction = tf.nn.softmax(logits)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=Config.learning_rate).minimize(loss_op)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return logits, loss_op, train_op, accuracy


def rnns(input_layer, training):
    # Three BRNNs with 0.5 dropout and 300 hidden units
    # Inputs is shape [700, 192]

    def _cell():
        cell = tf.contrib.rnn.GRUCell(num_units=300)
        if training:
            keep_prob = 0.5
        else:
            keep_prob = 1.0

        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    with tf.variable_scope("rnn_1"):
        # Outputs is a tuple (fw, bw) of [batch_size, 700, 300]
        outputs1, _ = tf.nn.bidirectional_dynamic_rnn(_cell(), _cell(), inputs=input_layer, dtype=tf.float32)

    with tf.variable_scope("rnn_2"):
        outputs2, _ = tf.nn.bidirectional_dynamic_rnn(_cell(), _cell(), inputs=tf.concat(outputs1, 2), dtype=tf.float32)

    with tf.variable_scope("rnn_3"):
        outputs3, _ = tf.nn.bidirectional_dynamic_rnn(_cell(), _cell(), inputs=tf.concat(outputs2, 2), dtype=tf.float32)

    return tf.concat(outputs3, 2)


def rnn_layer(embedded_x, hidden_size, bidirectional, cell_type='GRU', reuse=False):
    rnn_cells = {
        'GRU': tf.nn.rnn_cell.GRUCell,
        'LSTM': tf.nn.rnn_cell.BasicLSTMCell,
    }

    with tf.variable_scope('recurrent', reuse=reuse):
        cell = rnn_cells[cell_type]

        fw_rnn_cell = cell(hidden_size)

        if bidirectional:
            bw_rnn_cell = cell(hidden_size)
            (rnn_outputs, output_states) = bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, embedded_x, dtype=tf.float32)
            output = tf.concat([rnn_outputs[0], rnn_outputs[1]], axis=2)
        else:
            output, _ = tf.nn.dynamic_rnn(fw_rnn_cell,  embedded_x, dtype=tf.float32)
    return output
