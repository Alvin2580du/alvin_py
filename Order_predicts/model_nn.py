import os
import pandas as pd
import numpy as np
import sys
import re
import jieba
from sklearn.utils import shuffle
import tensorflow as tf

from pyduyp.logger.log import log


def Linear_Classification_model(inputs, in_size, out_size):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='w')
    biases = tf.Variable(tf.zeros([1, out_size]), name='b')
    outputs = tf.nn.softmax(tf.matmul(inputs, Weights) + biases, name='outputs')
    return outputs


def neural_networks(input_tensor, in_size, out_size):
    layer1_node = 50
    layer2_node = 20
    w1 = tf.Variable(tf.random_normal([in_size, layer1_node], stddev=1))
    b1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))
    w2 = tf.Variable(tf.random_normal([layer1_node, layer2_node], stddev=1))
    b2 = tf.Variable(tf.constant(0.1, shape=[layer2_node]))
    w3 = tf.Variable(tf.truncated_normal(shape=[layer2_node, out_size], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[out_size]))

    layer1 = tf.nn.relu(tf.matmul(input_tensor, w1) + b1)
    layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
    out = tf.nn.softmax(tf.matmul(layer2, w3) + b3)
    return out


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
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


def get_data(data_path, data_name, f_start, f_end):
    root = os.path.dirname(os.path.realpath(__file__))
    log.debug("{}".format(root))
    data_name = os.path.join(root, data_path, data_name)
    data = pd.read_csv(data_name)
    train_data_x = data.loc[:, f_start:f_end]
    train_data_x_columns_name = train_data_x.columns
    save_path = os.path.join(root, 'data_pre/columns_name_for_test.csv')
    df = pd.Series(train_data_x_columns_name)
    df.to_csv(save_path, index=None)

    train_data_x = train_data_x.loc[:len(train_data_x) - 2, :]
    shape = train_data_x.shape
    fenge = int(shape[0] * 0.9)
    train_x = train_data_x.loc[:fenge, :]
    train_y = data.loc[:fenge, "class_Normal":"class_Spam"]

    test_x = train_data_x.loc[fenge:, :]
    test_y = data.loc[fenge:len(data) - 2, "class_Normal":"class_Spam"]
    # assert len(test_x) == len(test_y)

    return train_x, train_y, test_x, test_y


def train_nnmodel(epoch, learning_rate, batch_size, data_path='datasets/results',
                  data_name="train.csv", class_number=2, checkpoint_dir="datasets/results/models"):
    root = os.path.dirname(os.path.realpath(__file__))

    data_path = os.path.join(root, data_path, data_name)
    df_ohe = pd.read_csv(data_path)
    log.info("{}".format(df_ohe.shape))
    df_ohe = shuffle(df_ohe)
    train_y = df_ohe['label']
    train_y = pd.get_dummies(train_y)

    del df_ohe['label']
    train_x = df_ohe

    x_data_holder = tf.placeholder(tf.float32, [None, train_x.shape[1]], name='inputs_x')
    y_data_holder = tf.placeholder(tf.float32, [None, class_number], name='inputs_y')
    y_prediction = neural_networks(x_data_holder, train_x.shape[1], class_number)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data_holder, logits=y_prediction))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    y_pre_max = tf.argmax(y_prediction, axis=1)  # 预测值的最大值的索引
    y_train_max = tf.argmax(y_data_holder, axis=1)  # 真实值的最大值的索引
    correct_prediction = tf.equal(y_pre_max, y_train_max)  # 返回bool值
    bool2float = tf.cast(correct_prediction, tf.float32)  # bool转float32
    accuracy = tf.reduce_mean(bool2float)  # 准确率

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)

        for e in range(epoch):
            counter = 0
            batch_count = len(train_x) // batch_size
            for batch_x, batch_y in minibatches(inputs=train_x, targets=train_y, batch_size=batch_size, shuffle=False):
                sess.run(train_step, feed_dict={x_data_holder: batch_x, y_data_holder: batch_y})
                train_loss = sess.run(loss, feed_dict={x_data_holder: batch_x, y_data_holder: batch_y})
                train_acc = sess.run(accuracy, feed_dict={x_data_holder: batch_x, y_data_holder: batch_y})
                if np.mod(counter, 10) == 1:
                    log_out = "Epoch:{} Batch Count: {}/{},  Train Accuracy: {:06f}; Loss: {:06f}"
                    log.info(log_out.format(e, counter, batch_count, train_acc, train_loss))
                counter += 1
                if np.mod(counter, 10) == 1:
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    checkpoint_name = os.path.join(root, checkpoint_dir)
                    saver.save(sess, save_path=os.path.join(checkpoint_name, "{}.model".format(counter)))
                    log.debug(" Model {} have save success ...".format(checkpoint_name))


def nntest(model_dir="datasets/results/models", class_number=2):
    root = os.path.dirname(os.path.realpath(__file__))

    pos = pd.read_csv("Order_predicts/datasets/results/test/action_pos_features.csv")
    neg = pd.read_csv("Order_predicts/datasets/results/test/action_neg_features.csv")
    data = pd.concat([pos, neg])
    data = shuffle(data)
    ids = data['id']
    data = data.fillna(-1).replace(np.inf, 100)
    del data['16_tmode']
    del data['10_t9']
    del data['28_tmode']
    del data['27_atmedian']
    del data['29_atptp']
    del data['continent']
    del data['province']
    del data['country']
    del data['city']
    del data['age']

    x_data_holder = tf.placeholder(tf.float32, [None, 33], name='inputs_x')
    y_prediction = neural_networks(x_data_holder, 33, class_number)

    for i in ids:
        batch_x = data[data['id'].isin([i])]
        del batch_x['id']
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.join(root, model_dir))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            y_pre = sess.run(y_prediction, feed_dict={x_data_holder: batch_x.values})
            log.info("{}".format(y_pre))
            normal, spam = y_pre[0][0], y_pre[0][1]
            log.info("{}, {}".format(normal, spam))
            res = {}
            if normal > spam:
                res['pos'] = normal
            elif normal < spam:
                res['neg'] = spam
            print(res)


# train_nnmodel(epoch=1, learning_rate=0.9, batch_size=100)

nntest()
