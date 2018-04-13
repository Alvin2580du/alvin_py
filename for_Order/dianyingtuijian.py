import pandas as pd
import tensorflow as tf
import pickle
import re
import os
import numpy as np
from sklearn.model_selection import train_test_split

import time
import datetime

users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
users = pd.read_table('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine='python')
print(users.head(2))
movies_title = ['MovieID', 'Title', 'Genres']
movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python')
print(movies.head(2))
ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
print(ratings.head(2))


def load_data():
    """
    Load Dataset from File
    """
    # 读取User数据
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine='python')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    # 改变User数据中性别和年龄
    gender_map = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)
    users.to_csv("user.csv", index=None)
    # 读取Movie数据集
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python')
    movies_orig = movies.values
    # 将Title中的年份去掉
    pattern = re.compile(r'^(.*)\((\d+)\)$')

    title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)

    # 电影类型转数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)

    genres_set.add('<PAD>')
    genres2int = {val: ii for ii, val in enumerate(genres_set)}

    # 将电影类型转成等长数字列表，长度是18
    genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])

    movies['Genres'] = movies['Genres'].map(genres_map)

    # 电影Title转数字字典
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)

    title_set.add('<PAD>')
    title2int = {val: ii for ii, val in enumerate(title_set)}

    # 将电影Title转成等长数字列表，长度是15
    title_count = 15
    title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['Title']))}

    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])

    movies['Title'] = movies['Title'].map(title_map)
    movies.to_csv("movies.csv", index=None)
    # 读取评分数据集
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')
    ratings.to_csv("ratings.csv", index=None)
    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)
    data.to_csv("traindata.csv", index=None)
    # 将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

    features = features_pd.values
    targets_values = targets_pd.values

    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))


class Model:
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
        open('preprocess.p', mode='rb'))

    # 嵌入矩阵的维度
    embed_dim = 32
    # 用户ID个数
    uid_max = max(features.take(0, 1)) + 1  # 6040
    # 性别个数
    gender_max = max(features.take(2, 1)) + 1  # 1 + 1 = 2
    # 年龄类别个数
    age_max = max(features.take(3, 1)) + 1  # 6 + 1 = 7
    # 职业个数
    job_max = max(features.take(4, 1)) + 1  # 20 + 1 = 21

    # 电影ID个数
    movie_id_max = max(features.take(1, 1)) + 1  # 3952
    # 电影类型个数
    movie_categories_max = max(genres2int.values()) + 1  # 18 + 1 = 19
    # 电影名单词个数
    movie_title_max = len(title_set)  # 5216

    # 对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
    combiner = "sum"

    # 电影名长度
    sentences_size = title_count  # = 15
    # 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
    window_sizes = {2, 3, 4, 5}
    # 文本卷积核数量
    filter_num = 8

    # 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
    movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}
    # Number of Epochs
    num_epochs = 5
    # Batch Size
    batch_size = 256

    dropout_keep = 0.5
    # Learning Rate
    learning_rate = 0.0001
    # Show stats for every n number of batches
    show_every_n_batches = 20
    dropout_keep_prob = 0.5

    save_dir = './save'

    def get_inputs(self):
        uid = tf.placeholder(tf.int32, [None, 1], name="uid")
        user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
        user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
        user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")

        movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
        movie_categories = tf.placeholder(tf.int32, [None, 18], name="movie_categories")
        movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
        targets = tf.placeholder(tf.int32, [None, 1], name="targets")
        LearningRate = tf.placeholder(tf.float32, name="LearningRate")
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, LearningRate, dropout_keep_prob

    def get_user_embedding(self, uid, user_gender, user_age, user_job):
        with tf.name_scope("user_embedding"):
            uid_embed_matrix = tf.Variable(tf.random_uniform([self.uid_max, self.embed_dim], -1, 1),
                                           name="uid_embed_matrix")
            uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name="uid_embed_layer")

            gender_embed_matrix = tf.Variable(tf.random_uniform([self.gender_max, self.embed_dim // 2], -1, 1),
                                              name="gender_embed_matrix")
            gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name="gender_embed_layer")

            age_embed_matrix = tf.Variable(tf.random_uniform([self.age_max, self.embed_dim // 2], -1, 1),
                                           name="age_embed_matrix")
            age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")

            job_embed_matrix = tf.Variable(tf.random_uniform([self.job_max, self.embed_dim // 2], -1, 1),
                                           name="job_embed_matrix")
            job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name="job_embed_layer")
        return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer

    def get_user_feature_layer(self, uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
        with tf.name_scope("user_fc"):
            # 第一层全连接
            uid_fc_layer = tf.layers.dense(uid_embed_layer, self.embed_dim, name="uid_fc_layer", activation=tf.nn.relu)
            gender_fc_layer = tf.layers.dense(gender_embed_layer, self.embed_dim, name="gender_fc_layer",
                                              activation=tf.nn.relu)
            age_fc_layer = tf.layers.dense(age_embed_layer, self.embed_dim, name="age_fc_layer", activation=tf.nn.relu)
            job_fc_layer = tf.layers.dense(job_embed_layer, self.embed_dim, name="job_fc_layer", activation=tf.nn.relu)

            # 第二层全连接
            user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer],
                                           2)  # (?, 1, 128)
            user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)  # (?, 1, 200)

            user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
        return user_combine_layer, user_combine_layer_flat

    def get_movie_id_embed_layer(self, movie_id):
        with tf.name_scope("movie_embedding"):
            movie_id_embed_matrix = tf.Variable(tf.random_uniform([self.movie_id_max, self.embed_dim], -1, 1),
                                                name="movie_id_embed_matrix")
            movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name="movie_id_embed_layer")
        return movie_id_embed_layer

    def get_movie_categories_layers(self, movie_categories):
        with tf.name_scope("movie_categories_layers"):
            movie_categories_embed_matrix = tf.Variable(
                tf.random_uniform([self.movie_categories_max, self.embed_dim], -1, 1),
                name="movie_categories_embed_matrix")
            movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories,
                                                                  name="movie_categories_embed_layer")
            if self.combiner == "sum":
                movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
        #     elif combiner == "mean":

        return movie_categories_embed_layer

    def get_movie_cnn_layer(self, movie_titles):
        # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
        with tf.name_scope("movie_embedding"):
            movie_title_embed_matrix = tf.Variable(tf.random_uniform([self.movie_title_max, self.embed_dim], -1, 1),
                                                   name="movie_title_embed_matrix")
            movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                             name="movie_title_embed_layer")
            movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

        # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
        pool_layer_lst = []
        for window_size in self.window_sizes:
            with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
                filter_weights = tf.Variable(
                    tf.truncated_normal([window_size, self.embed_dim, 1, self.filter_num], stddev=0.1),
                    name="filter_weights")
                filter_bias = tf.Variable(tf.constant(0.1, shape=[self.filter_num]), name="filter_bias")

                conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID",
                                          name="conv_layer")
                relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

                maxpool_layer = tf.nn.max_pool(relu_layer, [1, self.sentences_size - window_size + 1, 1, 1],
                                               [1, 1, 1, 1],
                                               padding="VALID", name="maxpool_layer")
                pool_layer_lst.append(maxpool_layer)

        # Dropout层
        with tf.name_scope("pool_dropout"):
            pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
            max_num = len(self.window_sizes) * self.filter_num
            pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")

            dropout_layer = tf.nn.dropout(pool_layer_flat, self.dropout_keep_prob, name="dropout_layer")
        return pool_layer_flat, dropout_layer

    def get_movie_feature_layer(self, movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
        with tf.name_scope("movie_fc"):
            # 第一层全连接
            movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, self.embed_dim, name="movie_id_fc_layer",
                                                activation=tf.nn.relu)
            movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, self.embed_dim,
                                                        name="movie_categories_fc_layer", activation=tf.nn.relu)

            # 第二层全连接
            movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer],
                                            2)  # (?, 1, 96)
            movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)  # (?, 1, 200)

            movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
        return movie_combine_layer, movie_combine_layer_flat


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


def build_model():
    modle = Model()
    tf.reset_default_graph()
    train_graph = tf.Graph()
    with train_graph.as_default():
        # 获取输入占位符
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob = modle.get_inputs()
        # 获取User的4个嵌入向量
        uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = modle.get_user_embedding(uid,
                                                                                                         user_gender,
                                                                                                         user_age,
                                                                                                         user_job)
        # 得到用户特征
        user_combine_layer, user_combine_layer_flat = modle.get_user_feature_layer(uid_embed_layer, gender_embed_layer,
                                                                                   age_embed_layer, job_embed_layer)
        # 获取电影ID的嵌入向量
        movie_id_embed_layer = modle.get_movie_id_embed_layer(movie_id)
        # 获取电影类型的嵌入向量
        movie_categories_embed_layer = modle.get_movie_categories_layers(movie_categories)
        # 获取电影名的特征向量
        pool_layer_flat, dropout_layer = modle.get_movie_cnn_layer(movie_titles)
        # 得到电影特征
        movie_combine_layer, movie_combine_layer_flat = modle.get_movie_feature_layer(movie_id_embed_layer,
                                                                                      movie_categories_embed_layer,
                                                                                      dropout_layer)
        # 计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
        with tf.name_scope("inference"):
            # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
            #         inference_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)  #(?, 200)
            #         inference = tf.layers.dense(inference_layer, 1,
            #                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            #                                     kernel_regularizer=tf.nn.l2_loss, name="inference")
            # 简单的将用户特征和电影特征做矩阵乘法得到一个预测评分
            #        inference = tf.matmul(user_combine_layer_flat, tf.transpose(movie_combine_layer_flat))
            inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
            inference = tf.expand_dims(inference, axis=1)

        with tf.name_scope("loss"):
            # MSE损失，将计算值回归到评分
            cost = tf.losses.mean_squared_error(targets, inference)
            loss = tf.reduce_mean(cost)
        # 优化损失
        #     train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(lr)
        gradients = optimizer.compute_gradients(loss)  # cost
        train_op = optimizer.apply_gradients(gradients, global_step=global_step)

    losses = {'train': [], 'test': []}

    with tf.Session(graph=train_graph) as sess:

        # 搜集数据给tensorBoard用
        # Keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in gradients:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')),
                                                     tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Inference summaries
        inference_summary_op = tf.summary.merge([loss_summary])
        inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
        inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch_i in range(modle.num_epochs):

            # 将数据集分成训练集和测试集，随机种子不固定
            train_X, test_X, train_y, test_y = train_test_split(modle.features,
                                                                modle.targets_values,
                                                                test_size=0.2,
                                                                random_state=0)

            train_batches = get_batches(train_X, train_y, modle.batch_size)
            test_batches = get_batches(test_X, test_y, modle.batch_size)

            # 训练的迭代，保存训练损失
            for batch_i in range(len(train_X) // modle.batch_size):
                x, y = next(train_batches)

                categories = np.zeros([modle.batch_size, 18])
                for i in range(modle.batch_size):
                    categories[i] = x.take(6, 1)[i]

                titles = np.zeros([modle.batch_size, modle.sentences_size])
                for i in range(modle.batch_size):
                    titles[i] = x.take(5, 1)[i]

                feed = {
                    uid: np.reshape(x.take(0, 1), [modle.batch_size, 1]),
                    user_gender: np.reshape(x.take(2, 1), [modle.batch_size, 1]),
                    user_age: np.reshape(x.take(3, 1), [modle.batch_size, 1]),
                    user_job: np.reshape(x.take(4, 1), [modle.batch_size, 1]),
                    movie_id: np.reshape(x.take(1, 1), [modle.batch_size, 1]),
                    movie_categories: categories,  # x.take(6,1)
                    movie_titles: titles,  # x.take(5,1)
                    targets: np.reshape(y, [modle.batch_size, 1]),
                    dropout_keep_prob: modle.dropout_keep,
                    lr: modle.learning_rate}

                step, train_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed)  # cost
                losses['train'].append(train_loss)
                train_summary_writer.add_summary(summaries, step)  #

                # Show every <show_every_n_batches> batches
                if (epoch_i * (len(train_X) // modle.batch_size) + batch_i) % modle.show_every_n_batches == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        time_str, epoch_i, batch_i, (len(train_X) // modle.batch_size), train_loss))

            # 使用测试数据的迭代
            for batch_i in range(len(test_X) // modle.batch_size):
                x, y = next(test_batches)

                categories = np.zeros([modle.batch_size, 18])
                for i in range(modle.batch_size):
                    categories[i] = x.take(6, 1)[i]

                titles = np.zeros([modle.batch_size, modle.sentences_size])
                for i in range(modle.batch_size):
                    titles[i] = x.take(5, 1)[i]

                feed = {
                    uid: np.reshape(x.take(0, 1), [modle.batch_size, 1]),
                    user_gender: np.reshape(x.take(2, 1), [modle.batch_size, 1]),
                    user_age: np.reshape(x.take(3, 1), [modle.batch_size, 1]),
                    user_job: np.reshape(x.take(4, 1), [modle.batch_size, 1]),
                    movie_id: np.reshape(x.take(1, 1), [modle.batch_size, 1]),
                    movie_categories: categories,  # x.take(6,1)
                    movie_titles: titles,  # x.take(5,1)
                    targets: np.reshape(y, [modle.batch_size, 1]),
                    dropout_keep_prob: 1,
                    lr: modle.learning_rate}

                step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op], feed)  # cost

                # 保存测试损失
                losses['test'].append(test_loss)
                inference_summary_writer.add_summary(summaries, step)  #

                time_str = datetime.datetime.now().isoformat()
                if (epoch_i * (len(test_X) // modle.batch_size) + batch_i) % modle.show_every_n_batches == 0:
                    print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                        time_str,
                        epoch_i,
                        batch_i,
                        (len(test_X) // modle.batch_size),
                        test_loss))

        # Save Model
        saver.save(sess, modle.save_dir)  # , global_step=epoch_i
        print('Model Trained and Saved')


class Gifts:
    model = Model()
    load_dir = 'save'

    def get_tensors(self, loaded_graph):
        uid = loaded_graph.get_tensor_by_name("uid:0")
        user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
        user_age = loaded_graph.get_tensor_by_name("user_age:0")
        user_job = loaded_graph.get_tensor_by_name("user_job:0")
        movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
        movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
        movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
        targets = loaded_graph.get_tensor_by_name("targets:0")
        dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")
        lr = loaded_graph.get_tensor_by_name("LearningRate:0")
        # 两种不同计算预测评分的方案使用不同的name获取tensor inference
        #     inference = loaded_graph.get_tensor_by_name("inference/inference/BiasAdd:0")
        inference = loaded_graph.get_tensor_by_name(
            "inference/ExpandDims:0")  # 之前是MatMul:0 因为inference代码修改了 这里也要修改 感谢网友 @清歌 指出问题
        movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc/Reshape:0")
        user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
        return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat

    def rating_movie(self, user_id_val, movie_id_val):
        loaded_graph = tf.Graph()  #
        with tf.Session(graph=loaded_graph) as sess:  #
            # Load saved model
            loader = tf.train.import_meta_graph(self.load_dir + '.meta')
            loader.restore(sess, self.load_dir)

            # Get Tensors from loaded model
            uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, _, __ = self.get_tensors(
                loaded_graph)  # loaded_graph

            categories = np.zeros([1, 18])
            categories[0] = movies.values[self.model.movieid2idx[movie_id_val]][2]

            titles = np.zeros([1, self.model.sentences_size])
            titles[0] = movies.values[self.model.movieid2idx[movie_id_val]][1]

            feed = {
                uid: np.reshape(users.values[user_id_val - 1][0], [1, 1]),
                user_gender: np.reshape(users.values[user_id_val - 1][1], [1, 1]),
                user_age: np.reshape(users.values[user_id_val - 1][2], [1, 1]),
                user_job: np.reshape(users.values[user_id_val - 1][3], [1, 1]),
                movie_id: np.reshape(movies.values[self.model.movieid2idx[movie_id_val]][0], [1, 1]),
                movie_categories: categories,  # x.take(6,1)
                movie_titles: titles,  # x.take(5,1)
                dropout_keep_prob: 1}
            # Get Prediction
            inference_val = sess.run([inference], feed)
            return (inference_val)

    def modle_test(self):
        loaded_graph = tf.Graph()
        self.movie_matrics = []
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(self.load_dir + '.meta')
            loader.restore(sess, self.load_dir)

            # Get Tensors from loaded model
            uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, \
            lr, dropout_keep_prob, _, movie_combine_layer_flat, __ = self.get_tensors(loaded_graph)  # loaded_graph
            print(type(movie_combine_layer_flat), movie_combine_layer_flat)
            for item in movies.values:
                categories = np.zeros([1, 18])
                categories[0] = item.take(2)

                titles = np.zeros([1, self.model.sentences_size])
                titles[0] = item.take(1)

                feed = {
                    movie_id: np.reshape(item.take(0), [1, 1]),
                    movie_categories: categories,  # x.take(6,1)
                    movie_titles: titles,  # x.take(5,1)
                    dropout_keep_prob: 1}

                movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat], feed)
                self.movie_matrics.append(movie_combine_layer_flat_val)

        pickle.dump((np.array(self.movie_matrics).reshape(-1, 200)), open('movie_matrics.p', 'wb'))

    def gift1(self, ):
        loaded_graph = tf.Graph()  #
        users_matrics = []
        with tf.Session(graph=loaded_graph) as sess:  #
            # Load saved model
            loader = tf.train.import_meta_graph(self.load_dir + '.meta')
            loader.restore(sess, self.load_dir)

            # Get Tensors from loaded model
            uid, user_gender, user_age, user_job, movie_id, movie_categories, \
            movie_titles, targets, lr, dropout_keep_prob, _, __, user_combine_layer_flat = self.get_tensors(
                loaded_graph)  # loaded_graph
            print(type(user_combine_layer_flat), user_combine_layer_flat)
            for item in users.values:
                feed = {
                    uid: np.reshape(item.take(0), [1, 1]),
                    user_gender: np.reshape(item.take(1), [1, 1]),
                    user_age: np.reshape(item.take(2), [1, 1]),
                    user_job: np.reshape(item.take(3), [1, 1]),
                    dropout_keep_prob: 1}

                user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)
                users_matrics.append(user_combine_layer_flat_val)

        pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('users_matrics.p', 'wb'))

    def recommend_same_type_movie(self, movie_id_val, top_k=20):
        loaded_graph = tf.Graph()  #
        with tf.Session(graph=loaded_graph) as sess:  #
            # Load saved model
            loader = tf.train.import_meta_graph(self.load_dir + '.meta')
            loader.restore(sess, self.load_dir)

            norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(self.movie_matrics), 1, keep_dims=True))
            normalized_movie_matrics = self.movie_matrics / norm_movie_matrics
            # 推荐同类型的电影
            probs_embeddings = (self.movie_matrics[self.model.movieid2idx[movie_id_val]]).reshape([1, 200])
            probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
            sim = (probs_similarity.eval())
            print("您看的电影是：{}".format(movies_orig[self.model.movieid2idx[movie_id_val]]))
            print("以下是给您的推荐：")
            p = np.squeeze(sim)
            p[np.argsort(p)[:-top_k]] = 0
            p = p / np.sum(p)
            results = set()
            while len(results) != 5:
                c = np.random.choice(3883, 1, p=p)[0]
                results.add(c)
            for val in (results):
                print(val)
                print(movies_orig[val])
            return results

    def recommend_your_favorite_movie(self, user_id_val, top_k=10):
        users_matrics = pickle.load(open('users_matrics.p', mode='rb'))
        movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))

        loaded_graph = tf.Graph()  #
        with tf.Session(graph=loaded_graph) as sess:  #
            # Load saved model
            loader = tf.train.import_meta_graph(self.load_dir + '.meta')
            loader.restore(sess, self.load_dir)

            # 推荐您喜欢的电影
            probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])

            probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
            sim = (probs_similarity.eval())
            #     print(sim.shape)
            #     results = (-sim[0]).argsort()[0:top_k]
            #     print(results)

            #     sim_norm = probs_norm_similarity.eval()
            #     print((-sim_norm[0]).argsort()[0:top_k])

            print("以下是给您的推荐：")
            p = np.squeeze(sim)
            p[np.argsort(p)[:-top_k]] = 0
            p = p / np.sum(p)
            results = set()
            while len(results) != 5:
                c = np.random.choice(3883, 1, p=p)[0]
                results.add(c)
            for val in (results):
                print(val)
                print(movies_orig[val])

            return results


if __name__ == '__main__':
    import sys
    method = sys.argv[1]

    if method == 'load_data':
        title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()

        pickle.dump((title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data,
                     movies_orig, users_orig), open('preprocess.p', 'wb'))

    if method == 'build_model':
        build_model()

    if method == 'test':
        model = Gifts()
        model.gift1()
        model.modle_test()
    if method == 'gifts':
        model = Gifts()
        model.recommend_your_favorite_movie(234, 10)
