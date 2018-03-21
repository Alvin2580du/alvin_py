from tensorflow.contrib.factorization import KMeans
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

import numpy as np
import pandas as pd
import tensorflow as tf


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    if targets is not None:
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
        if targets is not None:
            yield inputs[excerpt], targets[excerpt]
        else:
            yield inputs[excerpt]


def train_Kmeans():
    data = pd.read_csv("./localdatasets/newvector.csv", header=None)
    name = data[0]
    del data[0]
    full_data_x = data.values

    num_steps = 50  # Total steps to train
    batch_size = 1024  # The number of samples per batch
    k = 25  # The number of clusters
    num_classes = 10  # The 10 digits
    num_features = 1000  # Each image is 28x28 pixels

    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # Labels (for assigning a label to a centroid and testing)
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])

    # K-Means Parameters
    kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine', use_mini_batch=True)
    training_graph = kmeans.training_graph()

    if len(training_graph) > 6:  # Tensorflow 1.4+
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         cluster_centers_var, init_op, train_op) = training_graph
    else:
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         init_op, train_op) = training_graph

    cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
    avg_distance = tf.reduce_mean(scores)

    # Initialize the variables (i.e. assign their default value)
    init_vars = tf.global_variables_initializer()

    # Start TensorFlow session
    sess = tf.Session()

    # Run the initializer
    sess.run(init_vars, feed_dict={X: full_data_x})
    sess.run(init_op, feed_dict={X: full_data_x})
    saver = tf.train.Saver()
    # Training
    for i in range(1, num_steps + 1):
        for batch_x in minibatches(inputs=full_data_x, batch_size=batch_size):
            _, d, idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={X: batch_x})
            if i % 10 == 0 or i == 1:
                print("Step %i, Avg Distance: %f" % (i, d))

            if i % 30 == 0 or i == 50:
                checkpoint_dir = './localdatasets'
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i + 1)


def RandForest():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("./localdatasets/mnist/", one_hot=False)

    # Parameters
    num_steps = 500  # Total steps to train
    batch_size = 1024  # The number of samples per batch
    num_classes = 10  # The 10 digits
    num_features = 784  # Each image is 28x28 pixels
    num_trees = 10
    max_nodes = 1000

    # Input and Target data
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # For random forest, labels must be integers (the class id)
    Y = tf.placeholder(tf.int32, shape=[None])

    # Random Forest Parameters
    hparams = tensor_forest.ForestHParams(num_classes=num_classes, num_features=num_features,
                                          num_trees=num_trees, max_nodes=max_nodes).fill()

    # Build the Random Forest
    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    # Get training graph and loss
    train_op = forest_graph.training_graph(X, Y)
    loss_op = forest_graph.training_loss(X, Y)

    # Measure the accuracy
    infer_op, _, _ = forest_graph.inference_graph(X)
    correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize the variables (i.e. assign their default value) and forest resources
    init_vars = tf.group(tf.global_variables_initializer(),
                         resources.initialize_resources(resources.shared_resources()))

    # Start TensorFlow session
    sess = tf.Session()

    # Run the initializer
    sess.run(init_vars)

    # Training
    for i in range(1, num_steps + 1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        if i % 50 == 0 or i == 1:
            acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
            print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

