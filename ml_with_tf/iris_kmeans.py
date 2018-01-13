import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

import sys

f = open("log", 'w')
sys.stdout = f


def make_iris():
    iris = datasets.load_iris()
    x = pd.DataFrame(iris.data)
    x.to_csv("iris_x.csv", sep=',', header=None, index=None)


# 计算类内平均值函数
def bucket_mean(data, bucket_ids, num_buckets):
    # 第一个参数是tensor，第二个参数是簇标签，第三个是簇数目
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total / count


def f():
    sample_number = 150
    variables = 4
    kluster_number = 3
    MAX_ITERS = 10000
    centers = [(1, 1), (2, 2), (3, 3)]

    data = pd.read_csv("iris_x.csv", header=None).values

    points = tf.Variable(data)

    cluster_assignments = tf.Variable(tf.zeros([sample_number], dtype=tf.int64))
    centroids = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [kluster_number, variables]))

    rep_centroids = tf.reshape(tf.tile(centroids, [sample_number, 1]), [sample_number, kluster_number, variables])
    rep_points = tf.reshape(tf.tile(points, [1, kluster_number]), [sample_number, kluster_number, variables])

    distance = tf.square(rep_points - rep_centroids)

    sum_squares = tf.sqrt(tf.reduce_sum(distance, reduction_indices=2))
    best_centroids = tf.argmin(sum_squares, 1)
    did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids, cluster_assignments))
    means = bucket_mean(points, best_centroids, kluster_number)

    with tf.control_dependencies([did_assignments_change]):
        do_updates = tf.group(centroids.assign(means), cluster_assignments.assign(best_centroids))

    changed = True
    iters = 0

    fig, ax = plt.subplots()
    colourindexes = [2, 1, 4]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while changed and iters < MAX_ITERS:
            fig, ax = plt.subplots()
            iters += 1
            [changed, _] = sess.run([did_assignments_change, do_updates])
            [centers, assignments] = sess.run([centroids, cluster_assignments])
            ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1], marker='8', s=200,
                       c=assignments)
            ax.scatter(centers[:, 0], centers[:, 1], marker='^', s=550, c=colourindexes)
            ax.set_title('Iteration ' + str(iters))
            plt.savefig("kmeans_" + str(iters) + ".png")

        ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1], marker='o', s=200, c=assignments)

        plt.savefig("s.png")


if __name__ == "__main__":
    f()
