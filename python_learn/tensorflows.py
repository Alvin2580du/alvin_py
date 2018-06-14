
import tensorflow as tf
import numpy as np

input_ids = tf.placeholder(dtype=tf.int32, shape=[3, 2])

embedding = tf.Variable(np.ones(5, dtype=np.int32))

input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(embedding.eval())
# print(sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]}))
print()
res = sess.run(input_embedding, feed_dict={input_ids: [[1, 2], [2, 1], [3, 3]]})
print(res.shape)
print(res)
