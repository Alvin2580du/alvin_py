# Segmentation Examples
import tensorflow as tf

sess = tf.InteractiveSession()
seg_ids = tf.constant([0, 1, 1, 2, 2])  # Group indexes : 0|1,2|3,4
print seg_ids.get_shape().as_list()
print
tens1 = tf.constant([[2, 5, 3, -5],
                     [0, 3, -2, 5],
                     [4, 3, 5, 3],
                     [6, 1, 4, 0],
                     [6, 1, 4, 0]])  # A sample constant matrix
print tens1.get_shape().as_list()
print
print tf.segment_sum(tens1, seg_ids).eval()  # Sum segmentation
print tf.segment_prod(tens1, seg_ids).eval().shape  # Product segmantation
print tf.segment_min(tens1, seg_ids).eval()  # minimun value goes to group
print tf.segment_max(tens1, seg_ids).eval()  # maximum value goes to group
print tf.segment_mean(tens1, seg_ids).eval()  # mean value goes to group
