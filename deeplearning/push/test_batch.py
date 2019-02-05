# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os
from PIL import Image
import common

flags = tf.app.flags
flags.DEFINE_string('frozen_graph_path', './models/frozen_inference_graph.pb', 'path to model frozen graph.')
FLAGS = flags.FLAGS

CLASS_NUM = 2
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_CHANNEL = 3


def main(_):
    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with model_graph.as_default():
        with tf.Session(graph=model_graph) as sess:
            inputs = model_graph.get_tensor_by_name('image_tensor:0')
            classes = model_graph.get_tensor_by_name('classes:0')
            correct = 0
            files = common.get_files('./test')
            all_num = len(files)
            for f in files:
                image = Image.open(f)
                image = image.resize([INPUT_HEIGHT, INPUT_WIDTH])
                image = np.array(image, dtype=np.uint8)
                image_np = np.expand_dims(image, axis=0)
                predicted_label = sess.run(classes, feed_dict={inputs: image_np})
                if f.lower().count('true') > 0:
                    gt = 1
                else:
                    gt = 0
                if predicted_label[0] == gt:
                    correct +=1
                print('predict label {}  vs  gt  {} '.format(predicted_label[0], gt))
            print('total {}  correct {} wrong {}  rate  {}'.format(all_num, correct, all_num - correct,
                                                                   float(correct)/all_num))

if __name__ == '__main__':
    tf.app.run()