import sys
import tensorflow as tf
import os
import pandas as pd
import numpy as np


def compute_labels_weights(weights_label, logits, labels):
    labels_predicts = np.argmax(logits, axis=1)
    for i in range(len(labels)):
        label = labels[i]
        label_predict = labels_predicts[i]
        weight = weights_label.get(label, None)

        if weight is None:
            if label_predict == label:
                weights_label[label] = (1, 1)
            else:
                weights_label[label] = (1, 0)
        else:
            number = weight[0]
            correct = weight[1]
            number = number + 1
            if label_predict == label:
                correct = correct + 1
            weights_label[label] = (number, correct)
    return weights_label


def get_weights_for_current_batch(answer_list, weights_dict):
    weights_list_batch = list(np.ones((len(answer_list))))
    answer_list = list(answer_list)
    for i, label in enumerate(answer_list):
        acc = weights_dict[label]
        weights_list_batch[i] = min(1.5, 1.0 / (acc + 0.001))
    return weights_list_batch


def loss(logits, labels, weights):
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, weights=weights)
    return loss


def get_weights_label_as_standard_dict(weights_label):
    weights_dict = {}
    for k, v in weights_label.items():
        count, correct = v
        weights_dict[k] = float(correct) / float(count)
    return weights_dict
