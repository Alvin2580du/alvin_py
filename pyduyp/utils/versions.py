# code=utf-8

import nltk
import sklearn
import tensorflow as tf
import pandas
import keras
import xgboost
import numpy

import jieba
import gensim

import flask


def version():
    print('The nltk version is {}.'.format(nltk.__version__))
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    print('The tensorflow version is {}.'.format(tf.__version__))
    print('The pandas version is {}'.format(pandas.__version__))
    print('The keras version is {}.'.format(keras.__version__))
    print('The xgboost version is {}.'.format(xgboost.__version__))
    print('The numpy version is {}.'.format(numpy.__version__))

    print('The jieba version is {}'.format(jieba.__version__))
    print('The gensim version is {}'.format(gensim.__version__))
    #print('The minepy version is {}'.format(minepy.__version__))

    print('The flash version is {}'.format(flask.__version__))

