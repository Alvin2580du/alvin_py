import random
import collections
import zipfile
import tensorflow as tf
import itertools
import os
import pandas as pd
from tqdm import tqdm
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import re
from collections import Counter

from pyduyp.logger.log import log
from pyduyp.preprocessing.Zhsegment import cut, cutpro
from pyduyp.utils.utils import replace_symbol

root_path = os.path.dirname(os.path.realpath(__file__))


def skipgrams(sequence, vocabulary_size, window_size=4, negative_samples=1., shuffle=True,
              categorical=False, sampling_table=None, seed=None):
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:
                    labels.append([0, 1])
                else:
                    labels.append(1)

    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [[words[i % len(words)],
                     random.randint(1, vocabulary_size - 1)] for i in range(num_negative_samples)]
        if categorical:
            labels += [[1, 0]] * num_negative_samples
        else:
            labels += [0] * num_negative_samples

    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def make_closing(base, **attrs):
    if not hasattr(base, '__enter__'):
        attrs['__enter__'] = lambda self: self
    if not hasattr(base, '__exit__'):
        attrs['__exit__'] = lambda self, type, value, traceback: self.close()
    return type('Closing' + base.__name__, (base, object), attrs)


def smart_open(fname, mode='rb'):
    _, ext = os.path.splitext(fname)
    if ext == '.bz2':
        from bz2 import BZ2File
        return make_closing(BZ2File)(fname, mode)
    if ext == '.gz':
        from gzip import GzipFile
        return make_closing(GzipFile)(fname, mode)
    return open(fname, mode)


if sys.version_info[0] >= 3:
    unicode = str


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


to_unicode = any2unicode


class TextBatch(object):
    """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""
    def __init__(self, fname, max_sentence_length):
        self.fname = fname
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        sentence, rest = [], b''
        linenumber, t0 = 0, time.time()
        with smart_open(self.fname) as fin:
            while True:
                text = rest + fin.readline()
                if text == rest:  # EOF
                    words = to_unicode(text).split()
                    for word in words:
                        if str(word).isdigit():
                            continue
                        sentence.extend(word)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')
                words, rest = (
                    to_unicode(text[:last_token]).split(), text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= self.max_sentence_length:
                    yield sentence[:self.max_sentence_length]
                    linenumber += 1
                    sentence = sentence[self.max_sentence_length:]
                    # log.info("{}, {}".format(linenumber, time.time()-t0))


def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data


class LineSentence(object):
    MAX_WORDS_IN_BATCH = 1000

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        try:
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with smart_open(self.source) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i: i + self.max_sentence_length]
                        i += self.max_sentence_length


def cut_data():
    out = []
    data_name = os.path.join(root_path, 'datasets/cd_by_nosplit.txt')
    with open(data_name, 'r') as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            line_cut = cut(replace_symbol(line), add_stopwords=True)
            for x in line_cut:
                out.append(x)
    log.info(" Length: {} ".format(len(out)))
    fw = open(os.path.join(root_path, "datasets/cd.txt"), 'w')
    fw.writelines(" ".join(out))
    fw.close()


def plot_embedding(embeddings_file):
    embeddings_file = sys.argv[2]
    wv, vocabulary = load_embeddings(embeddings_file)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[:1000, :])

    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.savefig("结果.png")


def load_embeddings(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in])
    wv = np.loadtxt(wv)
    return wv, vocabulary


def read_newdata():
    path = '/home/duyp/mayi_datasets/question/question_new'
    number = 0

    for file in os.listdir(path):
        filename = os.path.join(path, file)
        data = pd.read_csv(filename, lineterminator="\n").values
        out = []
        for x in tqdm(data):
            msg = x[0]
            if isinstance(msg, str):
                msgcut = cut(replace_symbol(msg), add_stopwords=True)
                for i in msgcut:
                    out.append(i)
                    number += 1
            else:
                continue
        fw = open(os.path.join(root_path, "datasets/rawdata/{}".format(file)), 'w')
        fw.writelines(" ".join(out))
        fw.close()
    log.info("{}".format(number))


def conact_csv():
    out = []
    path = 'word2vector_code/datasets/rawdata'
    for file in tqdm(os.listdir(path)):
        data_name = os.path.join(path, file)
        with open(data_name, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                for x in line.split():
                    if x.isdigit():
                        continue
                    out.append(x)
        log.info(" NEXT ")
    log.info("{}".format(len(out)))

    fw = open("word2vector_code/datasets/train.csv", 'w')
    fw.writelines(" ".join(out))


def getfreq():
    out = {}
    with open('word2vector_code/datasets/results/vector/vectors_question_8700000_2000000.csv', 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            word = line.split()[0]
            out[word] = 0
    with open("word2vector_code/datasets/rawdata/question_8700000_2000000.csv", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            for x in line.split():
                if x in out.keys():
                    out[x] += 1


def getdict():
    path = '/home/duyp/mayi_datasets/question/question_new'
    out = []

    for file in os.listdir(path):
        filename = os.path.join(path, file)
        data = pd.read_csv(filename, lineterminator="\n").values
        for x in tqdm(data):
            msg = x[0]
            res = cutpro(msg)
            for word in res:
                out.append(word)

    fw = open("word2vector_code/datasets/datesdict.csv", 'a+')
    for i, j in Counter(out).most_common(100000):
        fw.writelines("{}:{}".format(i, j)+"\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise Exception("[!] You should put more args")
    method = sys.argv[1]

    if method == 'cut':
        cut_data()
        log.info(" ! Build Success ! ")

    if method == 'train':
        import logging

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        max_words = 10000
        data_name = 'word2vector_code/datasets/train.csv'
        sentences = TextBatch(fname=data_name, max_sentence_length=max_words)
        model = word2vec.Word2Vec(sentences, iter=5, workers=4, size=1000, min_count=1, alpha=0.025,
                                  window=3, max_vocab_size=None, sample=1e-3, seed=1, min_alpha=0.0001,
                                  sg=1, hs=0, negative=5, cbow_mean=1, hashfxn=hash, null_word=0,
                                  trim_rule=None, sorted_vocab=1, batch_words=max_words,
                                  compute_loss=False)
        logging.info(model)
        save_dir = 'word2vector_code/datasets/results/vector'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        total_path = os.path.join(root_path, "datasets/results/vector/vectors_train.csv")
        fvocab_path = os.path.join(root_path, "datasets/results/vector/fvocab_train.csv")

        log.debug("save path : {}".format(total_path))
        model.wv.save_word2vec_format(total_path, fvocab=fvocab_path, binary=False)
        model.save(root_path + "/datasets/train.model")
        log.info(" ! Build Success ! ")

    if method == 'test':
        # distance = model.wmdistance(sentence_obama, sentence_president)
        total_path = os.path.join(root_path, "datasets/results/vector/vectors_traintest.csv")
        word_vectors = KeyedVectors.load_word2vec_format(total_path, binary=False)
        with open("word2vector_code/datasets/results/vector/fvocab_traintest.csv", 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                word = line.split(" ")[0]
                similar = word_vectors.most_similar(positive=word, topn=10)
                print("{}, {}".format(word, similar))
                print("= " * 20)

        log.info(" ! Build Success ! ")

    if method == 'plot':
        plot_embedding()

    if method == 'newdata':
        read_newdata()

    if method == 'conact_csv':
        conact_csv()

    if method == 'getfreq':
        getfreq()

    if method == 'getdict':
        getdict()