import random
import jieba
import collections
import zipfile
import tensorflow as tf
import itertools
import os
import sys
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

root_path = os.path.dirname(os.path.realpath(__file__))

jieba.load_userdict(os.path.join(root_path, "datasets/jieba_dict_sorted.csv"))


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
    MAX_WORDS_IN_BATCH = 100000

    def __init__(self, fname, max_sentence_length=MAX_WORDS_IN_BATCH):
        self.fname = fname
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest = [], b''
        with smart_open(self.fname) as fin:
            while True:
                text = rest + fin.readline()  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    words = to_unicode(text).split()
                    sentence.extend(words)  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration
                words, rest = (
                    to_unicode(text[:last_token]).split(), text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= self.max_sentence_length:
                    yield sentence[:self.max_sentence_length]
                    sentence = sentence[self.max_sentence_length:]


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
    out = ""
    data_name = os.path.join(root_path, 'datasets/test')
    with open(data_name, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line_cut = jieba.lcut(line)
            out += " ".join(line_cut)
    fw = open(os.path.join(root_path, "datasets/cd_test.txt"), 'w')
    fw.writelines(out)
    fw.close()


if __name__ == "__main__":
    import sys
    method = sys.argv[1]

    if method == 'train':
        data_name = os.path.join(root_path, "datasets/cd.txt")
        sentences = TextBatch(fname=data_name, max_sentence_length=6)
        model = word2vec.Word2Vec(sentences, iter=10, min_count=4, size=50, workers=4)
        total_path = os.path.join(root_path, "datasets/cd_test_vectors.txt")
        model.wv.save_word2vec_format(total_path, binary=False)
        model.save(root_path + "/datasets/cd_model")

    if method == 'test':
        total_path = os.path.join(root_path, "datasets/cd_test_vectors.txt")
        word_vectors = KeyedVectors.load_word2vec_format(total_path, binary=False)
        model = word2vec.Word2Vec.load(root_path + "/datasets/cd_model")
        print(model.wv['方便接待'])
        print(model.wv.similarity('方便接待', '我们'), model.wv.similarity('方便接待', '旅游'))

