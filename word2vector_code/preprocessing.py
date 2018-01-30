import random
import jieba
import jieba.analyse
import collections
import zipfile
import tensorflow as tf
import itertools
import os
import sys
import pandas as pd
from tqdm import tqdm
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

from pyduyp.logger.log import log

root_path = os.path.dirname(os.path.realpath(__file__))

jieba.load_userdict(os.path.join(root_path, "datasets/jieba_dict_sorted.csv"))

jieba.analyse.set_stop_words(os.path.join(root_path, 'datasets/stopwords_zh.csv'))
sw = pd.read_csv(os.path.join(root_path, 'datasets/stopwords_zh.csv'), lineterminator="\n").values.tolist()
sw2list = [j for i in sw for j in i]


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
    out = []
    data_name = os.path.join(root_path, 'datasets/cd_by_nosplit.txt')
    with open(data_name, 'r') as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            line_cut = jieba.lcut(line)
            for x in line_cut:
                if x not in sw2list:
                    out.append(x)
    log.info(" Length: {} ".format(len(out)))
    fw = open(os.path.join(root_path, "datasets/cd.txt"), 'w')
    fw.writelines(" ".join(out))
    fw.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise Exception("[!] You should put more args")
    method = sys.argv[1]

    if method == 'cut':
        cut_data()
        log.info(" ! Build Success ! ")

    if method == 'train':
        MAX_WORDS_IN_BATCH = 10000

        data_name = os.path.join(root_path, "datasets/cd.txt")
        sentences = TextBatch(fname=data_name, max_sentence_length=1000)
        model = word2vec.Word2Vec(sentences, size=128, alpha=0.025, window=5, min_count=5,
                                  max_vocab_size=1024, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                                  sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=100, null_word=0,
                                  trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False)
        """
        ·  sentences：可以是一个·ist，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
        ·  sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
        ·  size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
        ·  window：表示当前词与预测词在一个句子中的最大距离是多少
        ·  alpha: 是学习速率
        ·  seed：用于随机数发生器。与初始化词向量有关。
        ·  min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
        ·  max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。
                           每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
        ·  sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
        ·  workers: 参数控制训练的并行数。
        ·  hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
        ·  negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
        ·  cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（defau·t）则采用均值。只有使用CBOW的时候才起作用。
        ·  hashfxn： hash函数来初始化权重。默认使用python的hash函数
        ·  iter： 迭代次数，默认为5
        ·  trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。
                        可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
        ·  sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
        ·  batch_words：每一批的传递给线程的单词的数量，默认为10000
        """
        total_path = os.path.join(root_path, "datasets/cd_vectors.txt")
        model.wv.save_word2vec_format(total_path, binary=False)
        model.save(root_path + "/datasets/cd.model")
        log.info(" ! Build Success ! ")

    if method == 'test':
        # distance = model.wmdistance(sentence_obama, sentence_president)
        total_path = os.path.join(root_path, "datasets/cd_test_vectors.txt")
        word_vectors = KeyedVectors.load_word2vec_format(total_path, binary=False)
        model = word2vec.Word2Vec.load(root_path + "/datasets/cd.model")
        log.info("{}, {}".format(len(model.wv['方便接待']), model.wv['方便接待']))
        log.info("{}, {}".format(model.wv.similarity('方便接待', '我们'), model.wv.similarity('方便接待', '旅游')))
        log.info(" ! Build Success ! ")
