from __future__ import division
import argparse
import logging
import math
import os
import random
import re
import time
from collections import Counter
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
from nltk.stem.porter import PorterStemmer
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch.optim.optimizer import Optimizer

try:
    import ipdb
except ImportError:
    pass
stemmer = PorterStemmer()

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

lower = True
seq_length = 100
report_every = 100000
shuffle = 1

logger = logging.getLogger(__name__)


def add_data_options(parser):
    ## Data options
    parser.add_argument('-save_path', default='./results',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")

    # tmp solution for load issue
    parser.add_argument('-online_process_data', action="store_true")
    parser.add_argument('-process_shuffle', action="store_true")
    parser.add_argument('-train_src', default='')
    parser.add_argument('-src_vocab', default='')
    parser.add_argument('-train_tgt', default='')
    parser.add_argument('-tgt_vocab', default='')

    # Test options
    parser.add_argument('-dev_input_src',
                        help='Path to the dev input file.')
    parser.add_argument('-dev_ref',
                        help='Path to the dev reference file.')
    parser.add_argument('-beam_size', type=int, default=12,
                        help='Beam size')
    parser.add_argument('-max_sent_length', type=int, default=100,
                        help='Maximum sentence length.')


def add_model_options(parser):
    ## Model options
    parser.add_argument('-layers', type=int, default=1,
                        help='Number of layers in the LSTM encoder/decoder')
    parser.add_argument('-enc_rnn_size', type=int, default=512,
                        help='Size of LSTM hidden states')
    parser.add_argument('-dec_rnn_size', type=int, default=512,
                        help='Size of LSTM hidden states')
    parser.add_argument('-word_vec_size', type=int, default=300,
                        help='Word embedding sizes')
    parser.add_argument('-att_vec_size', type=int, default=512,
                        help='Concat attention vector sizes')
    parser.add_argument('-maxout_pool_size', type=int, default=2,
                        help='Pooling size for MaxOut layer.')
    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")
    # parser.add_argument('-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")
    parser.add_argument('-brnn', action='store_true',
                        help='Use a bidirectional encoder')
    parser.add_argument('-brnn_merge', default='concat',
                        help="""Merge action for the bidirectional hidden states:
                        [concat|sum]""")


def add_train_options(parser):
    ## Optimization options
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but uses
                        more memory.""")
    parser.add_argument('-epochs', type=int, default=13,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init)""")
    parser.add_argument('-optim', default='sgd',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-max_weight_value', type=float, default=15,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.3,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-curriculum', type=int, default=1,
                        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""")
    parser.add_argument('-extra_shuffle', action="store_true",
                        help="""By default only shuffle mini-batch order; when true,
                        shuffle and re-assign mini-batches""")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=1.0,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-start_eval_batch', type=int, default=15000,
                        help="""evaluate on dev per x batches.""")
    parser.add_argument('-eval_per_batch', type=int, default=1000,
                        help="""evaluate on dev per x batches.""")
    parser.add_argument('-halve_lr_bad_count', type=int, default=6,
                        help="""evaluate on dev per x batches.""")

    # pretrained word vectors
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")

    # GPU
    parser.add_argument('-gpus', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")

    parser.add_argument('-log_interval', type=int, default=100,
                        help="logger.info stats at this interval.")

    parser.add_argument('-seed', type=int, default=-1,
                        help="""Random seed used for the experiments
                        reproducibility.""")
    parser.add_argument('-cuda_seed', type=int, default=-1,
                        help="""Random CUDA seed used for the experiments
                        reproducibility.""")

    parser.add_argument('-log_home', default='',
                        help="""log home""")


parser = argparse.ArgumentParser(description='train.py')
add_data_options(parser)
add_model_options(parser)
add_train_options(parser)

opt = parser.parse_args()

logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
if opt.log_home:
    log_file_name = os.path.join(opt.log_home, log_file_name)
file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)

logger.info('My PID is {0}'.format(os.getpid()))
logger.info(opt)

if torch.cuda.is_available() and not opt.gpus:
    logger.info("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.seed > 0:
    torch.manual_seed(opt.seed)
opt.gpus = False
if opt.gpus:
    if opt.cuda_seed > 0:
        torch.cuda.manual_seed(opt.cuda_seed)
    cuda.set_device(opt.gpus)


class Dict(object):
    def __init__(self, data=None, lower=False):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)

    def size(self):
        return len(self.idxToLabel)

    # Load entries from a file.
    def loadFile(self, filename):
        for line in open(filename, encoding='utf-8'):
            fields = line.split(' ')
            label = fields[0]
            idx = int(fields[1])
            self.add(label, idx)

    # Write entries to a file.
    def writeFile(self, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))

        file.close()

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, size):
        if size >= self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor(
            [self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)

        newDict = Dict()
        newDict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for i in idx[:size]:
            newDict.add(self.idxToLabel[i])

        return newDict

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec)

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop):
        labels = []

        for i in idx:
            labels += [self.getLabel(i)]
            if i == stop:
                break

        return labels


def makeVocabulary(filenames, size):
    vocab = Dict([PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD], lower=lower)
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().split(' '):
                    vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    logger.info('Created dictionary of size %d (pruned from %d)' %
                (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        logger.info('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = Dict()
        vocab.loadFile(vocabFile)
        logger.info('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        logger.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    logger.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            logger.info('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        srcWords = sline.split(' ')
        tgtWords = tline.split(' ')

        if len(srcWords) <= seq_length and len(tgtWords) <= seq_length:
            src += [srcDicts.convertToIdx(srcWords,
                                          UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          UNK_WORD,
                                          BOS_WORD,
                                          EOS_WORD)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if shuffle == 1:
        logger.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    logger.info('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(src), ignored, seq_length))
    return src, tgt


def prepare_data_online(train_src, src_vocab, train_tgt, tgt_vocab):
    dicts = {}
    dicts['src'] = initVocabulary('source', [train_src], src_vocab, 0)
    dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, 0)

    logger.info('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeData(train_src,
                                          train_tgt,
                                          dicts['src'],
                                          dicts['tgt'])

    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
               }
    return dataset


# logger.info('My seed is {0}'.format(torch.initial_seed()))
# logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))
#

class Rouge(object):
    def __init__(self, stem=True, use_ngram_buf=False):
        self.N = 2
        self.stem = stem
        self.use_ngram_buf = use_ngram_buf
        self.ngram_buf = {}

    @staticmethod
    def _format_sentence(sentence):
        s = sentence.lower()
        s = re.sub(r"[^0-9a-z]", " ", s)
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        return s

    def _create_n_gram(self, raw_sentence, n, stem):
        if self.use_ngram_buf:
            if raw_sentence in self.ngram_buf:
                return self.ngram_buf[raw_sentence]
        res = {}
        sentence = Rouge._format_sentence(raw_sentence)
        tokens = sentence.split(' ')
        if stem:
            # try:  # TODO older NLTK has a bug in Porter Stemmer
            tokens = [stemmer.stem(t) for t in tokens]
            # except:
            #     pass
        sent_len = len(tokens)
        for _n in range(n):
            buf = Counter()
            for idx, token in enumerate(tokens):
                if idx + _n >= sent_len:
                    break
                ngram = ' '.join(tokens[idx: idx + _n + 1])
                buf[ngram] += 1
            res[_n] = buf
        if self.use_ngram_buf:
            self.ngram_buf[raw_sentence] = res
        return res

    def get_ngram(self, sents, N, stem=False):
        if isinstance(sents, list):
            res = {}
            for _n in range(N):
                res[_n] = Counter()
            for sent in sents:
                ngrams = self._create_n_gram(sent, N, stem)
                for this_n, counter in ngrams.items():
                    # res[this_n] = res[this_n] + counter
                    self_counter = res[this_n]
                    for elem, count in counter.items():
                        if elem not in self_counter:
                            self_counter[elem] = count
                        else:
                            self_counter[elem] += count
            return res
        elif isinstance(sents, str):
            return self._create_n_gram(sents, N, stem)
        else:
            raise ValueError

    def get_mean_sd_internal(self, x):
        mean = np.mean(x)
        sd = st.sem(x)
        res = st.t.interval(0.95, len(x) - 1, loc=mean, scale=sd)
        return (mean, sd, res)

    def compute_rouge(self, references, systems):
        assert (len(references) == len(systems))

        peer_count = len(references)

        result_buf = {}
        for n in range(self.N):
            result_buf[n] = {'p': [], 'r': [], 'f': []}

        for ref_sent, sys_sent in zip(references, systems):
            ref_ngrams = self.get_ngram(ref_sent, self.N, self.stem)
            sys_ngrams = self.get_ngram(sys_sent, self.N, self.stem)
            for n in range(self.N):
                ref_ngram = ref_ngrams[n]
                sys_ngram = sys_ngrams[n]
                ref_count = sum(ref_ngram.values())
                sys_count = sum(sys_ngram.values())
                match_count = 0
                for k, v in sys_ngram.items():
                    if k in ref_ngram:
                        match_count += min(v, ref_ngram[k])
                p = match_count / sys_count if sys_count != 0 else 0
                r = match_count / ref_count if ref_count != 0 else 0
                f = 0 if (p == 0 or r == 0) else 2 * p * r / (p + r)
                result_buf[n]['p'].append(p)
                result_buf[n]['r'].append(r)
                result_buf[n]['f'].append(f)

        res = {}
        for n in range(self.N):
            n_key = 'rouge-{0}'.format(n + 1)
            res[n_key] = {}
            if len(result_buf[n]['p']) >= 50:
                res[n_key]['p'] = self.get_mean_sd_internal(result_buf[n]['p'])
                res[n_key]['r'] = self.get_mean_sd_internal(result_buf[n]['r'])
                res[n_key]['f'] = self.get_mean_sd_internal(result_buf[n]['f'])
            else:
                # not enough samples to calculate confidence interval
                res[n_key]['p'] = (np.mean(np.array(result_buf[n]['p'])), 0, (0, 0))
                res[n_key]['r'] = (np.mean(np.array(result_buf[n]['r'])), 0, (0, 0))
                res[n_key]['f'] = (np.mean(np.array(result_buf[n]['f'])), 0, (0, 0))

        return res


def calculate_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity function. The values are as follows:

    ============ ==========================================
    nonlinearity gain
    ============ ==========================================
    linear       :math:`1`
    conv{1,2,3}d :math:`1`
    sigmoid      :math:`1`
    tanh         :math:`5 / 3`
    relu         :math:`\sqrt{2}`
    leaky_relu   :math:`\sqrt{2 / (1 + negative\_slope^2)}`
    ============ ==========================================

    Args:
        nonlinearity: the nonlinear function (`nn.functional` name)
        param: optional parameter for the nonlinear function

    Examples:
        >>> gain = nn.init.gain('leaky_relu')
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def uniform(tensor, a=0, b=1):
    """Fills the input Tensor or Variable with values drawn from the uniform distribution :math:`U(a, b)`.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.uniform(w)
    """
    if isinstance(tensor, Variable):
        uniform(tensor.data, a=a, b=b)
        return tensor

    return tensor.uniform_(a, b)


def normal(tensor, mean=0, std=1):
    """Fills the input Tensor or Variable with values drawn from the normal distribution :math:`N(mean, std)`.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.normal(w)
    """
    if isinstance(tensor, Variable):
        normal(tensor.data, mean=mean, std=std)
        return tensor

    return tensor.normal_(mean, std)


def constant(tensor, val):
    """Fills the input Tensor or Variable with the value `val`.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.constant(w)
    """
    if isinstance(tensor, Variable):
        constant(tensor.data, val)
        return tensor

    return tensor.fill_(val)


def eye(tensor):
    """Fills the 2-dimensional input Tensor or Variable with the identity matrix. Preserves the identity of the inputs
    in Linear layers, where as many inputs are preserved as possible.

    Args:
        tensor: a 2-dimensional torch.Tensor or autograd.Variable

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.eye(w)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    if isinstance(tensor, Variable):
        eye(tensor.data)
        return tensor

    return tensor.copy_(torch.eye(tensor.size(0), tensor.size(1)))


def dirac(tensor):
    """Fills the {3, 4, 5}-dimensional input Tensor or Variable with the Dirac delta function. Preserves the identity of
    the inputs in Convolutional layers, where as many input channels are preserved as possible.

    Args:
        tensor: a {3, 4, 5}-dimensional torch.Tensor or autograd.Variable

    Examples:
        >>> w = torch.Tensor(3, 16, 5, 5)
        >>> nn.init.dirac(w)
    """
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    if isinstance(tensor, Variable):
        dirac(tensor.data)
        return tensor

    sizes = tensor.size()
    min_dim = min(sizes[0], sizes[1])
    tensor.zero_()

    for d in range(min_dim):
        if dimensions == 3:  # Temporal convolution
            tensor[d, d, tensor.size(2) // 2] = 1
        elif dimensions == 4:  # Spatial convolution
            tensor[d, d, tensor.size(2) // 2, tensor.size(3) // 2] = 1
        else:  # Volumetric convolution
            tensor[d, d, tensor.size(2) // 2, tensor.size(3) // 2, tensor.size(4) // 2] = 1
    return tensor


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the
    difficulty of training deep feedforward neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from :math:`U(-a, a)` where
    :math:`a = gain \\times \sqrt{2 / (fan\_in + fan\_out)} \\times \sqrt{3}`. Also known as Glorot initialisation.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        gain: an optional scaling factor

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('relu'))
    """
    if isinstance(tensor, Variable):
        xavier_uniform(tensor.data, gain=gain)
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the
    difficulty of training deep feedforward neural networks" - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from :math:`N(0, std)` where
    :math:`std = gain \\times \sqrt{2 / (fan\_in + fan\_out)}`. Also known as Glorot initialisation.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        gain: an optional scaling factor

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.xavier_normal(w)
    """
    if isinstance(tensor, Variable):
        xavier_normal(tensor.data, gain=gain)
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return tensor.normal_(0, std)


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform(tensor, a=0, mode='fan_in'):
    """Fills the input Tensor or Variable with values according to the method described in "Delving deep into
    rectifiers: Surpassing human-level performance on ImageNet classification" - He, K. et al. (2015), using a uniform
    distribution. The resulting tensor will have values sampled from :math:`U(-bound, bound)` where
    :math:`bound = \sqrt{2 / ((1 + a^2) \\times fan\_in)} \\times \sqrt{3}`. Also known as He initialisation.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        a: the negative slope of the rectifier used after this layer (0 for ReLU by default)
        mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in` preserves the magnitude of the variance of the
              weights in the forward pass. Choosing `fan_out` preserves the magnitudes in the backwards pass.

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.kaiming_uniform(w, mode='fan_in')
    """
    if isinstance(tensor, Variable):
        kaiming_uniform(tensor.data, a=a, mode=mode)
        return tensor

    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain('leaky_relu', a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-bound, bound)


def kaiming_normal(tensor, a=0, mode='fan_in'):
    """Fills the input Tensor or Variable with values according to the method described in "Delving deep into
    rectifiers: Surpassing human-level performance on ImageNet classification" - He, K. et al. (2015), using a normal
    distribution. The resulting tensor will have values sampled from :math:`N(0, std)` where
    :math:`std = \sqrt{2 / ((1 + a^2) \\times fan\_in)}`. Also known as He initialisation.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        a: the negative slope of the rectifier used after this layer (0 for ReLU by default)
        mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in` preserves the magnitude of the variance of the
              weights in the forward pass. Choosing `fan_out` preserves the magnitudes in the backwards pass.

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.kaiming_normal(w, mode='fan_out')
    """
    if isinstance(tensor, Variable):
        kaiming_normal(tensor.data, a=a, mode=mode)
        return tensor

    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain('leaky_relu', a)
    std = gain / math.sqrt(fan)
    return tensor.normal_(0, std)


def orthogonal(tensor, gain=1):
    """Fills the input Tensor or Variable with a (semi) orthogonal matrix, as described in "Exact solutions to the
    nonlinear dynamics of learning in deep linear neural networks" - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable, where n >= 2
        gain: optional scaling factor

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.orthogonal(w)
    """
    if isinstance(tensor, Variable):
        orthogonal(tensor.data, gain=gain)
        return tensor

    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)
    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    # Pad zeros to Q (if rows smaller than cols)
    if rows < cols:
        padding = torch.zeros(rows, cols - rows)
        if q.is_cuda:
            q = torch.cat([q, padding.cuda()], 1)
        else:
            q = torch.cat([q, padding], 1)

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor


def sparse(tensor, sparsity, std=0.01):
    """Fills the 2D input Tensor or Variable as a sparse matrix, where the non-zero elements will be drawn from
    the normal distribution :math:`N(0, 0.01)`, as described in "Deep learning via
    Hessian-free optimization" - Martens, J. (2010).

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate the non-zero values

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.sparse(w, sparsity=0.1)
    """
    if isinstance(tensor, Variable):
        sparse(tensor.data, sparsity, std=std)
        return tensor

    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    tensor.normal_(0, std)
    rows, cols = tensor.size(0), tensor.size(1)
    num_zeros = int(math.ceil(cols * sparsity))

    for col_idx in range(tensor.size(1)):
        row_indices = list(range(rows))
        random.shuffle(row_indices)
        zero_indices = row_indices[:num_zeros]
        for row_idx in zero_indices:
            tensor[row_idx, col_idx] = 0

    return tensor


def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def loss_function(g_outputs, g_targets, generator, crit, eval=False):
    g_out_t = g_outputs.view(-1, g_outputs.size(2))
    g_prob_t = generator(g_out_t)

    g_loss = crit(g_prob_t, g_targets.view(-1))
    total_loss = g_loss
    report_loss = total_loss.data[0]
    return total_loss, report_loss, 0


def addPair(f1, f2):
    for x, y1 in zip(f1, f2):
        yield (x, y1)
    yield (None, None)


def load_dev_data(translator, src_file, tgt_file):
    dataset, raw = [], []
    srcF = open(src_file, encoding='utf-8')
    tgtF = open(tgt_file, encoding='utf-8')
    src_batch, tgt_batch = [], []
    for line, tgt in addPair(srcF, tgtF):
        if (line is not None) and (tgt is not None):
            src_tokens = line.strip().split(' ')
            src_batch += [src_tokens]
            tgt_tokens = tgt.strip().split(' ')
            tgt_batch += [tgt_tokens]

            if len(src_batch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(src_batch) == 0:
                break
        data = translator.buildData(src_batch, tgt_batch)
        dataset.append(data)
        raw.append((src_batch, tgt_batch))
        src_batch, tgt_batch = [], []
    srcF.close()
    tgtF.close()
    return (dataset, raw)


evalModelCount = 0
totalBatchCount = 0
rouge_calculator = Rouge()


def evalModel(model, translator, evalData):
    global evalModelCount
    global rouge_calculator
    evalModelCount += 1
    ofn = 'dev.out.{0}'.format(evalModelCount)
    if opt.save_path:
        ofn = os.path.join(opt.save_path, ofn)
    predict, gold = [], []
    processed_data, raw_data = evalData
    for batch, raw_batch in zip(processed_data, raw_data):
        # (wrap(srcBatch), lengths), (wrap(tgtBatch), ), indices
        src, tgt, indices = batch[0]
        src_batch, tgt_batch = raw_batch

        #  (2) translate
        pred, predScore, attn, _ = translator.translateBatch(src, tgt)
        pred, predScore, attn = list(zip(
            *sorted(zip(pred, predScore, attn, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            n = 0
            predBatch.append(
                translator.buildTargetTokens(pred[b][n], src_batch[b], attn[b][n])
            )
        gold += [' '.join(r) for r in tgt_batch]
        predict += [' '.join(sents) for sents in predBatch]
    scores = rouge_calculator.compute_rouge(gold, predict)
    with open(ofn, 'w', encoding='utf-8') as of:
        for p in predict:
            of.write(p + '\n')
    return scores['rouge-2']['f'][0]


def trainModel(model, translator, trainData, validData, dataset, optim):
    logger.info(model)
    model.train()
    logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))

    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    start_time = time.time()

    def saveModel(metric=None):
        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(
            opt.gpus) > 1 else model.generator.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }
        save_model_path = 'model'
        if opt.save_path:
            if not os.path.exists(opt.save_path):
                os.makedirs(opt.save_path)
            save_model_path = opt.save_path + os.path.sep + save_model_path
        if metric is not None:
            torch.save(checkpoint, '{0}_devRouge_{1}_e{2}.pt'.format(save_model_path, round(metric, 4), epoch))
        else:
            torch.save(checkpoint, '{0}_e{1}.pt'.format(save_model_path, epoch))

    def trainEpoch(epoch):

        if opt.extra_shuffle and epoch > opt.curriculum:
            logger.info('Shuffling...')
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
        start = time.time()
        for i in range(len(trainData)):
            global totalBatchCount
            totalBatchCount += 1
            # (wrap(srcBatch), lengths), (wrap(tgtBatch)), indices
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx][:-1]  # exclude original indices

            model.zero_grad()
            g_outputs = model(batch)
            targets = batch[1][0][1:]  # exclude <s> from targets
            loss, res_loss, num_correct = loss_function(g_outputs, targets, model.generator, criterion)

            # update the parameters
            loss.backward()
            optim.step()

            num_words = targets.data.ne(PAD).sum()
            report_loss += res_loss
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += batch[0][-1].data.sum()
            total_loss += res_loss
            total_num_correct += num_correct
            total_words += num_words
            if i % opt.log_interval == -1 % opt.log_interval:
                logger.info(
                    "Epoch %2d, %6d/%5d/%5d; acc: %6.2f; loss: %6.2f; words: %5d; ppl: %6.2f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                    (epoch, totalBatchCount, i + 1, len(trainData),
                     report_num_correct / report_tgt_words * 100,
                     report_loss,
                     report_tgt_words,
                     math.exp(report_loss / report_tgt_words),
                     report_src_words / (time.time() - start),
                     report_tgt_words / (time.time() - start),
                     time.time() - start))

                report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                start = time.time()

            if validData is not None and totalBatchCount % opt.eval_per_batch == -1 % opt.eval_per_batch \
                    and totalBatchCount >= opt.start_eval_batch:
                model.eval()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
                valid_bleu = evalModel(model, translator, validData)
                model.train()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
                model.decoder.attn.mask = None
                logger.info('Validation Score: %g' % (valid_bleu * 100))
                if valid_bleu >= optim.best_metric:
                    saveModel(valid_bleu)
                optim.updateLearningRate(valid_bleu, epoch)

        return total_loss / total_words, total_num_correct / total_words

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        logger.info('')
        global eeee
        eeee = epoch
        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        logger.info('Train perplexity: %g' % train_ppl)
        logger.info('Train accuracy: %g' % (train_acc * 100))
        logger.info('Saving checkpoint for epoch {0}...'.format(epoch))
        saveModel()


class Encoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2
        assert opt.enc_rnn_size % self.num_directions == 0
        self.hidden_size = opt.enc_rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=PAD)
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=opt.layers,
                          dropout=opt.dropout,
                          bidirectional=True)
        self.selective_gate = nn.Linear(self.hidden_size * 2 * 2, self.hidden_size * 2)
        self.sigmoid = nn.Sigmoid()

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        """
        input: (wrap(srcBatch), wrap(srcBioBatch), lengths)
        """
        lengths = input[-1].data.view(-1).tolist()  # lengths data is wrapped inside a Variable
        wordEmb = self.word_lut(input[0])
        emb = pack(wordEmb, lengths)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        forward_last = hidden_t[0]
        backward_last = hidden_t[1]
        time_step = outputs.size(0)
        batch_size = outputs.size(1)
        sentence_vector = torch.cat((forward_last, backward_last), dim=1)
        exp_buf = torch.cat((outputs, sentence_vector.unsqueeze(0).expand_as(outputs)), dim=2)
        selective_value = self.sigmoid(self.selective_gate(exp_buf.view(-1, exp_buf.size(2))))
        selective_value = selective_value.view(time_step, batch_size, -1)
        outputs = outputs * selective_value
        return hidden_t, outputs


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class MaxOut(nn.Module):
    def __init__(self, pool_size):
        super(MaxOut, self).__init__()
        self.pool_size = pool_size

    def forward(self, input):
        """
        input:
        reduce_size:
        """
        input_size = list(input.size())
        assert input_size[-1] % self.pool_size == 0
        output_size = [d for d in input_size]
        output_size[-1] = output_size[-1] // self.pool_size
        output_size.append(self.pool_size)
        last_dim = len(output_size) - 1
        input = input.view(*output_size)
        input, idx = input.max(last_dim, keepdim=True)
        output = input.squeeze(last_dim)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '({0})'.format(self.pool_size)


class ConcatAttention(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim):
        super(ConcatAttention, self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, precompute=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        if precompute is None:
            precompute00 = self.linear_pre(context.contiguous().view(-1, context.size(2)))
            precompute = precompute00.view(context.size(0), context.size(1), -1)  # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp10 = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim
        tmp20 = self.tanh(tmp10)  # batch x sourceL x att_dim
        energy = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL
        if self.mask is not None:
            # energy.data.masked_fill_(self.mask, -float('inf'))
            # energy.masked_fill_(self.mask, -float('inf'))   # TODO: might be wrong
            energy = energy * (1 - self.mask) + self.mask * (-1000000)
        score = self.sm(energy)
        score_m = score.view(score.size(0), 1, score.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(score_m, context).squeeze(1)  # batch x dim

        return weightedContext, score, precompute

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
               + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
               + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'


class Decoder(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.enc_rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=PAD)
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.attn = ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.readout = nn.Linear((opt.enc_rnn_size + opt.dec_rnn_size + opt.word_vec_size), opt.dec_rnn_size)
        self.maxout = MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size

        self.hidden_size = opt.dec_rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, src_pad_mask, init_att):
        emb = self.word_lut(input)

        g_outputs = []
        cur_context = init_att
        self.attn.applyMask(src_pad_mask)
        precompute = None
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            input_emb = emb_t
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_context], 1)
            output, hidden = self.rnn(input_emb, hidden)
            cur_context, attn, precompute = self.attn(output, context.transpose(0, 1), precompute)

            readout = self.readout(torch.cat((emb_t, output, cur_context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            g_outputs += [output]
        g_outputs = torch.stack(g_outputs)
        return g_outputs, hidden, attn, cur_context


class DecInit(nn.Module):
    def __init__(self, opt):
        super(DecInit, self).__init__()
        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.enc_rnn_size = opt.enc_rnn_size
        self.dec_rnn_size = opt.dec_rnn_size
        self.initer = nn.Linear(self.enc_rnn_size // self.num_directions, self.dec_rnn_size)
        self.tanh = nn.Tanh()

    def forward(self, last_enc_h):
        # batchSize = last_enc_h.size(0)
        # dim = last_enc_h.size(1)
        return self.tanh(self.initer(last_enc_h))


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, decIniter):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decIniter = decIniter

    def make_init_att(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def forward(self, input):
        """
        input: (wrap(srcBatch), wrap(srcBioBatch), lengths), (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch))
        """
        # ipdb.set_trace()
        src = input[0]
        tgt = input[1][0][:-1]  # exclude last target from inputs
        src_pad_mask = Variable(src[0].data.eq(PAD).transpose(0, 1).float(), requires_grad=False,
                                volatile=False)
        enc_hidden, context = self.encoder(src)

        init_att = self.make_init_att(context)
        enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)  # [1] is the last backward hiden

        g_out, dec_hidden, _attn, _attention_vector = self.decoder(tgt, enc_hidden, context, src_pad_mask, init_att)

        return g_out


class Dataset(object):
    def __init__(self, srcData, tgtData, batchSize, cuda, volatile=False):
        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)
        self.volatile = volatile

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index * self.batchSize:(index + 1) * self.batchSize])
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        if tgtBatch is None:
            batch = zip(indices, srcBatch)
        else:
            batch = zip(indices, srcBatch, tgtBatch)
        # batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch = zip(*batch)
        else:
            indices, srcBatch, tgtBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)

        return (wrap(srcBatch), lengths), (wrap(tgtBatch),), indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])


class Beam(object):
    def __init__(self, size, cuda=False):

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []
        self.all_length = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(PAD)]
        self.nextYs[0][0] = BOS

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def getCurrentState(self):
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def getCurrentOrigin(self):
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.
    def advance(self, wordLk, attnOut):
        numWords = wordLk.size(1)

        # self.length += 1  # TODO: some is finished so do not acc length for them
        if len(self.prevKs) > 0:
            finish_index = self.nextYs[-1].eq(EOS)
            if any(finish_index):
                wordLk.masked_fill_(finish_index.unsqueeze(1).expand_as(wordLk), -float('inf'))
                for i in range(self.size):
                    if self.nextYs[-1][i] == EOS:
                        wordLk[i][EOS] = 0
            # set up the current step length
            cur_length = self.all_length[-1]
            for i in range(self.size):
                cur_length[i] += 0 if self.nextYs[-1][i] == EOS else 1

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            prev_score = self.all_scores[-1]
            now_acc_score = wordLk + prev_score.unsqueeze(1).expand_as(wordLk)
            beamLk = now_acc_score / cur_length.unsqueeze(1).expand_as(now_acc_score)
        else:
            self.all_length.append(self.tt.FloatTensor(self.size).fill_(1))
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        predict = bestScoresId - prevK * numWords

        if len(self.prevKs) > 0:
            self.all_length.append(cur_length.index_select(0, prevK))
            self.all_scores.append(now_acc_score.view(-1).index_select(0, bestScoresId))
        else:
            self.all_scores.append(self.scores)

        self.prevKs.append(prevK)
        self.nextYs.append(predict)
        self.attn.append(attnOut.index_select(0, prevK))

        # End condition is when every one is EOS.
        if all(self.nextYs[-1].eq(EOS)):
            self.done = True

        return self.done

    def sortBest(self):
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def getBest(self):
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def getHyp(self, k):
        hyp, attn = [], []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]

        return hyp[::-1], torch.stack(attn[::-1])


class Translator(object):
    def __init__(self, opt, model=None, dataset=None):
        self.opt = opt

        if model is None:

            checkpoint = torch.load(opt.model)

            model_opt = checkpoint['opt']
            self.src_dict = checkpoint['dicts']['src']
            self.tgt_dict = checkpoint['dicts']['tgt']

            self.enc_rnn_size = model_opt.enc_rnn_size
            self.dec_rnn_size = model_opt.dec_rnn_size
            encoder = Encoder(model_opt, self.src_dict)
            decoder = Decoder(model_opt, self.tgt_dict)
            decIniter = DecInit(model_opt)
            model = NMTModel(encoder, decoder, decIniter)

            generator = nn.Sequential(
                nn.Linear(model_opt.dec_rnn_size // model_opt.maxout_pool_size, self.tgt_dict.size()),
                nn.LogSoftmax())

            model.load_state_dict(checkpoint['model'])
            generator.load_state_dict(checkpoint['generator'])

            if opt.cuda:
                model.cuda()
                generator.cuda()
            else:
                model.cpu()
                generator.cpu()

            model.generator = generator
        else:
            self.src_dict = dataset['dicts']['src']
            self.tgt_dict = dataset['dicts']['tgt']

            self.enc_rnn_size = opt.enc_rnn_size
            self.dec_rnn_size = opt.dec_rnn_size
            self.opt.cuda = True if len(opt.gpus) >= 1 else False
            self.opt.n_best = 1
            self.opt.replace_unk = False

        self.tt = torch.cuda if opt.cuda else torch
        self.model = model
        self.model.eval()

        self.copyCount = 0

    def buildData(self, srcBatch, goldBatch):
        srcData = [self.src_dict.convertToIdx(b, UNK_WORD) for b in srcBatch]
        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b, UNK_WORD, BOS_WORD, EOS_WORD) for b in goldBatch]

        return Dataset(srcData, tgtData, self.opt.batch_size, self.opt.cuda, volatile=True)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, srcBatch, tgtBatch):
        batchSize = srcBatch[0].size(1)
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(srcBatch)
        srcBatch = srcBatch[0]  # drop the lengths needed for encoder

        decStates = self.model.decIniter(encStates[1])  # batch, dec_hidden

        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        context = Variable(context.data.repeat(1, beamSize, 1))
        decStates = Variable(decStates.unsqueeze(0).data.repeat(1, beamSize, 1))
        att_vec = self.model.make_init_att(context)
        padMask = Variable(
            srcBatch.data.eq(PAD).transpose(0, 1).unsqueeze(0).repeat(beamSize, 1, 1).float(),
            volatile=True)

        beam = [Beam(beamSize, self.opt.cuda) for k in range(batchSize)]
        batchIdx = list(range(batchSize))
        remainingSents = batchSize

        for i in range(self.opt.max_sent_length):
            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).transpose(0, 1).contiguous().view(1, -1)
            g_outputs, decStates, attn, att_vec = self.model.decoder(
                Variable(input, volatile=True), decStates, context, padMask.view(-1, padMask.size(2)), att_vec)

            # g_outputs: 1 x (beam*batch) x numWords
            g_outputs = g_outputs.squeeze(0)
            g_out_prob = self.model.generator.forward(g_outputs)

            # batch x beam x numWords
            wordLk = g_out_prob.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            father_idx = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]
                    father_idx.append(beam[b].prevKs[-1])  # this is very annoying

            if not active:
                break

            # to get the real father index
            real_father_idx = []
            for kk, idx in enumerate(father_idx):
                real_father_idx.append(idx * len(father_idx) + kk)

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t, rnnSize):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx) \
                                .view(*newSize), volatile=True)

            decStates = updateActive(decStates, self.dec_rnn_size)
            context = updateActive(context, self.enc_rnn_size)
            att_vec = updateActive(att_vec, self.enc_rnn_size)
            padMask = padMask.index_select(1, Variable(activeIdx, volatile=True))

            # set correct state for beam search
            previous_index = torch.stack(real_father_idx).transpose(0, 1).contiguous()
            previous_index = Variable(previous_index, volatile=True)
            decStates = decStates.view(-1, decStates.size(2)).index_select(0, previous_index.view(-1)).view(
                *decStates.size())
            att_vec = att_vec.view(-1, att_vec.size(1)).index_select(0, previous_index.view(-1)).view(*att_vec.size())

            remainingSents = len(active)

        # (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            valid_attn = srcBatch.data[:, b].ne(PAD).nonzero().squeeze(1)
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            allHyp += [hyps]
            allAttn += [attn]

        return allHyp, allScores, allAttn, None

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        # (wrap(srcBatch),  lengths), (wrap(tgtBatch), ), indices
        src, tgt, indices = dataset[0]

        #  (2) translate
        pred, predScore, attn, _ = self.translateBatch(src, tgt)
        pred, predScore, attn = list(zip(
            *sorted(zip(pred, predScore, attn, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, None


class MyAdam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(MyAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = exp_avg_sq.sqrt().add_(group['eps'] * math.sqrt(bias_correction2))
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_input = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.linear_hidden = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hidden, mask=None):
        x_W = self.linear_input(input)
        h_U = self.linear_hidden(hidden)
        x_Ws = x_W.split(self.hidden_size, 1)
        h_Us = h_U.split(self.hidden_size, 1)
        r = self.sigmoid(x_Ws[0] + h_Us[0])
        z = self.sigmoid(x_Ws[1] + h_Us[1])
        h1 = self.tanh(x_Ws[2] + r * h_Us[2])
        h = (h1 - hidden) * z + hidden
        if mask:
            h = (h - hidden) * mask.unsqueeze(1).expand_as(hidden) + hidden
        return h

    def __repr__(self):
        return self.__class__.__name__ + '({0}, {1})'.format(self.input_size, self.hidden_size)


class Optim(object):
    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            # self.optimizer = optim.Adam(self.params, lr=self.lr)
            self.optimizer = MyAdam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm, max_weight_value=None, lr_decay=1, start_decay_at=None,
                 decay_bad_count=6):
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.max_weight_value = max_weight_value
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.decay_bad_count = decay_bad_count
        self.best_metric = 0
        self.bad_count = 0

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()
        if self.max_weight_value:
            for p in self.params:
                p.data.clamp_(0 - self.max_weight_value, self.max_weight_value)

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        # if self.start_decay_at is not None and epoch >= self.start_decay_at:
        #     self.start_decay = True
        # if self.last_ppl is not None and ppl > self.last_ppl:
        #     self.start_decay = True
        #
        # if self.start_decay:
        #     self.lr = self.lr * self.lr_decay
        #     print("Decaying learning rate to %g" % self.lr)

        # self.last_ppl = ppl
        if ppl >= self.best_metric:
            self.best_metric = ppl
            self.bad_count = 0
        else:
            self.bad_count += 1
        logger.info('Bad_count: {0}\tCurrent lr: {1}'.format(self.bad_count, self.lr))
        logger.info('Best metric: {0}'.format(self.best_metric))

        if self.bad_count >= self.decay_bad_count and self.lr >= 1e-6:
            self.lr = self.lr * self.lr_decay
            logger.info("Decaying learning rate to %g" % self.lr)
            self.bad_count = 0
        self.optimizer.param_groups[0]['lr'] = self.lr


def main():
    seq_length = opt.max_sent_length
    shuffle = 1 if opt.process_shuffle else 0
    dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab)

    trainData = Dataset(dataset['train']['src'], dataset['train']['tgt'], opt.batch_size, opt.gpus)
    dicts = dataset['dicts']
    logger.info(' * vocabulary size. source = %d; target = %d' %
                (dicts['src'].size(), dicts['tgt'].size()))
    logger.info(' * number of training sentences. %d' %
                len(dataset['train']['src']))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    logger.info('Building model...')

    encoder = Encoder(opt, dicts['src'])
    decoder = Decoder(opt, dicts['tgt'])
    decIniter = DecInit(opt)

    generator = nn.Sequential(
        nn.Linear(opt.dec_rnn_size // opt.maxout_pool_size, dicts['tgt'].size()),  # TODO: fix here
        nn.LogSoftmax())

    model = NMTModel(encoder, decoder, decIniter)
    model.generator = generator
    translator = Translator(opt, model, dataset)

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    for pr_name, p in model.named_parameters():
        logger.info(pr_name)
        # p.data.uniform_(-opt.param_init, opt.param_init)
        if p.dim() == 1:
            # p.data.zero_()
            p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
        else:
            xavier_normal(p, math.sqrt(3))
            # xavier_uniform(p)

    encoder.load_pretrained_vectors(opt)
    decoder.load_pretrained_vectors(opt)

    optim = Optim(
        opt.optim, opt.learning_rate,
        max_grad_norm=opt.max_grad_norm,
        max_weight_value=opt.max_weight_value,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        decay_bad_count=opt.halve_lr_bad_count
    )

    optim.set_parameters(model.parameters())

    validData = None
    if opt.dev_input_src and opt.dev_ref:
        validData = load_dev_data(translator, opt.dev_input_src, opt.dev_ref)
    trainModel(model, translator, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
