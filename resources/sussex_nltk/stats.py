'''
.. codeauthor::
    Matti Lyra
'''

import os
import random
import matplotlib.pyplot as plt
from numpy import average
try:
    import pickle as pickle
except:
    import pickle

import nltk

import sussex_nltk


fh = open(os.path.join(sussex_nltk._sussex_root, 'data', 'sentiword',
                       'sentiword.pickle'), 'rb')
_sent_words = pickle.load(fh)
fh.close()
del fh


def expected_token_freq(tokens, word, _n_norm=5000):
    """Expected frequency of each item in `words` for every `_n_norm` tokens.

    :param int _n_norm: token window size
    :return: float
    """
    stats = []
    for head, tail in zip(list(range(0, len(tokens), _n_norm)),
                          list(range(_n_norm, len(tokens) + _n_norm, _n_norm))):
        chunk = tokens[head:tail]
        stats.append(chunk.count(word))
    
    return average(stats)


def expected_sentiment_tokens(tokens, _n_norm=500):
    """Expected number of sentiment bearing tokens for every `_n_norm` tokens.

    The sentiment bearing word frequency is computed based on the score given
    to words in SentiWord.

    `Link SentiWord <http://sentiwordnet.isti.cnr.it/>`_

    :param list tokens: list of tokens to compute ratio of sentiment bearing
        tokens over.
    :param int _n_norm: token window size
    :return: ratio of sentiment bearing tokens in each `_n_word` chunk
    """
    if len(tokens) < _n_norm:
        raise ValueError('Not enough data to calculate statistic, tokens must'
                         'be longer than %i items.' % _n_norm)
    
    keys = list(_sent_words.keys())
    keys = [k.replace('_', ' ') for k in keys]
    _stats = []
    for head,tail in zip(list(range(0, len(tokens), _n_norm)),
                         list(range(_n_norm, len(tokens) + _n_norm, _n_norm))):
        chunk = tokens[head:tail]
        vocab = set(chunk)
        intersection = vocab.intersection(keys)
        fd = nltk.probability.FreqDist(chunk)
        _stats.append(sum([fd[w] for w in intersection]))
    
    return sum(_stats) / (len(_stats) + 0.0)


def prob_short_sents(sents):
    """Calculates the probability of a sentence of 2 or less tokens.
    """
    return len([sent for sent in sents if len(sent) < 3]) / float(len(sents))


def normalised_lexical_diversity(tokens, _n_norm=500):
    """Calculates the average lexical diversity per `_n_norm` tokens.

    Lexical diversity is computed as the ratio of types to the size of the
    window, as determined by `_n_norm`.

    :param list tokens:
    :param int _n_norm: token window size
    :return: float
    """
    if len(tokens) < _n_norm:
        raise ValueError('Not enough data to calculate statistic,'
                         'tokens must be longer than %i items.' % _n_norm)
    
    _stats = []
    for head, tail in zip(list(range(0, len(tokens), _n_norm)),
                          list(range(_n_norm, len(tokens) + _n_norm, _n_norm))):
        _stats.append(_n_norm / (len(set(tokens[head:tail])) + 0.0)) 
    
    return sum(_stats) / (len(_stats) + 0.0)


def percentage(count, total):
    return 100 * count / (total + 0.0)


def sample_from_corpus(corpus,sample_size):
    n = corpus.enumerate_sents()
    sample_indices = set(random.sample(range(n),sample_size))
    return [sent for i,sent in enumerate(corpus.sents()) if i in sample_indices]


def zipf_dist(freqdist, num_of_ranks=50, show_values=True):
    '''Plot the frequency distribution of a text.

    Given a frequency distribution object, rank all types
    in order of frequency of occurrence (where rank 1 is most
    frequent word), and plot the ranks against the frequency
    of occurrence. If num_of_ranks=20, then 20 types will
    be plotted.
    If show_values = True, then display the bar values above them.
    '''
    x = list(range(1,num_of_ranks+1))                #x values are the ranks of types
    y = list(freqdist.values())[:num_of_ranks]       #y values are the frequencies of the ranked types
    plt.bar(x,y,color="#1AADA4")            #plot a bar graph of x and y
    plt.xlabel("Rank of types ordered by frequency of occurrence")
    plt.ylabel("Frequency of occurrence")   #set the label of the y axis
    plt.grid(True)                          #display grid on graph
    plt.xticks(list(range(1,num_of_ranks+1,2)),list(range(1,num_of_ranks+1,2)))  #set what values appears on the x axis
    plt.xlim([0,num_of_ranks+2])            #limit the display on the x axis
    if show_values:                            #if show_values is True, then show the y values on the bars
        for xi,yi in zip(x,y):
            plt.text(xi+0.25,yi+50,yi,verticalalignment="bottom",rotation=55,fontsize="small")
    plt.show()                              #display the graph
    print("Plot complete.")


def evaluate_wordlist_classifier(cls, pos_test_data, neg_test_data):
    acc = 0
    for review in pos_test_data:
        acc += 1 if cls.classify(review.words()) == "P" else 0
    
    for review in neg_test_data:
        acc += 1 if cls.classify(review.words()) == "N" else 0
    
    return acc / (len(pos_test_data) + len(neg_test_data) + 0.0)
