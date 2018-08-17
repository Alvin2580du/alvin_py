"""
.. codeauthor::
    Matti Lyra
"""

from . import cmu
import itertools



def twitter_tokenize_batch(sents):
    """Tokenizes a list of sentences using the CMU twitter tokenizer.

    Calling the batch method is faster than sequentially calling
    `twitter_tokenize` is for a large number of sentences.

    :param list sents: list of sentences to tokenize
    :return: list of tokenized sentences
    :rtype: [[(word, tag), ...], [(word, tag), ...]]
    """
    _output_data = cmu.tag(sents)
    _output_data = _output_data.strip()
    _output_tokens = []
    for line in _output_data.split('\n'):
        token, _, _ = line.partition('\t')
        _output_tokens.append(token)
    return [list(g) for k,g in itertools.groupby(_output_tokens,lambda x:x in ['',]) if not k]


def twitter_tokenize_batch_backup(sents):
    """Tokenizes a list of sentences using the CMU twitter tokenizer.

    Calling the batch method is faster than sequentially calling
    `twitter_tokenize` is for a large number of sentences.

    :param list sents: list of sentences to tokenize
    :return: list of tokenized sentences
    :rtype: [[(word, tag), ...], [(word, tag), ...]]
    """
    _output_data = cmu.tag(sents)
    _output_data = _output_data.strip()
    _output_tokens = []
    for line in _output_data.split('\n'):
        token, _, _ = line.partition('\t')
        _output_tokens.append(token)
    
    return _output_tokens

def twitter_tokenize(sent):
    """Tokenizes a sentence using the CMU twitter tokenizer.

    :param str sent: sentence to tokenize
    :return: tokenized sentence
    :rtype: [word, ...]
    """
    return twitter_tokenize_batch([sent])[0]

  #  _output_tokens = twitter_tokenize_batch([sent])
  #  _output_tokens = _output_tokens[0]
  #  return _output_tokens
    