'''
Created on Nov 15, 2012

The sussex_nltk.parse module provides access to a Python implementation of
Arc-Eager dependency parsing.

  Deterministic dependency parsing is a robust and efficient approach to
  syntactic parsing of unrestricted natural language text.

`Link Incrementality in Deterministic Dependency Parsing <http://dl.acm.org/citation.cfm?doid=1613148.1613156>`_

@author: mattilyra
'''

import os
try:
    import pickle as pickle
except ImportError:
    import pickle

import sussex_nltk as susx
from sussex_nltk.depparse.parsing import experimentation


def dep_parse_sentences_arceager(pos_sents):
    """Dependency parse a list of part-of-speech tagged sentences.

    The `pos_sents` list should be a nested list of tagged sentences.

    `[[('John', 'NNP'), ("'s", 'POS'), ('big', 'JJ'), ('idea', 'NN'), ...]]`


    :param pos_sents: list of POS tagged sentences
    :return: list of `ParsedSentence` objects.
    """
    penn_stanford_index = os.path.join(susx._sussex_root,
                                       'depparse_model',
                                       'penn-stanford-index-susx')
    penn_stanford_model = os.path.join(susx._sussex_root,
                                       'depparse_model',
                                       'penn-stanford-model')

    return list(experimentation.parse_sentences(pos_sents,
                                                penn_stanford_index,
                                                penn_stanford_model))


def get_parser():
    """Get a reference to the trained parser.

    :return: trained DependencyParser object
    """
    penn_stanford_index = os.path.join(susx._sussex_root,
                                       'depparse_model',
                                       'penn-stanford-index-susx')
    penn_stanford_model = os.path.join(susx._sussex_root,
                                       'depparse_model',
                                       'penn-stanford-model')
    return experimentation.get_parser(penn_stanford_index, penn_stanford_model)


def parse(parser, sents):
    """Parses `sents` using `parser`.

    :param DependencyParser parser: trained dependency parser
    :param list sents: pos tagged sentences
    :return: list of parsed sentences
    """
    return list(experimentation.parse_with_parser(parser, sents))


def load_parsed_dvd_sentences(query):
    """Load a list of pre parsed sentences from the movie category.

    Parsing a large number of sentences can be quite slow. This is a
    convenience method for accessing pre parsed sentences from the Amazon
    Movie Reviews corpus.

    :param str query: one of [plot, acting, dialogue, effects, characters,
        cinematography, choreography]
    :return: a list of parsed sentences
    """
    terms = ['plot', 'acting', 'dialogue', 'effects', 'characters',
             'cinematography', 'choreography']
    if query not in terms:
        raise ValueError('Unacceptable query term \'%s\'. Valid query terms '
                         'are %s' % (query, ','.join()))

    return load_parsed_sentences(os.path.join(susx._sussex_root,
                                              'sussex_nltk',
                                              '%s.pickle' % query))


def load_parsed_example_sentences():
    """Load a list of pre parsed sentences for testing.

    Parsing a large number of sentences can be quite slow. This is a
    convenience method for accessing a small list of example sentences. Due
    to the limited number of example sentences this method should only be
    used for development and testing, not production code.

    :param str query: one of [plot, acting, dialogue, effects, characters,
        cinematography, choreography]
    :return: a list of parsed sentences
    """
    return load_parsed_sentences(os.path.join(susx._sussex_root,
                                              'sussex_nltk',
                                              'example_sents.pickle'))


def load_parsed_sentences(fname):
    """Load a list of pre parsed of sentences from disk.

    Parsing a large number of sentences can be quite slow. This function allows
    you to load a list of parsed sentences. The functions uses `pickle`
    internally.

    :param str fname: file path to a previously saved cache of sentences
    :return: list of sentences
    """
    with open(fname, 'rb') as infh:
        sents = pickle.load(infh)
    return sents


def save_parsed_sentences(fname, sents):
    """Save a list of parsed sentences to a file.

    Parsing a large number of sentences can be quite slow. This function allows
    you to save a list of parsed sentences to be used many times. The function
    will internally use `pickle` to dump the list of sentences to disk as a
    binary file format.

    :param fname: file path where to save the sentences to.
    :param sents:
    :return:
    """
    with open(fname, 'wb') as outfh:
        pickle.dump(sents, outfh, protocol=pickle.HIGHEST_PROTOCOL)

