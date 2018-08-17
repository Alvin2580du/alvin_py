"""
The Sussex NLTK package provides extensions to the functionality provided by
the standard NLTK distribution, along with additional corpora.

.. codeauthor::
    Simon Wibberly
    Matti Lyra
"""

import os
from os.path import dirname
import sys
#from exceptions import EnvironmentError 
import multiprocessing as mp
import re
from nltk.tag import untag


# set the root directory of the package if it has not been set before
try:
    _sussex_root
except NameError:
    _sussex_root = dirname(dirname(sys.modules[__name__].__file__))    
    print('Sussex NLTK root directory is', _sussex_root)


def _set_root(root):
    if not os.path.exists(root):
        raise EnvironmentError('The specified root path (%s) does not exist.'%root)
    
    if not os.path.exists(os.path.join(root, 'data')):
        raise EnvironmentError('Can not find directory \'data\' under root (%s).'%root)
    
    if not os.path.exists(os.path.join(root, 'stanford')):
        raise EnvironmentError('Can not find directory \'stanford\' under root (%s).'%root)
    
    if not os.path.exists(os.path.join(root, 'CMU')):
        raise EnvironmentError('Can not find directory \'CMU\' under root (%s).'%root)
    
    print('Setting root for sussex_nltk package to %s'%(root))
    global _sussex_root
    _sussex_root = root


def _lemmatize_tagged(tagged):
    from nltk.tag import simplify_tag
    from nltk.corpus.reader import wordnet 
    from nltk.stem.wordnet import WordNetLemmatizer

    wordnet_pos = dict( list(zip( wordnet.POS_LIST, wordnet.POS_LIST )) )
    lemmatizer = WordNetLemmatizer()

    wordnet_tag = wordnet_pos.get(simplify_tag(tagged[1]).lower(), "n")
    lemma = lemmatizer.lemmatize(tagged[0], wordnet_tag ), tagged[1]
    return lemma


def _extract_by_pos(tagged_sequences, regex_patterns):
    extracted = []
    for tagged in tagged_sequences:
        match = None
        for pattern in regex_patterns:
            if isinstance(pattern, str):
                match = match or re.match(pattern, tagged[1])
            else:
                match = True
                for tagged_token, regex in zip(tagged, pattern):
                    match = match and re.match(regex, tagged_token[1])
        if match:
            extracted.append(tagged)
    return extracted


def _untag_sequence(tagged):
    try:
        if isinstance(tagged[0][0], str):
            return tuple( untag(tagged) )
        else :
            return [tuple(untag(t)) for t in tagged]
    except IndexError:
        return []