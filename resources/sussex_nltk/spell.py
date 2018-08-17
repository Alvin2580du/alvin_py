"""
.. codeauthor::
    Matti Lyra
"""

import os
from exceptions import AttributeError

import sussex_nltk as susx


_dict_types = ['aspell']
_dict_languages = ['en', 'en_GB', 'en_US', 'en_CA']


def _read_word_list(file_handle):
    wl = set()
    for line in file_handle:
        wl.add(line.strip())
    return wl


def dictionary(dict_type='aspell', dict_language='en_GB'):
    wordlist = set()
    join = os.path.join
    fpath = join(susx._sussex_root, 'data', 'aspell', 'en-common.wl')
    with open(fpath, 'r') as fh:
        wordlist.union(_read_word_list(fh))
    
    if dict_type not in _dict_types:
        raise AttributeError('Unrecognized dictionary type (%s), must be '
                             'one of %s.'%(dict_type, ' '.join(_dict_types)))
    
    if dict_language not in _dict_languages:
        raise AttributeError('Unrecognized dictionary language (%s), must be '
                             'one of %s.'
                             % (dict_language, ' '.join(_dict_languages)))
    
    wl_files = []
    if dict_type == 'aspell':
        wl_files = os.listdir(os.path.join(susx._sussex_root, 'data','aspell'))
        wl_files = [f for f in wl_files if f.startswith(dict_language)]
        for wl in wl_files:
            with open(join(susx._sussex_root,'data','aspell',wl),'r') as fh:
                wordlist = wordlist.union(_read_word_list(fh))
    
    return wordlist