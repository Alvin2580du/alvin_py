import re
import os
import collections
import json
import jieba
import jieba.posseg as pseg
from pyduyp.config.conf import get_dictionary
import jieba.analyse
from pyduyp.utils.utils import replace_symbol
import pandas as pd
from pyduyp.logger.log import log

args = get_dictionary()
not_cuts = re.compile(u'([\da-zA-Z \.]+)|《(.*?)》|“(.{1,10})”')
re_replace = re.compile(u'[^\u4e00-\u9fa50-9a-zA-Z《》\(\)（）“”·\.]')
jieba.load_userdict(os.path.join(args.get('path'), 'jieba_dict.csv'))
jieba.analyse.set_stop_words(os.path.join(args.get('path'), 'stopwords_zh.csv'))
sw = pd.read_csv("pyduyp/dictionary/stopwords_zh.csv", lineterminator="\n").values.tolist()
sw2list = [j for i in sw for j in i]
log.debug("dict load success")


def cut(s, add_stopwords=True):
    out = []
    scut = jieba.lcut(s)
    for x in scut:
        if add_stopwords:
            if x not in sw2list:
                out.append(x)
        else:
            out.append(x)
    return out


def posseg(wl):
    if len(str(wl)) == 0:
        return ''
    words = pseg.cut(wl)
    d = collections.OrderedDict()
    for w in words:
        d[w.word] = w.flag
    return json.dumps(d)


def posseg_dict(wl):
    if len(str(wl)) == 0:
        return ''
    words = pseg.cut(wl)
    d = {}
    for w in words:
        d[w.word] = w.flag
    return d


def posseg_list_by_type(wl, types):
    if len(str(wl)) == 0:
        return ''
    words = pseg.cut(wl)
    l = []
    for w in words:
        if w.flag in types and w.word not in l and len(w.word) > 1:
            l.append(w.word)
    return l


def posegcut(inputs):
    if isinstance(inputs, str):
        msg_cut = jieba.posseg.lcut(replace_symbol(inputs))
        _msg_cut = [i for i in msg_cut if i not in sw2list]

        msg_cut_tags = []
        for w in _msg_cut:
            wf = "{}_{}".format(w.word, w.flag)
            msg_cut_tags.append(wf)
        p = re.compile("[0-9]+?[元|块]").findall(inputs)
        if p:
            for price in p:
                msg_cut_tags.append("{}_{}".format(price, 'n'))
        if len(msg_cut_tags) > 0:
            return "|".join(msg_cut_tags)
        else:
            return inputs
    else:
        return inputs
