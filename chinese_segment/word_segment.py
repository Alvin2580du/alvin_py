# coding:utf-8
import jieba
import re
import jieba.analyse
import jieba.posseg
import pandas as pd


from pyduyp.utils.utils import replace_symbol

jieba.load_userdict("chinese_segment/dictionary/jieba_dict.csv")
jieba.analyse.set_stop_words("chinese_segment/dictionary/stopwords_zh.csv")
sw = pd.read_csv("chinese_segment/dictionary/stopwords_zh.csv", lineterminator="\n").values.tolist()
sw2list = [j for i in sw for j in i]


def final():
    f = open("荷塘月色.txt", "r")
    lines = f.readlines()
    out = open("result.txt", 'w')
    for line in lines:
        res = line.replace("\n", "").strip()
        symbol = '。！？……'
        res = re.split(symbol, res)
        stopwords = {}.fromkeys([line.rstrip() for line in open('stopwords.txt')])
        for r in res:
            segs = jieba.cut(r, cut_all=False)
            for seg in segs:
                if seg not in stopwords:
                    print(seg, sep='\n', end='\n', file=out)


def seg_label():
    out = []
    s = ''
    s_cut = jieba.posseg.lcut(s)
    for w in s_cut:
        flag, word = w.flag, w.word
        out.append("{},{}".format(flag, word))
    return out


def cut_jieba(inputs):
    if isinstance(inputs, str):
        res = jieba.analyse.extract_tags(replace_symbol(inputs), topK=10)
        keywords = '|'.join(res)

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
            return "|".join(msg_cut_tags), keywords
        else:
            return inputs, keywords
    else:
        return inputs, inputs

if __name__ == "__main__":
    final()
