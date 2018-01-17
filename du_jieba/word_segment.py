# coding:utf-8
import jieba
import re

from jieba import posseg as pseg


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
    s_cut = pseg.lcut(s)
    for w in s_cut:
        flag, word = w.flag, w.word
        out.append("{},{}".format(flag, word))
    return out


if __name__ == "__main__":
    final()
