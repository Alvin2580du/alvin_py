import json
import pandas as pd
from collections import OrderedDict
import re
import jieba.analyse
from gensim.summarization.summarizer import summarize


def fun2():
    da = 'evaluation_without_ground_truth.txt'
    fr = open(da, 'r', encoding='utf-8')
    lines = fr.readlines()
    res = []
    for line in lines:
        line2json = json.loads(line)

        rows = OrderedDict(
            {'index': line2json['index'], 'article': line2json['article'], "summarization": line2json['summarization']})
        res.append(rows)
    df = pd.DataFrame(res)
    df.to_csv("test.csv", index=None)


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def fun3():
    da = './datasets/train_with_summ.txt'
    fr = open(da, 'r', encoding='utf-8')
    lines = fr.readlines()
    res = []
    for line in lines:
        line2json = json.loads(line)
        p = re.compile('\(来源:\w+.?\)').findall(line2json['article'])
        if p:
            content = line2json['article'].replace("<Paragraph>", "").replace(p[0], "")
        else:
            content = line2json['article'].replace("<Paragraph>", "")
        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
        new_summ = re.sub(r, "", line2json['summarization'])
        tmp_summ = re.sub('\w+.?', "", new_summ)
        if len(tmp_summ) == 0:
            continue

        rows = OrderedDict(
            {'article': content, "summarization": line2json['summarization']})
        res.append(rows)
    df = pd.DataFrame(res)
    df.to_csv("./datasets/train_with_summ.csv", index=None)


def fun4():
    data = pd.read_csv("./datasets/train_with_summ.csv")
    for one in data.values:
        text, summ = one[0], one[1]
        print(text)
        p = summarize(text)
        print(p)
        exit(1)


fun4()
