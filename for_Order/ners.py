import en_core_web_sm
import pandas as pd
from collections import Counter

"""
姓名,评论内容,标题


就是我这里需要分析在线评论，有两部分内容，一部分是评论内容，一部分是评论标题，现在需要

1.对于评论内容部分提取其中所有的名词（包括单词以及词组），并统计出词频

2.对于标题部分需要提取出其中所有的名词以及形容词，这个部分既需要将标题汇总后提取统计，也需要单独列出每一标题中的adj、N

"""
parser = en_core_web_sm.load()
da = pd.read_csv("xuqiu.csv")


def noun_chunk(inputs):
    try:
        doc = parser(inputs)
        chunk_text = []
        for chunk in doc.noun_chunks:
            chunk_text.append(chunk.text)
        return "|".join(chunk_text)
    except:
        return ""


def noun_find(inputs):
    try:
        out = []
        for sent in parser(inputs).sents:
            for token in sent:
                if token.pos_ == "NOUN":
                    out.append(str(token))
        return "|".join(out)
    except:
        return ""


def adj_find(inputs):
    try:
        out = []
        for sent in parser(inputs).sents:
            for token in sent:
                if token.pos_ == "ADJ":
                    out.append(str(token))
        return "|".join(out)
    except:
        return ""


def build_one():
    da["评论内容_名词词组"] = da['评论内容'].apply(noun_chunk)
    da['评论内容_名词'] = da['评论内容'].apply(noun_find)
    da['标题_名词'] = da['标题'].apply(noun_find)
    da['标题_形容词'] = da['标题'].apply(adj_find)
    da.to_csv("results__.txt", index=None)


def splits(inputs):
    out = []
    for x in inputs:
        try:
            xx = x.split("|")
            for i in xx:
                out.append(i)
        except:
            continue
    return out


def build_two():
    da = pd.read_csv("results__.txt")
    num = 10000000
    save1 = []
    for x, y in Counter(splits(da['评论内容_名词词组'].values.tolist())).most_common(num):
        rows = {'评论内容_名词词组': x, '词频': y}
        save1.append(rows)
    df1 = pd.DataFrame(save1)
    df1.to_csv("评论内容_名词词组_频率.txt", index=None, encoding='utf-8')

    save2 = []
    for x, y in Counter(splits(da['评论内容_名词'].values.tolist())).most_common(num):
        rows = {'评论内容_名词': x, '词频': y}
        save2.append(rows)
    df2 = pd.DataFrame(save1)
    df2.to_csv("评论内容_名词_频率.txt", index=None, encoding='utf-8')

    save3 = []
    for x, y in Counter(splits(da['标题_名词'].values.tolist())).most_common(num):
        rows = {'标题_名词': x, '词频': y}
        save3.append(rows)
    df3 = pd.DataFrame(save1)
    df3.to_csv("标题_名词_频率.txt", index=None, encoding='utf-8')

    save4 = []
    for x, y in Counter(splits(da['标题_形容词'].values.tolist())).most_common(num):
        rows = {'标题_形容词': x, '词频': y}
        save4.append(rows)
    df4 = pd.DataFrame(save1)
    df4.to_csv("标题_形容词_频率.txt", index=None, encoding='utf-8')


if __name__ == '__main__':
    method = 'build_two'

    if method == 'build_one':
        build_one()

    if method == 'build_two':
        build_two()
