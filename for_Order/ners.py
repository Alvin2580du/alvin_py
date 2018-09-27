import en_core_web_sm
import pandas as pd
from collections import Counter, OrderedDict
import nltk
from nltk.corpus import stopwords

"""
姓名,评论内容,标题


就是我这里需要分析在线评论，有两部分内容，一部分是评论内容，一部分是评论标题，现在需要

1.对于评论内容部分提取其中所有的名词（包括单词以及词组），并统计出词频

2.对于标题部分需要提取出其中所有的名词以及形容词，这个部分既需要将标题汇总后提取统计，也需要单独列出每一标题中的adj、N

"""
parser = en_core_web_sm.load()


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


def adv_find(inputs):
    try:
        out = []
        for sent in parser(inputs).sents:
            for token in sent:
                if token.pos_ == "ADV":
                    out.append(str(token))
        return "|".join(out)
    except:
        return ""


def adj_find_v1(inputs):
    try:
        out = []
        for sent in parser(inputs).sents:
            for token in sent:
                if token.pos_ == "ADJ":
                    out.append(str(token))
        return out
    except:
        return ""


def adv_find_v1(inputs):
    try:
        out = []
        for sent in parser(inputs).sents:
            for token in sent:
                if token.pos_ == "ADV":
                    out.append(str(token))
        return out
    except:
        return ""


def build_one():
    da = pd.read_csv("xuqiu.csv")
    da["评论内容_名词词组"] = da['评论内容'].apply(noun_chunk)
    da['评论内容_名词'] = da['评论内容'].apply(noun_find)
    da['标题_名词'] = da['标题'].apply(noun_find)
    da['标题_形容词'] = da['标题'].apply(adj_find)
    da.to_csv("results__.txt", index=None)
    da1 = pd.DataFrame()

    da1['标题'] = da['标题']
    da1['标题_名词'] = da['标题_名词']
    da1['标题_形容词'] = da['标题_形容词']
    da1.to_csv("results__1.txt", index=None)


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


keys_words = ['place', 'trip', 'entrance', 'step', 'water', 'view', 'park', 'crowd', 'food', 'people', 'tower',
              'mountain', 'experience', 'wall', 'ticket', 'building', 'history', 'cable car', 'bus', 'guide', 'hotel']


def find_Keywrds(inputs):
    inputs_split_juzi = inputs.split(".")
    save = []

    for juzi in inputs_split_juzi:
        for k in keys_words:
            if k in juzi:
                inputs_split = juzi.split(k)
                adj_pre = adj_find(inputs_split[0])
                adv_pre = adv_find(inputs_split[0])
                adj_after = adj_find(inputs_split[1])
                adv_after = adv_find(inputs_split[1])
                if adj_pre or adv_pre:
                    words = "{}|{}|{}|{}".format(adj_pre, adv_pre, adj_after, adv_after)
                    res = "{},{},{}".format(juzi, k, words)
                    save.append(res)
    if len(save) > 0:
        return save
    else:
        return None


def build_three():
    with open("xuqiu_2.csv", 'r', encoding='utf-8') as fr:
        save = []
        C = 0
        while True:
            line = fr.readline()
            if line:
                res = find_Keywrds(line)
                if res:
                    length = len(res)
                    for one in res:
                        rows = OrderedDict()
                        rows['句子'] = ",".join(one.split(",")[:-2])
                        rows['关键字'] = one.split(",")[-2]
                        rows['关键字前后的形容词和副词'] = one.split(",")[-1]
                        rows['标记'] = length
                        save.append(rows)
                        C += 1
                if C % 100 == 0:
                    print('============= Process {}=============  '.format(C))
            else:
                break

        df = pd.DataFrame(save)
        df.to_csv("结果文件.csv", index=None, sep=",", encoding='utf-8')


"""

1.?识别名词关键词前后8个词内的形容词（只识别最近的一个形容就OK）
2.?识别形容词前一个单词是否是副词，若是，则连形容词一起提取出来，若不是，则只提取形容词，此外，如果副词前有no 或者not 也要提取
3.?如果关键词前后8个词内没有形容词，则显示为空
4.标记可以不要
"""


def adj_find_v2(inputs):
    try:
        out = []
        for sent in parser(inputs).sents:
            length = len(sent)
            for i in range(length):
                if sent[i].pos_ == 'ADJ':
                    out.append("{}".format(sent[i]))
                    return out
                if i + 1 < length:
                    if sent[i].pos_ == 'ADV' and sent[i + 1].pos_ == 'ADJ':
                        res1 = "{}|{}".format(sent[i], sent[i + 1])
                        out.append(res1)
                        return out
                if i + 2 < length:
                    if sent[i] == 'no' or sent[i] == 'not' and sent[i + 1].pos_ == 'ADV' and sent[i + 2].pos_ == 'ADJ':
                        res2 = "{}|{}|{}".format(sent[i], sent[i + 1], sent[i + 2])
                        out.append(res2)
                        return out
    except Exception as e:
        print(inputs, e)
        return None


def adj_find_v3(inputs):
    try:
        out = []
        for sent in parser(inputs).sents:
            for token in sent:
                out.append(token.pos_)

    except Exception as e:
        return None


def find_patern(d):
    length = len(d)
    out = []
    for i in range(length):
        if d[i].values() == 'ADJ':
            out.append("{}".format(d[i].keys()))

        if i + 1 < length:
            if d[i].values() == 'ADV' and d[i + 1].values() == 'ADJ':
                res1 = "{}|{}".format(d[i].keys(), d[i + 1].keys())
                out.append(res1)

        if i + 2 < length:
            if d[i].values() == 'no' or d[i].values() == 'not' and d[i + 1].values() == 'ADV' and d[
                        i + 2].values() == 'ADJ':
                res2 = "{}|{}|{}".format(d[i].keys(), d[i + 1].keys(), d[i + 2].keys())
                out.append(res2)
    return out


def apply_fun(inputs, keywords):
    try:
        if keywords:
            if keywords in inputs.split():
                inputs_split = inputs.split(keywords)
                pre = " ".join(inputs_split[0].split(" ")[::-1][:9][::-1])
                after = " ".join(inputs_split[1].split(" ")[:9])
                adj_pre = find_patern(adj_find_v3(pre))
                adj_after = find_patern(adj_find_v3(after))
                print(adj_pre, adj_after)

                if adj_pre[0] and adj_after[-1]:
                    return "{}|{}".format(adj_pre[0], adj_after[-1])
                if adj_pre[0] and not adj_after[-1]:
                    return "{}".format(adj_pre[0])
                if adj_after[-1] and not adj_pre[0]:
                    return "{}".format(adj_after[-1])
        else:
            print("1:{}".format(keywords))
    except Exception as e:
        return " "


def build_four():
    # 句子,关键字,关键字前后的形容词和副词,标记
    save = []
    data = pd.read_csv("结果文件.csv")
    for x, y in data.iterrows():
        juzi = y['句子']
        keywords = y['关键字']
        res = apply_fun(juzi, keywords)
        rows = OrderedDict()
        rows['句子'] = juzi
        rows['关键字'] = keywords
        rows['关键字前后的形容词和副词'] = res
        save.append(rows)

    df = pd.DataFrame(save)
    df.to_csv("关键字前后的形容词和副词.csv", index=None)
    print("关键字前后的形容词和副词.csv 已保存")


"""
    JJ	形容词
    JJR	形容词，比较
    JJS	形容词，最高级
    RB	副词
    RBR	副词，比较
    RBS	副词，最高级
    """
""" 
需求：
    1.把数据里面的形容词全部提取出来，同时如果形容词前有副词也需要提取出来，使之成为词组。
     比如：提取到了形容词 beautiful，之后需要识别beautiful前面有没有副词以及no not，若有，
     则将其一起提出出来成为一个词，若没有则只提取形容词。
    2.将提取出来的形容词以及副词形容词组成词典，由我来将词典按照不同分值分开成12345分的词典。
    3.历遍每条评论，识别20个名词关键词，在关键词前后8个词内匹配上述有分值的词典，若匹配上，
    则按已经给定的分值给关键词赋分，若匹配不上，则按0分赋值。
"""


def pos_find(text):
    text_list = nltk.word_tokenize(text)
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '’', "''",
                            '``']
    text_list = [word for word in text_list if word not in english_punctuations]
    # 去掉停用词
    stops = set(stopwords.words("english"))
    text_list = [word for word in text_list if word not in stops]
    pos_list = nltk.pos_tag(text_list)

    output = []
    length = len(pos_list)
    for i in range(length):
        if i + 1 < length:
            if 'RB' in pos_list[i][1] and 'JJ' in pos_list[i + 1][1]:
                # 副词+形容词
                BRJJ = '{} {}'.format(pos_list[i][0], pos_list[i + 1][0])
                output.append(BRJJ)

            if 'JJ' in pos_list[i][1]:
                # 形容词
                JJ = pos_list[i][0]
                output.append(JJ)
            if pos_list[i][0] == 'no' or pos_list[i][0] == 'not' and pos_list[i + 1][1] == 'JJ':
                # no not + 形容词
                notJJ = '{} {}'.format(pos_list[i][0], pos_list[i + 1][0])

                output.append(notJJ)

    return output


def build_five():
    all_cidian = []

    with open("xuqiu_2.csv", 'r', encoding='utf-8') as fr:
        k = 0

        while True:
            line = fr.readline()
            if line:
                k += 1
                res = pos_find(line)
                if len(res) > 0:
                    for x in res:
                        if x not in all_cidian:
                            if 'www' in x:
                                continue
                            if len(x) > 1 and str(x).isalpha():
                                all_cidian.append(x)
                if k % 1000 == 1:
                    print(k)
            else:
                break

    df = pd.DataFrame(all_cidian)
    df.to_csv("词典.csv", index=None, header=None, encoding='utf-8')
    print("词典已保存， 数量为{}".format(df.shape))


"""
    3.历遍每条评论，识别20个名词关键词，在关键词前后8个词内匹配上述有分值的词典，若匹配上，
    则按已经给定的分值给关键词赋分，若匹配不上，则按0分赋值。

"""
cidian_score = pd.read_csv("cidian_score.csv")
cidian_score_dict = {}

for x, y in cidian_score.iterrows():
    name = y['name']
    score = y['score']
    cidian_score_dict[name] = score


def juzi2list(juzi):
    sentences = juzi.split('.')
    save = []
    for sen in sentences:
        out = []
        for x in sen.split():
            out.append(x.replace(".", ""))
        save.append(out)
    return save


def get_eight_words(l, x):
    if x in l:
        index = l.index(x)
        if index < 8:
            left = l[:index + 1]
            right = l[index: index + 8]
            all_words = left + right
            return " ".join(all_words)

        else:
            left = l[index - 8:index + 1]
            right = l[index: index + 8]
            all_words = left + right
            return " ".join(all_words)
    else:
        return 0


def make_dict(find_socre):
    keys_words_dict = {"place": 0, 'trip': 0, 'entrance': 0, 'step': 0, 'water': 0, 'view': 0, 'park': 0, 'crowd': 0,
                       'food': 0, 'people': 0, 'tower': 0, 'mountain': 0, 'experience': 0, 'wall': 0, 'ticket': 0,
                       'building': 0, 'history': 0, 'cable car': 0, 'bus': 0, 'guide': 0, 'hotel': 0}

    for x in find_socre:
        keys_words_dict[list(x.keys())[0]] = list(x.values())[0]
    return keys_words_dict


def get_sentence_score(sentence):
    res = []
    for x in keys_words:
        max_score = get_eight_words(sentence, x)
        try:
            for jj in list(cidian_score_dict.keys()):
                if jj in max_score:
                    score = cidian_score_dict[jj]
                    rows = {x: score}
                    res.append(rows)
                else:
                    rows = {x: 4}
                    res.append(rows)
        except Exception as e:
            continue
    return res


def get_biaoti_score(line):
    all_sentences = juzi2list(line)
    save = []

    for sentence in all_sentences:
        res = get_sentence_score(sentence)
        for x in res:
            if x not in save:
                save.append(x)
    return save


def build_six():
    with open("xuqiu_2.csv", 'r', encoding='utf-8') as fr:
        k = 0
        limit = 100000000
        save_df = []
        while True:
            line = fr.readline()
            if line:
                k += 1
                if k == 1:
                    continue
                get_biaoti_score_res = get_biaoti_score(line=line)
                get_dict = make_dict(get_biaoti_score_res)
                get_dict['标题'] = line.replace("\n", "")
                save_df.append(get_dict)
                if k > limit:
                    break
            else:
                break
        df = pd.DataFrame(save_df)
        df.to_csv("评论内容——分数.csv", index=None, encoding='utf-8')


def biaoti_adj_find(text):
    text_list = nltk.word_tokenize(text)
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '’',
                            "''",
                            '``']
    text_list = [word for word in text_list if word not in english_punctuations]
    # 去掉停用词
    stops = set(stopwords.words("english"))
    text_list = [word for word in text_list if word not in stops]
    pos_list = nltk.pos_tag(text_list)

    adj_out = []
    length = len(pos_list)
    for i in range(length):
        if 'JJ' in pos_list[i][1]:
            adj_out.append(pos_list[i][0])

    return " ".join(adj_out)


def biaoti_nn_find(text):
    text_list = nltk.word_tokenize(text)
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '’',
                            "''",
                            '``']
    text_list = [word for word in text_list if word not in english_punctuations]
    # 去掉停用词
    stops = set(stopwords.words("english"))
    text_list = [word for word in text_list if word not in stops]
    pos_list = nltk.pos_tag(text_list)

    nn_out = []
    length = len(pos_list)
    for i in range(length):
        if 'NN' in pos_list[i][1]:
            nn_out.append(pos_list[i][0])

    return " ".join(nn_out)


if __name__ == '__main__':
    method = 'build_one'

    if method == 'build_one':
        build_one()

    if method == 'build_two':
        build_two()

    if method == 'build_three':
        build_three()

    if method == 'build_four':
        build_four()

    if method == 'test':
        d = OrderedDict({1: 2, 3: 4})
        length = len(d)
        for i in range(length):
            print(d[i])

    if method == 'build_five':
        build_five()

    if method == 'build_six':
        build_six()

    if method == 'build_seven':
        data = pd.read_excel("评论标题.xlsx", sheet_name='Sheet1')
        data['形容词'] = data['标题'].apply(biaoti_adj_find)
        data['名词'] = data['标题'].apply(biaoti_nn_find)

        data.to_csv("标题——结果.csv", index=None, encoding='utf-8')
