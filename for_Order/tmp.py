import nltk
from nltk.corpus import stopwords
import pandas as pd
import re

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


def is_alpha(inputs):
    inputs_sp = inputs.split()
    out = False
    for x in inputs_sp:
        if x.isalpha():
            out = True
    return out


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
                            if len(x) > 1 and is_alpha(x):
                                all_cidian.append(x)
                if k % 1000 == 1:
                    print(k)
            else:
                break
        print("{}".format(k))
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


keys_words = ['place', 'trip', 'entrance', 'step', 'water', 'view', 'park', 'crowd', 'food', 'people', 'tower',
              'mountain', 'experience', 'wall', 'ticket', 'building', 'history', 'cable car', 'bus', 'guide',
              'hotel']


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


def build():
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
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '’', "''",
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
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '’', "''",
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


data = pd.read_excel("评论标题.xlsx", sheet_name='Sheet1')
data['形容词'] = data['标题'].apply(biaoti_adj_find)
data['名词'] = data['标题'].apply(biaoti_nn_find)

data.to_csv("标题——结果.csv", index=None, encoding='utf-8')
