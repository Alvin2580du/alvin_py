import pandas as pd
from collections import defaultdict
import jieba
from gensim import corpora, models

"""

需求：
文本情感
主题模型

"""


def readLines(filename):
    out = []
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            out.append(line.replace("\n", ""))
    return out


stopwords = readLines('stopwords.txt')


def sent2word(sentence):
    """
    Segment a sentence to words
    Delete stopwords
    """
    segList = jieba.lcut(sentence)
    newSent = [i for i in segList if i not in stopwords]
    return newSent


def classifyWords(wordList):
    # (1) 情感词
    with open("BosonNLP_sentiment_score.txt", 'r', encoding='utf8') as f:
        senList = f.readlines()
    senDict = defaultdict()
    for s in senList:
        try:
            senDict[s.split(' ')[0]] = s.split(' ')[1].strip('\n')
        except Exception:
            pass
    # (2) 否定词
    with open("notDict.txt", 'r', encoding='utf-8') as f:
        notList = f.read().splitlines()
    # (3) 程度副词
    with open("degreeDict.txt", 'r', encoding='utf-8') as f:
        degreeList = f.read().splitlines()
    degreeDict = defaultdict()
    for index, d in enumerate(degreeList):
        if 3 <= index <= 71:
            degreeDict[d] = 2
        elif 74 <= index <= 115:
            degreeDict[d] = 1.25
        elif 118 <= index <= 154:
            degreeDict[d] = 1.2
        elif 157 <= index <= 185:
            degreeDict[d] = 0.8
        elif 188 <= index <= 199:
            degreeDict[d] = 0.5
        elif 202 <= index <= 231:
            degreeDict[d] = 1.5
        else:
            pass

    senWord = defaultdict()
    notWord = defaultdict()
    degreeWord = defaultdict()

    for index, word in enumerate(wordList):
        if word in senDict.keys() and word not in notList and word not in degreeDict.keys():
            senWord[index] = senDict[word]
        elif word in notList and word not in degreeDict.keys():
            notWord[index] = -1
        elif word in degreeDict.keys():
            degreeWord[index] = degreeDict[word]
    return senWord, notWord, degreeWord


def scoreSent(senWord, notWord, degreeWord, segResult):
    W = 1
    score = 0
    # 存所有情感词的位置的列表
    senLoc = list(senWord.keys())
    notLoc = list(notWord.keys())
    degreeLoc = list(degreeWord.keys())
    senloc = -1
    # notloc = -1
    # degreeloc = -1

    # 遍历句中所有单词segResult，i为单词绝对位置
    for i in range(0, len(segResult)):
        # 如果该词为情感词
        if i in senLoc:
            # loc为情感词位置列表的序号
            senloc += 1
            # 直接添加该情感词分数
            score += W * float(senWord[i])
            # print "score = %f" % score
            if senloc < len(senLoc) - 1:
                # 判断该情感词与下一情感词之间是否有否定词或程度副词
                # j为绝对位置
                for j in range(senLoc[senloc], senLoc[senloc + 1]):
                    # 如果有否定词
                    if j in notLoc:
                        W *= -1
                    # 如果有程度副词
                    elif j in degreeLoc:
                        W *= float(degreeWord[j])
        # i定位至下一个情感词
        if senloc < len(senLoc) - 1:
            i = senLoc[senloc + 1]
    return score


def get_wordDicts():
    words = {}

    with open('comment_12,9.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        num = 0
        for line in lines:
            newSent = sent2word(line.strip().replace("\n", ""))
            for x in newSent:
                if x not in words.keys():
                    words[x] = num
                    num += 1
    return words


def build_sentments():
    wordsDicts = get_wordDicts()
    senWord, notWord, degreeWord = classifyWords(wordsDicts)
    save = []
    with open('comment_12,9.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            newSent = sent2word(line.strip().replace("\n", ""))
            scores = scoreSent(senWord, notWord, degreeWord, newSent)
            if scores >= 0:
                rows = {"sents": line.strip().replace("\n", ""), 'scores': 'pos'}
            else:
                rows = {"sents": line.strip().replace("\n", ""), 'scores': 'neg'}
            save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("携程亲子园评论数据情感得分.xlsx", index=None)
    print(df.shape)


def get_stop_words_set(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return set([line.strip() for line in file])


def get_words_list(file_name, stop_word_file):
    stop_words_set = get_stop_words_set(stop_word_file)
    word_list = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            tmp_list = list(jieba.cut(line.strip(), cut_all=False))
            word_list.append(
                [term for term in tmp_list if str(term) not in stop_words_set])  # 注意这里term是unicode类型，如果不转成str，判断会为假
    return word_list


def build_lda():
    print("build_lda ")

    raw_msg_file = 'comment_12,9.txt'
    stop_word_file = "stopwords.txt"

    word_list = get_words_list(raw_msg_file, stop_word_file)  # 列表，其中每个元素也是一个列表，即每行文字分词后形成的词语列表
    word_dict = corpora.Dictionary(word_list)  # 生成文档的词典，每个词与一个整型索引值对应
    corpus_list = [word_dict.doc2bow(text) for text in word_list]  # 词频统计，转化成空间向量格式
    print("corpus_list:", len(corpus_list))
    lda = models.ldamodel.LdaModel(corpus=corpus_list, id2word=word_dict, num_topics=20, alpha='auto')

    output_file = 'lda_output.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for pattern in lda.show_topics(num_topics=5):
            print(pattern)
            f.writelines("{},{}\n".format(pattern[0], pattern[1]))


if __name__ == '__main__':

    method = 'build_sentments'

    if method == 'build_sentments':
        build_sentments()

    if method == 'build_lda':
        build_lda()
