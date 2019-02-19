import pandas as pd
from collections import defaultdict
import jieba
import matplotlib.pyplot as plt
import wordcloud
from collections import Counter

"""苏州高铁新城评论 情感分析 """

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

build = False


def readLines(filename):
    out = []
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            out.append(line.replace("\n", ""))
    return out


stopwords = readLines('./resources/stopwords.txt')


def sent2word(sentence):
    if isinstance(sentence, str):
        segList = jieba.lcut(sentence.strip().replace("\n", ""))
        newSent = [i for i in segList if i not in stopwords]
        return newSent
    else:
        return sentence


def get_wordDicts():
    words = {}
    data = pd.read_csv("./raw_data/dataAll.csv")
    lines = data['content'].values
    num = 0
    for line in lines:
        try:
            newSent = sent2word(line)
            for x in newSent:
                if x not in words.keys():
                    words[x] = num
                    num += 1
        except:
            continue
    return words


def classifyWords(wordList):
    # (1) 情感词
    with open("./resources/BosonNLP_sentiment_score.txt", 'r', encoding='utf8') as f:
        senList = f.readlines()
    senDict = defaultdict()
    for s in senList:
        try:
            senDict[s.split(' ')[0]] = s.split(' ')[1].strip('\n')
        except Exception:
            pass
    # (2) 否定词
    with open("./resources/notDict.txt", 'r', encoding='utf-8') as f:
        notList = f.read().splitlines()
    # (3) 程度副词
    with open("./resources/degreeDict.txt", 'r', encoding='utf-8') as f:
        degreeList = f.read().splitlines()
    degreeDict = defaultdict()
    for index, d in enumerate(degreeList):
        if 3 <= index <= 71:
            degreeDict[d] = 2

        elif 73 <= index <= 114:
            degreeDict[d] = 1.25

        elif 117 <= index <= 153:
            degreeDict[d] = 1.2

        elif 156 <= index <= 184:
            degreeDict[d] = 0.8

        elif 187 <= index <= 198:
            degreeDict[d] = 0.5

        elif 201 <= index <= 230:
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


if build:
    wordsDicts = get_wordDicts()
    print(wordsDicts)
    senWord, notWord, degreeWord = classifyWords(wordsDicts)
    print(senWord)
    print(notWord)
    print(degreeWord)


def get_scores(line):
    try:
        newSent = sent2word(line)
        print(newSent)
        scores = scoreSent(senWord, notWord, degreeWord, newSent)
        return scores
    except Exception as e:
        return ""


def build_last():
    data = pd.read_csv("./raw_data/dataAll.csv")
    data['sentiments'] = data['content'].apply(get_scores)
    data.to_csv("./raw_data/dataAll_scores.csv")


data = pd.read_csv("./raw_data/dataAll_scores.csv")


def plot_value():
    plt.figure()
    plt.plot(data['sentiments'].values.tolist())
    plt.savefig("sentiments_value.png")
    plt.show()
    plt.close()


def static():
    pos = 0
    pos_sum = 0
    pos_max = 0
    pos_five = 0
    neg = 0
    neg_sum = 0
    neg_max = 0
    eos = 0
    eos_sum = 0
    shape = data.shape[0]
    for x in data['sentiments'].values.tolist():
        if x > 0:
            pos += 1
            pos_sum += x
            if x > pos_max:
                pos_max = x
            if x > 5:
                pos_five += 1
        elif x < 0:
            neg += 1
            neg_sum += x
            if x < neg_max:
                neg_max = x
        else:
            eos += 1
            eos_sum += x

    pos_mean = pos_sum / pos
    neg_mean = neg_sum / neg
    eos_mean = eos_sum / eos
    print("- " * 10)
    print("积极频数 频率 积极均值 积极极值 评分>5 比重")
    print(pos, "{:0.3f}".format(pos / shape), pos_mean, pos_max, pos_five, "{:0.3f}".format(pos_five / pos))
    print("- " * 10)
    print("消极频数 频率 消极均值 消极极值 ")
    print(neg, "{:0.3f}".format(neg / shape), neg_mean, neg_max, )
    print("- " * 10)
    print("中立频数 频率 中立均值")
    print(eos, "{:0.3f}".format(eos / shape), eos_mean)
    print("- " * 10)


def build_diff_source():
    plt.figure()
    for x, y in data.groupby(by='src'):
        if y.shape[0] < 100:
            continue
        scores = y['sentiments'].values[:100]
        plt.plot(scores, label=x)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("情感值")
    plt.savefig("不同来源数据的情感倾向.png")


def cipin():
    data = pd.read_csv("./raw_data/dataAll.csv")
    lines = data['content'].values
    words = []
    for line in lines:
        try:
            newSent = sent2word(line)
            for x in newSent:
                words.append(x)
        except:
            continue
    df = pd.DataFrame(words)
    df.to_csv("words_all.csv", index=None)

    save = []
    for x, y in Counter(words).most_common(1000):
        rows = {'word': x, 'freq': y}
        save.append(rows)
    df1 = pd.DataFrame(save)
    df1.to_csv("wordfreq.csv", index=None)


def plot_word_cloud(file_name, savename):
    text = open(file_name, 'r', encoding='utf-8').read()
    wc = wordcloud.WordCloud(background_color="white", width=800, height=600,
                             max_font_size=50,
                             random_state=1,
                             max_words=1000,
                             font_path='msyh.ttf')
    wc.generate(text)
    plt.axis("off")
    plt.figure(dpi=600)
    wc.to_file(savename)


def make_decise(inputs):
    if inputs > 0:
        return 'pos'
    if inputs == 0:
        return 'eos'
    if inputs < 0:
        return "neg"


def get_pos_data():
    data = pd.read_csv("./raw_data/dataAll_scores.csv", usecols=['sentiments', 'content'])
    data['senti'] = data['sentiments'].apply(make_decise)
    pos = data[data['senti'].isin(['pos'])]
    neg = data[data['senti'].isin(['neg'])]
    print(pos.head())
    pos['cut'] = pos['content'].apply(sent2word)
    neg['cut'] = neg['content'].apply(sent2word)
    del pos['content']
    del neg['content']
    del pos['sentiments']
    del neg['sentiments']
    del pos['senti']
    del neg['senti']

    pos.to_csv("pos_content.csv", index=None)
    neg.to_csv("neg_content.csv", index=None)


if __name__ == "__main__":
    method = 'build_diff_source'

    if method == 'build_last':
        build_last()

    if method == 'static':
        static()

    if method == 'get_pos_data':
        get_pos_data()

    if method == 'build_diff_source':
        build_diff_source()

    if method == 'plot_word_cloud':
        plot_word_cloud("neg_content.csv", "neg_content.png")
