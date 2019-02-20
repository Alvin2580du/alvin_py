import pandas as pd
from collections import Counter
import jieba
import matplotlib.pyplot as plt
import wordcloud
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据集名称
data_name = 'data.xlsx'
# 计算情感得分的时候，设置全局变量build为True， 否则为False
build = False


def readLines(filename):
    out = []
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            out.append(line.replace("\n", ""))
    return out


stopwords = readLines('./cidian/stopwords.txt')


def cipin():
    # 词频统计
    data = pd.read_excel("./datasets/{}".format(data_name))
    print(data.shape)

    vocal = []
    for one in data['content'].values:
        one_cut = [i for i in jieba.lcut(one) if i not in stopwords and i is not None]
        for x in one_cut:
            vocal.append(x)

    save = []
    for x, y in Counter(vocal).most_common(10000000):
        rows = {'word': x, 'freq': y}
        save.append(rows)
    df = pd.DataFrame(save)
    df.to_csv("wordfreq.csv", index=None)


def plot_word_cloud(file_name, savename):
    # 词云图方法
    text = open(file_name, 'r', encoding='utf-8').read()
    wc = wordcloud.WordCloud(background_color="white", width=600, height=400,
                             collocations=False,
                             max_font_size=60,
                             random_state=1,
                             max_words=500,
                             font_path='msyh.ttf')
    wc.generate(text)
    plt.axis("off")
    plt.figure(dpi=600)
    wc.to_file(savename)


def build_one():
    qinggan_cidian = readLines("./cidian/qingganci.txt")
    data = pd.read_excel("./datasets/{}".format(data_name))
    words = []
    vocal = []
    for one in data['content'].values:
        one_cut = [i for i in jieba.lcut(one) if i not in stopwords and i is not None]
        for x in one_cut:
            vocal.append(x)
            if x in qinggan_cidian:  # 获取全部的情感词数据,保存到qingganci.csv文件中
                words.append(x)
    print("情感词共计：{}个".format(len(words)))
    df = pd.DataFrame(vocal)
    df.to_csv('vocal.csv', index=None)
    df = pd.DataFrame(words)
    df.to_csv('qingganci.csv', index=None)


def build_50(times=20):
    # 词频大于50和20的情感词，times设置次数

    words = pd.read_csv('qingganci.csv').values.tolist()
    words = [i for j in words for i in j]
    rows_50 = {}
    for x, y in Counter(words).most_common(10000):
        if y > times:
            rows_50[x] = y
    data = pd.read_excel("./datasets/{}".format(data_name))
    words = []
    for one in data['content'].values:
        one_cut = [i for i in jieba.lcut(one.replace("\n", "").replace(" ", "").strip()) if
                   i not in stopwords and i is not None]
        for x in one_cut:
            if x in list(rows_50.keys()):
                words.append(x)

    df = pd.DataFrame(words)
    df.to_csv('qingganci_{}.csv'.format(times), index=None)


# 低频正负情感词的评论
def build_two(file_name):
    qinggan_cidian = readLines("{}.txt".format(file_name))  # pos.txt, neg.txt
    data = pd.read_excel("./datasets/{}".format(data_name))
    words = []
    vocal = []
    for one in data['content'].values:
        one_cut = [i for i in jieba.lcut(one) if i not in stopwords and i is not None]
        for x in one_cut:
            vocal.append(x)
            if x in qinggan_cidian:
                words.append(x)

    df = pd.DataFrame(words)
    df.to_csv('qingganci_{}.csv'.format(file_name), index=None)


def build_dipin(file_name):
    # 次数大于5次的情感词
    times = 5
    words = pd.read_csv('qingganci_{}.csv'.format(file_name)).values.tolist()
    words = [i for j in words for i in j]
    rows_50 = {}
    for x, y in Counter(words).most_common(10000):
        if y > times:
            rows_50[x] = y
    data = pd.read_excel("./datasets/{}".format(data_name))
    words = []
    for one in data['content'].values:
        one_cut = [i for i in jieba.lcut(one.replace("\n", "").replace(" ", "").strip()) if
                   i not in stopwords and i is not None]
        for x in one_cut:
            if x in list(rows_50.keys()):
                words.append(x)

    df = pd.DataFrame(words)
    df.to_csv('qingganci_{}_{}.csv'.format(file_name, times), index=None)


def sent2word(sentence):
    if isinstance(sentence, str):
        segList = jieba.lcut(sentence.strip().replace("\n", ""))
        newSent = [i for i in segList if i not in stopwords]
        return newSent
    else:
        return sentence


def get_wordDicts():
    words = {}
    data = pd.read_excel("./datasets/{}".format(data_name))
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
    with open("./cidian/BosonNLP_sentiment_score.txt", 'r', encoding='utf8') as f:
        senList = f.readlines()
    senDict = defaultdict()
    for s in senList:
        try:
            senDict[s.split(' ')[0]] = s.split(' ')[1].strip('\n')
        except Exception:
            pass
    # (2) 否定词
    with open("./cidian/notDict.txt", 'r', encoding='utf-8') as f:
        notList = f.read().splitlines()
    # (3) 程度副词
    with open("./cidian/degreeDict.txt", 'r', encoding='utf-8') as f:
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
    senWord, notWord, degreeWord = classifyWords(wordsDicts)


def get_scores(line, ):
    try:
        newSent = sent2word(line)
        scores = scoreSent(senWord, notWord, degreeWord, newSent)
        print(scores)
        return scores
    except Exception as e:
        return ""


def get_scores_meixi(line):
    if '梅西' in line:
        try:
            newSent = sent2word(line)
            scores = scoreSent(senWord, notWord, degreeWord, newSent)
            return scores
        except Exception as e:
            return ""
    else:
        return ""


def get_scores_cluo(line):
    if 'C罗' in line:
        try:
            newSent = sent2word(line)
            scores = scoreSent(senWord, notWord, degreeWord, newSent)
            return scores
        except Exception as e:
            return ""
    else:
        return ""


def build_scores_big():
    data = pd.read_excel("./datasets/{}".format(data_name))
    print(data.shape)
    data['sentiments'] = data['content'].apply(get_scores)
    data.to_csv("dataAll_scores.csv", index=None)
    print(data.shape)


def build_scores_meixi():
    data = pd.read_excel("./datasets/{}".format(data_name))
    data['sentiments'] = data['content'].apply(get_scores_meixi)
    print(data.shape)
    data = data[~data['sentiments'].isin([""])]
    data.to_csv("dataAll_scores_meixi.csv", index=None)
    print(data.shape)


def build_scores_cluo():
    data = pd.read_excel("./datasets/{}".format(data_name))
    print(data.shape)
    data['sentiments'] = data['content'].apply(get_scores_cluo)
    data = data[~data['sentiments'].isin([""])]
    data.to_csv("dataAll_scores_cluo.csv", index=None)
    print(data.shape)


def get_time(inputs):
    try:
        return inputs[5:12].replace("月", '-').replace('日', '')
    except:
        return ""


def get_class(inputs):
    if inputs > 0:
        return '积极'
    if inputs < 0:
        return "消极"


def plots(data_name):
    print("执行情感强度随时间变化图 方法")
    data = pd.read_csv(data_name)
    data['times'] = data['时间'].apply(get_time)
    data['class'] = data['sentiments'].apply(get_class)

    data = data[~data['times'].isin([""])]
    dates = []
    pos_data = []
    neg_data = []
    eos_data = []
    for x, y in data.groupby(by='times'):
        dates.append(x)
        pos_mean = y[y['class'].isin(['积极'])]['sentiments'].mean()
        pos_data.append(pos_mean)
        neg_mean = y[y['class'].isin(['消极'])]['sentiments'].mean()
        neg_data.append(neg_mean)
        eos_data.append(0)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, pos_data, c='red', label='积极')
    plt.plot(dates, neg_data, c='blue', label='消极')
    plt.plot(dates, eos_data, c='yellow', label='中性')
    plt.xlabel("时间")
    plt.ylabel("极性强度")
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.savefig('{}.png'.format(data_name.replace(".csv", "")))
    plt.show()


def get_zhuanfa(inputs):
    out = []
    for x in inputs:
        res = x.replace("转发", "").replace(" ", "")
        out.append(int(res))
    return sum(out)


def degree_analysis():
    save = []
    data = pd.read_excel("./datasets/dataBig.xlsx")
    for x, y in data.groupby(by='标题'):
        try:
            rudu = y['点赞数'].sum()
            chudu = get_zhuanfa(y['转发数'].values.tolist())
            rows = {"name": x, "入度": rudu, "出度": chudu}
            save.append(rows)
        except:
            continue

    df = pd.DataFrame(save)
    df.to_excel("出入度分析.xlsx", index=None)
    print(df.shape)


if __name__ == '__main__':

    method = 'degree_analysis'  # 修改这里，分别执行下面的代码

    if method == 'cipin':
        cipin()

    if method == 'build_one':
        build_one()

    if method == 'build_50':
        times = 20  # 50和20
        build_50(times=times)

    if method == 'plot_word_cloud_gaopin':
        times = 50  # 50和20
        plot_word_cloud("qingganci_{}.csv".format(times), "qingganci_{}.png".format(times))

    if method == 'build_two':
        build_two(file_name='neg')

    if method == 'build_dipin':
        build_dipin(file_name='neg')

    if method == 'plot_word_cloud_dipin':
        names = 'neg'
        times = 5
        plot_word_cloud("qingganci_{}_{}.csv".format(names, times), "qingganci_{}_{}.png".format(names, times))

    if method == 'build_scores':
        # 计算评论的情感得分
        build_scores_big()

    if method == "build_scores_meixi":
        build_scores_meixi()

    if method == 'build_scores_cluo':
        build_scores_cluo()

    if method == 'plots':
        datas = 'dataAll_scores.csv'  # 这里选择下面的不同的关键词，画图
        # dataAll_scores
        # dataAll_scores_cluo
        # dataAll_scores_meixi
        plots(data_name=datas)

    if method == 'degree_analysis':
        degree_analysis()
