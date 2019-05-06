import pandas as pd
import snownlp
import matplotlib.pyplot as plt
import jieba
from collections import Counter
import wordcloud
from sklearn.cluster import KMeans
import os
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models.doc2vec import Doc2Vec

# 新建一个计算结果保存的文件夹
if not os.path.exists("./results"):
    os.makedirs('./results')

# 画图 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

######### 分词 ##############
jieba.load_userdict("userdict.txt")


def readLines(filename):
    # 读文件的方法
    out = []
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            out.append(line.replace("\n", ""))
    return out


stopwords = readLines('stopwords.txt')


def sent2word(sentence):
    # 分词方法，输入句子，输出list
    segList = jieba.lcut(sentence)
    newSent = []
    for i in segList:
        if not i.replace(" ", ""):
            continue
        if i in stopwords:
            continue
        newSent.append(i)
    if len(newSent) > 0:
        return newSent
    else:
        return None


# 日期处理函数
def transDate_zhai(inputs):
    date = str(inputs).replace("\n", "").replace("月", "-").replace('日', '').replace("今天", '04-26')
    if len(date) == 5:
        return date
    else:
        return None


def transDate_liu(inputs):
    try:
        return str(inputs).split()[0]
    except:
        return None


def cut_content(data_name):
    # 数据预处理，清洗数据
    data = pd.read_excel("./results/{}".format(data_name))
    print(data.shape)
    save = []
    num = 0
    not_num = 0
    for x, y in data.iterrows():
        try:
            comment = y['comment']
            if "翟天临" in data_name:
                date = transDate_zhai(y['date'])
            else:
                date = transDate_liu(y['date'])

            praise = y['praise']
            if str(praise).isdigit():
                comment_cut = sent2word(comment)
                if comment_cut:
                    if date:
                        rows = {'comment_cut': " ".join(comment_cut), 'date': date, 'praise': praise,
                                'comment': comment}
                        save.append(rows)
                        num += 1
                    else:
                        not_num += 1
                else:
                    not_num += 1
            else:
                not_num += 1
        except Exception as e:
            not_num += 1
            continue
    print(num)
    print(not_num)

    df = pd.DataFrame(save)
    # 分词后的文件
    df.to_excel("./results/{}_Cut.xlsx".format(data_name.replace(".xlsx", "")), index=None)
    print(df.shape)


def wordFreq(data_name):
    # 统计词频
    data = pd.read_excel("./results/{}_Cut.xlsx".format(data_name))
    content = data['comment_cut'].values.tolist()
    content_list = []
    fw = open('./results/{}_cut.txt'.format(data_name), 'w', encoding='utf-8')

    for i in content:
        if isinstance(i, str):
            for j in i.split():
                if len(j) == 1:
                    continue
                content_list.append(j)
                fw.writelines(j + " ")

    save = []

    for x, y in Counter(content_list).most_common(500):
        rows = {'wrod': x, 'freq': y}
        save.append(rows)
    df_freq = pd.DataFrame(save)
    # 词频结果文件
    df_freq.to_excel("./results/{}_词频统计.xlsx".format(data_name), index=None)
    print(df_freq.shape)


def plot_word_cloud(data_name):
    text = open('./results/{}_cut.txt'.format(data_name), 'r', encoding='utf-8').read()

    wc = wordcloud.WordCloud(width=400, height=400, background_color="white",
                             max_font_size=40,
                             random_state=1,
                             max_words=50,
                             font_path='C:\\Windows\\msyh.ttf', repeat=True)
    wc.generate(text)
    plt.axis("off")
    plt.figure()
    wc.to_file('./results/{}_word_cloud.png'.format(data_name))


def cluster(data_name):
    x_train = []
    train_data = pd.read_excel("./results/{}_Cut.xlsx".format(data_name))
    print(train_data.shape)
    poster_date = train_data['date']
    comment = train_data['comment']

    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    limit = 80000
    for i, y in enumerate(train_data['comment_cut'].values):
        # 遍历每一条评论
        word_list = y.split()
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
        if i > limit:
            break

    print("document length : {}".format(len(x_train)))
    # 训练词向量，大小是100维， 这里维度可以调整
    model_dm = Doc2Vec(x_train, min_count=1, window=3, vector_size=100, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=100)
    model_dm.save('model_dm.model')  # 保存模型

    infered_vectors_list = []
    i = 0
    for text, label in x_train:
        vector = model_dm.infer_vector(text)  # 计算指定句子的向量
        infered_vectors_list.append(vector)
        i += 1

    print("infered_vectors_list length :{}".format(len(infered_vectors_list)))
    n_cluster = 10

    print("============= Kmeans 硬聚类 =============")
    kmean_model = KMeans(n_clusters=n_cluster, init='k-means++', n_init=10,
                         max_iter=1000, tol=1e-4, precompute_distances='auto',
                         verbose=0, random_state=None, copy_x=True,
                         n_jobs=None, algorithm='auto')
    kmean_model.fit(infered_vectors_list)
    cluster_label = kmean_model.labels_

    df = pd.DataFrame()
    df['user'] = poster_date
    df['comment'] = comment
    df['label'] = cluster_label
    df.to_excel("./results/{}_硬聚类结果.xlsx".format(data_name), index=None)


def get_sentiment_cn(text):
    s = snownlp.SnowNLP(text)
    return s.sentiments


def sentiment_analysis(data_name):
    data = pd.read_excel("./results/{}_Cut.xlsx".format(data_name))
    data['senti'] = data['comment_cut'].apply(get_sentiment_cn)
    data.to_excel("./results/{}_senti.xlsx".format(data_name), index=None)


def getNew_y1(y1):
    max_ = max(y1)
    min_ = min(y1)
    res = []
    for i in y1:
        res.append((i - min_) / (max_ - min_))
    return res


def time_scores(data_name):
    data = pd.read_excel("./results/{}_senti.xlsx".format(data_name))
    rows = {}
    rows2 = {}
    for x, y in data.groupby(by='date'):
        score_mean = y['senti'].mean()
        praise_mean = y['praise'].mean()
        rows[x] = score_mean
        rows2[x] = praise_mean

    names = list(rows.keys())
    x = range(len(names))
    y = list(rows.values())
    y1 = getNew_y1(list(rows2.values()))

    plt.figure(figsize=(20, 10))
    plt.plot(x, y, marker='o', mec='r', mfc='w', label=u'得分')
    plt.plot(x, y1, marker='*', ms=10, label=u'点赞')

    plt.legend()  # 让图例生效
    plt.xticks(x, names, rotation=35)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"日期")  # X轴标签
    plt.ylabel("情感值")  # Y轴标签
    plt.title("{}情感随时间变化图".format(data_name))  # 标题
    plt.savefig("./results/{}_scores.png".format(data_name))
    plt.show()


def plot_hist(data_name):
    data = pd.read_excel("./results/{}_Cut.xlsx".format(data_name))

    rows = {}
    for x, y in data.groupby(by='date'):
        rows[x] = y.shape[0]

    value = list(rows.values())
    names = list(rows.keys())
    plt.figure()
    plt.plot(names, value)
    plt.title("{}评论数随时间变化折线图".format(data_name))
    plt.xticks(rotation=35)
    plt.savefig("./results/{}-评论数.png".format(data_name))


def plot_cluster(data_name):
    import random
    data = pd.read_excel("./results/{}_硬聚类结果.xlsx".format(data_name))
    rows = {}

    for x, y in data.groupby(by='label'):
        names = random.choice(y['comment'].values.tolist())
        rows[names] = "{:0.2f}".format(y.shape[0]/data.shape[0])

    value = list(rows.values())
    names = list(rows.keys())
    plt.figure(figsize=(40, 10))
    plt.barh(names, value)
    plt.title("{}-意见领袖".format(data_name))
    plt.xticks(rotation=15)
    plt.savefig("./results/{}-意见领袖.png".format(data_name))


if __name__ == '__main__':

    method = 'plot_cluster'

    data = '翟天临事件'  # 刘强东事件，奔驰维权事件， 翟天临事件

    if method == 'cut_content':
        # 分词函数，输入是原始文件，输出是分词后的文件
        cut_content(data_name='{}.xlsx'.format(data))

    if method == 'wordFreq':
        # 输入是分词后的文件，选择分词那一列数据，统计词频
        wordFreq(data_name='{}'.format(data))

    if method == 'plot_word_cloud':
        # 输入是分词过程中保存的只有评论的文件，是一个txt文件
        plot_word_cloud(data_name='{}'.format(data))

    if method == 'sentiment_analysis':
        # 输入是分词后的文件，输出是情感分析结果，对所有评论计算情感得分
        sentiment_analysis(data_name='{}'.format(data))

    if method == 'time_scores':
        # 输入是情感分析的结果文件，输出是情感得分的均值随时间变化的折线图
        time_scores(data_name='{}'.format(data))

    if method == 'plot_hist':
        plot_hist(data_name='{}'.format(data))

    if method == 'cluster':
        # 输入是分词后的文件，输出聚类结果，对所有的评论进行聚类
        cluster(data_name='{}'.format(data))

    if method == 'plot_cluster':
        # 主要意见占比条形图
        plot_cluster(data_name='{}'.format(data))
