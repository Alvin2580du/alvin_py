import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib import error
import urllib.parse
import pandas as pd
import logging
import urllib
import os
from collections import OrderedDict
from tqdm import trange
import jieba
import matplotlib.pyplot as plt
import wordcloud
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import preprocessing
import gensim

jieba.load_userdict("userdict.txt")

if not os.path.exists('./data'):
    os.makedirs('./data')
def get_rost_data():
    da = pd.read_excel("datasource.xlsx")
    da['comment'].to_csv("rost_data.txt", index=None)


def isurl(url):
    if requests.get(url).status_code == 200:
        return True
    else:
        return False


def urlhelper(url):
    user_agents = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", user_agents)
        req.add_header("Accept", "*/*")
        req.add_header("Accept-Language", "zh-CN,zh;q=0.9")
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')
        return html
    except error.URLError as e:
        logging.warning("{}".format(e))


def spyder_muke():
    for pg in trange(1, 8):
        urls = "https://www.imooc.com/course/list?page={}".format(pg)
        if isurl(urls):
            html = urlhelper(urls)
            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('a', attrs={"class": "course-card"})
            for one in resp:
                try:
                    href = one['href'].split("/")[-1]
                    save = []
                    title_links = "https://www.imooc.com/coursescore/{}".format(href)
                    title_html = urlhelper(title_links)
                    title_soup = BeautifulSoup(title_html, "lxml")
                    title = title_soup.findAll('div', attrs={"class": "hd clearfix"})[0].text.strip().replace("\n", "")
                    print(title)
                    for page in range(1, 305):
                        try:
                            course_links = "https://www.imooc.com/course/coursescore/id/{}?page={}".format(href, page)
                            course_html = urlhelper(course_links)
                            course_soup = BeautifulSoup(course_html, "lxml")
                            content = course_soup.findAll('div', attrs={"class": "content-box"})
                            if not content:
                                continue
                            for one in content:
                                try:
                                    text = one.findAll('p', attrs={"class": "content"})[0].text
                                    star = one.find_all('span')[0].text
                                    time_re = one.findAll('span', attrs={"class": "time r"})[0].text.replace("时间：", "")
                                    username = one.findAll('a', attrs={"class": "username"})[0].text
                                    rows = OrderedDict({
                                        'username': username,
                                        'time': time_re,
                                        "comment": text,
                                        'star': star.replace("分", ""),
                                    })
                                    save.append(rows)
                                except:
                                    continue
                        except:
                            continue
                    df = pd.DataFrame(save)
                    df.to_excel("./data/{}.xlsx".format(title), index=None)
                    print(df.shape)
                except Exception as e:
                    print(e)
                    continue


def get_path():
    save = []
    for pg in trange(1, 10):
        urls = "https://www.imooc.com/course/list?page={}".format(pg)
        if isurl(urls):
            html = urlhelper(urls)
            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('a', attrs={"class": "course-card"})
            for one in resp:
                try:
                    href = one['href'].split("/")[-1]
                    title_links = "https://www.imooc.com/coursescore/{}".format(href)
                    title_html = urlhelper(title_links)
                    title_soup = BeautifulSoup(title_html, "lxml")
                    title_path = title_soup.findAll('div', attrs={"class": "path"})[0].text.replace("\n", "")
                    save.append(title_path)
                except:
                    continue

    df = pd.DataFrame(save)
    df.to_excel("title_path.xlsx", index=None)
    print(df.shape)


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
    try:
        segList = jieba.lcut(sentence.replace("\n", "").strip())
        newSent = []
        for i in segList:
            if not i.replace(" ", ""):
                continue
            if i in stopwords:
                continue
            newSent.append(i)

        if len(sentence) > 30:
            return " ".join(newSent)
        elif len(list(set(newSent))) < len(newSent) - 3:
            return '机械压缩句子，忽略'

        elif len(newSent) > 0:
            return " ".join(newSent)
        else:
            return None
    except:
        return None


# 日期处理函数
def transDate_liu(inputs):
    try:
        return str(inputs).split()[0]
    except:
        return None


def get_senti(inputs):
    if int(inputs) < 6:
        return "neg"
    elif int(inputs) == 6:
        return 'neu'
    else:
        return "pos"


def get_classPath(inputs):
    data_path = pd.read_excel("title_path.xlsx")
    res = {}
    for x, y in data_path.iterrows():
        paths = y[0].values.split("/")
        res[paths[-1]] = paths[1]
    return res[inputs]


def contact_dfs():
    data_path = pd.read_excel("title_path.xlsx")
    class_label = {}
    class_name = {}

    for x, y in data_path.iterrows():
        paths = y[0].split("\\")
        class_label[paths[-1]] = paths[1]
        class_name[paths[-1]] = paths[2]

    dfs = []
    for file in os.listdir('./data'):
        try:
            file_name = os.path.join('./data', file)
            data = pd.read_excel(file_name)
            data['senti'] = data['star'].apply(get_senti)
            data['course_label'] = class_label[file.replace(".xlsx", "")]
            data['course_name'] = class_name[file.replace(".xlsx", "")]
            data['course'] = file.replace('.xlsx', "")
            data['comment_cut'] = data['comment'].apply(sent2word)
            dfs.append(data)
        except:
            continue
    df = pd.concat(dfs)

    df = df[~df['comment_cut'].isin(['机械压缩句子，忽略'])]
    print(df.shape)
    df = df[~df['comment_cut'].isin([None])]
    print(df.shape)
    df.to_excel("datasource.xlsx", index=None)
    df['comment_cut'].to_csv("comment_cut.csv", index=None, header=None, encoding='utf-8')


# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算词向量
def get_train_vecs(x_train, x_test):
    n_dim = 200
    # 初始化模型和词表
    imdb_w2v = Word2Vec(size=n_dim, window=5, min_count=10, workers=12)
    imdb_w2v.build_vocab(x_train)

    imdb_w2v.train(x_train, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])
    train_vecs = preprocessing.scale(train_vecs)

    np.save('train_vecs.npy', train_vecs)
    # 在测试集上训练
    imdb_w2v.train(x_test, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
    imdb_w2v.save('w2v_model.pkl')
    # Build test tweet vectors then scale
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    test_vecs = preprocessing.scale(test_vecs)
    np.save('test_vecs.npy', test_vecs)
    return train_vecs, test_vecs


def get_chinese(inputs):
    if isinstance(inputs, str):
        return '1'
    else:
        return '0'


if __name__ == "__main__":

    method = 'get_path'

    if method == 'spyder_muke':
        # 爬虫代码
        spyder_muke()

    if method == 'get_path':
        # 抓取每门课程的类别
        get_path()

    if method == 'contact_dfs':
        # 合并每门课程的评论为一个文件夹。
        contact_dfs()

    if method == 'wordcloud':
        # 画词云图
        file_name = 'comment_cut.csv'
        savename = 'muke.png'

        text = open(file_name, 'r', encoding='utf-8').read()

        wc = wordcloud.WordCloud(width=400, height=400, background_color="white",
                                 max_font_size=40,
                                 random_state=1,
                                 max_words=50,
                                 font_path='C:\\Windows\\msyh.ttf', repeat=True)
        wc.generate(text)
        plt.axis("off")
        plt.figure()
        wc.to_file(savename)

    if method == 'train_model':
        # 训练模型

        data = pd.read_excel("datasource.xlsx")
        print(data.shape)
        data['type'] = data['comment_cut'].apply(get_chinese)
        data = data[data['type'].isin(['1'])]
        print(data.shape)
        data = shuffle(data)

        #  载入数据，做预处理(分词)，切分训练集与测试集
        neg = data[data['senti'].isin(['neg'])]
        pos = data[data['senti'].isin(['pos'])]
        neu = data[data['senti'].isin(['neu'])]
        print(len(neg), len(neu), len(pos))
        print("- " * 20)
        y = np.concatenate((np.ones(len(pos)), np.zeros(len(neu)), -np.ones(len(neg))), axis=0)

        x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['comment_cut'], neu['comment_cut'],
                                                                            neg['comment_cut'])),
                                                            y,
                                                            test_size=0.3)
        train_vecs, test_vecs = get_train_vecs(x_train, x_test)

        model = 'SVC'
        print("- * -" * 8, "{}".format(model), '- * -' * 8)
        clf = SVC(C=0.8, kernel='rbf', verbose=True, shrinking=False,
                  max_iter=29999, gamma=0.5, class_weight={-1: 1, 0: 1, 1: 1000})

        clf.fit(train_vecs, y_train)
        joblib.dump(clf, '{}_model.pkl'.format(model))
        print("{} 准确率：{:0.4f}".format(model, clf.score(test_vecs, y_test)))
        y_pred = clf.predict(test_vecs)
        print("- * -" * 20)

    if method == 'predicts':
        # 预测情感极性
        imdb_w2v = gensim.models.word2vec.Word2Vec.load('w2v_model.pkl')
        clf = joblib.load('SVC_model.pkl')

        # 对单个句子进行情感判断
        def svm_predict(string):
            n_dim = 200
            words = jieba.lcut(string)
            words_vecs = build_sentence_vector(words, n_dim, imdb_w2v)
            result = clf.predict(words_vecs)
            if int(result[0]) == 1:
                return "pos"
            elif int(result[0]) == 0:
                return 'neu'
            else:
                return "neg"


        data = pd.read_excel("datasource.xlsx")
        data['type'] = data['comment_cut'].apply(get_chinese)
        data = data[data['type'].isin(['1'])]
        data['predicts'] = data['comment_cut'].apply(svm_predict)
        data.to_excel("datasourcePredicts.xlsx", index=None)

    if method == 'LdaModel':
        # LDA 主题模型，输出文件ldaOutput.txt 表示输出的文档主题。
        from gensim.corpora import Dictionary
        from gensim.models import LdaModel
        from gensim import models

        data = pd.read_excel("datasource.xlsx")
        # LDA 主题模型
        # 构建训练语料
        Listdata = data['comment_cut'].values.tolist()
        train_set = [listi.replace("\n", "").strip().split(' ') for listi in Listdata if isinstance(listi, str)]
        print(len(train_set))
        dictionary = Dictionary(train_set)
        corpus = [dictionary.doc2bow(text) for text in train_set]  # 构建稀疏向量
        tfidf = models.TfidfModel(corpus)  # 统计tfidf
        corpus_tfidf = tfidf[corpus]  # 得到每个文本的tfidf向量，稀疏矩阵
        lda = LdaModel(corpus_tfidf, id2word=dictionary, num_topics=50, iterations=100)
        test = lda.print_topics(20)
        fw = open('ldaOutput.txt', 'w', encoding='utf-8')
        for i in test:
            print("{} {}\n".format(i[0], i[1]))
            fw.writelines("{} {}\n".format(i[0], i[1]))

    if method == 'snownlp':
        # 基于snownlp包的情感分析
        import snownlp

        def get_sentiment_cn(text):
            try:
                s = snownlp.SnowNLP(text).sentiments
                if s == 0.5:
                    return '中性'
                elif s > 0.5:
                    return "积极"
                else:
                    return "消极"
            except:
                return text


        data = pd.read_excel("datasource.xlsx")
        data['senti'] = data['comment_cut'].apply(get_sentiment_cn)
        data.to_excel("datasource_SnowNlp_senti.xlsx", index=None)

    if method == 'plot_hist':
        # 统计每种情感的数量画条形图

        # 画图 显示中文
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        data = pd.read_excel("datasource_SnowNlp_senti.xlsx")
        names = []
        values = []
        for x, y in data.groupby(by='senti'):
            names.append(x)
            values.append(y.shape[0])

        plt.figure()
        plt.bar(names, values)
        plt.savefig("snownlp_hist.png")

        plt.show()
