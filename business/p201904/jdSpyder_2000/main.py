import pandas as pd
import os
import snownlp
import matplotlib.pyplot as plt
import jieba
from collections import Counter
import wordcloud


def get_content():
    save = []
    for file in os.listdir('./data'):
        file_name = os.path.join('./data', file)
        num = 0
        with open(file_name, 'r', encoding='gbk') as fr:
            lines = fr.readlines()
            for line in lines:
                num += 1
                if num == 1:
                    continue
                line_sp = line.split(",")
                if len(line_sp) == 13:
                    content = line_sp[4]
                    rows = {'content': content}
                    save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("content.xlsx", index=None)
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
    segList = jieba.lcut(sentence)
    newSent = []
    for i in segList:
        if not i.replace(" ", ""):
            continue
        if i in stopwords:
            continue
        newSent.append(i)
    if len(newSent) > 0:
        return " ".join(newSent)
    else:
        return ''


def wordFreq():
    data = pd.read_excel("content.xlsx")
    data['content_cut'] = data['content'].apply(sent2word)
    data['content_cut'].to_csv('content.csv', index=None)

    content_list = data['content_cut']
    words = []
    for one in content_list:
        for i in one.split():
            words.append(i)
    save = []
    df = pd.DataFrame(words)
    df.to_csv("words_for_wordcloud.csv", index=None, header=None)

    for x, y in Counter(words).most_common(500):
        rows = {'wrod': x, 'freq': y}
        save.append(rows)
        print(rows)
    df_freq = pd.DataFrame(save)
    df_freq.to_excel("词频统计.xlsx", index=None)
    print(df_freq.shape)


def plot_word_cloud(file_name, savename):
    text = open(file_name, 'r', encoding='utf-8').read()

    wc = wordcloud.WordCloud(width=400, height=400, background_color="white",
                             max_font_size=40,
                             random_state=1,
                             max_words=50,
                             font_path='C:\\Windows\\MSYH.TTF', repeat=True)
    wc.generate(text)
    plt.axis("off")
    plt.figure()
    wc.to_file(savename)


def get_sentiment_cn(text):
    s = snownlp.SnowNLP(text)
    res = s.sentiments
    if res > 0.3:
        return "积极"
    else:
        return "消极"


def sentiment_analysis():
    data = pd.read_excel("contentCut.xlsx")
    data['senti'] = data['comment_cut'].apply(get_sentiment_cn)
    data.to_excel("contentCut_senti.xlsx", index=None)


def comments_cut():
    data = pd.read_excel("content.xlsx", header=None)
    print(data.columns)
    data['comment_cut'] = data[0].apply(sent2word)
    data.to_excel("contentCut.xlsx", index=None)
    print(data.shape)


if __name__ == "__main__":

    method = 'plot_word_cloud'

    if method == 'sentiment_analysis':
        sentiment_analysis()

    if method == 'wordFreq':
        wordFreq()

    if method == 'plot_word_cloud':
        plot_word_cloud(file_name='words_for_wordcloud.csv', savename='iphone.png')