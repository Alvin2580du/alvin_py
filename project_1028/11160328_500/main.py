import jieba
import pandas as pd
from collections import Counter
import jieba.analyse
from snownlp import SnowNLP
import re
import os

jieba.load_userdict("userdict.txt")
jieba.analyse.set_stop_words("mystopwords.txt")
stopwords = pd.read_csv("mystopwords.txt").values.tolist()  # 读取停用词
stopwords = [i for j in stopwords for i in j]


def makeData():
    save = []
    # d2013.txt 是直接从网页上复制出来的文本文件
    # 这个函数是把段落和文本对应起来， 存到excel里面
    data_path = './data'  # 这里是你每个年份的数据的目录， 比如 d2015.txt
    for file in os.listdir(data_path):
        file_name = os.path.join(data_path, file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            para = 0
            for line in lines:
                if len(line) > 1:
                    rows = {}
                    rows['段落'] = para
                    rows['文本'] = line
                    save.append(rows)
                    para += 1
        df = pd.DataFrame(save)
        save_path = './df_path'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_excel("./df_path/df_{}.xlsx".format(file.replace(".txt", "").replace('d', '')), index=None)
        print(df.shape)


keywords = ['国企改革', '国企', '国有企业', '国有企业改革', '经济制度', '经济']


def build_1():
    data_path = './df_path'  # 这里是你每个年份的数据的目录
    out = []
    keywords = ['国企改革', '国企', '国有企业', '国有企业改革', '经济制度', '经济']

    for file in os.listdir(data_path):
        file_name = os.path.join(data_path, file)
        data = pd.read_excel(file_name)
        all_words = []
        for x, y in data.iterrows():
            message = y['文本']
            message_cut = jieba.lcut(message)  # 分词
            for w in message_cut:
                # 把分词的结果全部保存到列表里面
                all_words.append(w)

        w_freq_tmp = {}
        # 统计词频
        fw = open('词频统计.txt', 'w', encoding='utf-8')
        for w, freq in Counter(all_words).most_common(10000000000):
            if w in keywords:
                w_freq_tmp[w] = freq
            if isinstance(w, str):
                if w.isalpha():
                    if w not in stopwords:  # 排除在stopwords里面的
                        res = "{},{}".format(w, freq)
                        # 写到文件
                        fw.writelines(res + "\n")

        fw.close()

        print("w_freq_tmp:  {}".format(w_freq_tmp))
        # 统计关键词的词频
        for x1, y1 in data.iterrows():
            para = y1['段落']
            message = y1['文本']
            message_cut = jieba.lcut(message)
            for w in message_cut:
                if w in keywords:  # 只筛选关键词
                    rows = {}
                    rows['关键词'] = w
                    rows['频率'] = w_freq_tmp[w]  # 用上面的词频字典，获取关键词的词频
                    rows['段落'] = para
                    rows['段落字数'] = len(message)
                    rows['年份'] = file.replace(".xlsx", "").split("_")[1]
                    out.append(rows)

    df = pd.DataFrame(out)
    df = df.drop_duplicates()  # 最后保存到excel里面
    df.to_excel("关键词统计.xlsx", index=None)

makeData()
print('====================================继续运行=================================')
build_1()
exit(1)


def isIncludekew(kw, cutsen):
    # 判断一个句子是不是包含关键字，并返回包含的关键字个数
    out = 0
    for i in kw:
        if i in cutsen:
            out += 1
    if out == 0:
        return False
    else:
        return out


def build():
    data = pd.read_excel("df_2013.xlsx", )
    fw = open('结果.txt', 'w', encoding='utf-8')
    k = 0
    num_sents = 3
    for x1, y1 in data.iterrows():  # 遍历所有的行
        k += 1
        para = y1['段落']
        message = y1['文本'].replace(" ", "").replace("\n", "")
        message_split = message.split('。')  # 用句号分割文本，
        if len(message_split) > num_sents:
            length = len(message_split)
            for i in range(length - num_sents):
                six_sen = message_split[i:i + num_sents]  # 根据指定的长度切割句子，
                six_sen_cut = jieba.lcut("。".join(six_sen))  # 分词
                times = isIncludekew(keywords, six_sen_cut)  # 判断组合里面是否包含关键字
                if times:
                    k = 1
                    while k <= times:  # 如果包含多个关键字，重复判断出现的次数遍
                        scores = SnowNLP("。".join(six_sen)).sentiments
                        res = "{},{:0.3f},{}".format(para, scores, "。".join(six_sen))
                        print(res)
                        fw.writelines(res + "\n")
                        k += 1
        else:  # 如果长度小于3句或者 6句，直接进行判断是不是包含关键字
            six_sen_cut = jieba.lcut("。".join(message_split))
            if isIncludekew(keywords, six_sen_cut):  # 如果包含关键字，计算情感极性
                if len(message_split) > 0:
                    scores = SnowNLP(message).sentiments
                    res = "{},{:0.3f},{}".format(para, scores, message)
                    print(res)
                    fw.writelines(res + "\n")


def make11():
    data = pd.read_excel("df_2013.xlsx", )
    z1 = re.compile('[\x80-\xff]{2}')
    z2 = re.compile('[\x80-\xff]{4}')
    z3 = re.compile('[\x80-\xff]{6}')
    z4 = re.compile('[\x80-\xff]{8}')
    k = 2
    for z in [z1, z2, z3, z4]:
        wfile = open('./data/result_{}words.txt'.format(k), 'w')
        k += 2
        dict = {}
        for x1, y1 in data.iterrows():  # 遍历所有的行
            message = y1['文本'].replace(" ", "").replace("\n", "")
            m = re.compile('[\x80-\xff]+').findall(message)
            for i in m:
                x = i.encode('gb18030')
                i = z.findall(x)
                for j in i:
                    if j in dict:
                        dict[j] += 1
                    else:
                        dict[j] = 1
        dict = sorted(dict.items(), key=lambda d: d[1])
        for a, b in dict:
            if b > 0:
                wfile.write(a + ',' + str(b) + '\n')
            print(a, b)


make11()
