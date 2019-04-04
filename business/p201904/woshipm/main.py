import pandas as pd
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import datetime
import jieba
import wordcloud
from scipy.misc import imread
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

jieba.load_userdict("userdict.txt")  # 加载分词词典

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# 读取停用词
st = []
with open("stopwords.txt", 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        st.append(line.replace("\n", ""))

"""
统计一共有多少种文章类型content_type，每种占比情况，可视化（饼图.环状..都行，您那个顺手用哪个，好看更好）
"""


def content_type_pie():
    data = pd.read_excel("./datasets/woshipm.xlsx")
    print(data.columns.tolist())
    content_type = data['content_type'].values.tolist()

    labels = []
    sizes = []
    for x, y in Counter(content_type).most_common():
        labels.append(x)
        sizes.append(y)
    plt.figure(dpi=300, figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=150)
    plt.title("content_type 饼图")
    plt.savefig("content_type_pie.png")

"""
作者发文量统计 author_name，作者对应的发文量，降序排列，目的是看哪些作者比较活跃，篇幅如果有限 取前50 名即可
"""


def author_name_count():
    data = pd.read_excel("./datasets/woshipm.xlsx")
    print(data.columns.tolist())
    author_name = data['author_name'].values.tolist()
    save = []
    for x, y in Counter(author_name).most_common(50):  # 取前50名
        rows = {"name": x, "times": y}
        save.append(rows)

    df = pd.DataFrame(save)
    # 保存前50名作者的名字和发文量到excel文件中。
    df.to_excel("author_name_count.xlsx", index=None)


def read_num_process(inputs):
    try:
        if '万' in inputs:
            return float(inputs.replace("万", "")) * 10000
        else:
            return inputs
    except:
        return inputs


def pianhaofenxi():
    data = pd.read_excel("./datasets/woshipm.xlsx")
    print(data.columns.tolist())
    # 按阅读量排序，并保存到文件read_num_highest.xlsx, 保存标题和阅读量
    data['read_um_pro'] = data['read_um'].apply(read_num_process)
    data_sort1 = data.sort_values(by='read_um_pro', ascending=False)
    read_um_pro = data_sort1['title'].head(100)
    df1 = pd.DataFrame()
    df1['read_num_highest'] = read_um_pro
    df1['read_um'] = data_sort1['read_um_pro'].head(100)
    df1.to_excel("read_num_highest.xlsx", index=None)

    # 按收藏量排序，collection_num_highest.xlsx, 保存标题和收藏量
    data_sort2 = data.sort_values(by='collection_num', ascending=False)
    collection_num = data_sort2['title'].head(100)
    df2 = pd.DataFrame()
    df2['collection_num_highest'] = collection_num
    df2['collection_num'] = data_sort2['collection_num'].head(100)
    df2.to_excel("collection_num_highest.xlsx", index=None)

    # 按点赞量排序，praise_num_highest.xlsx, 保存标题和点赞量
    data_sort3 = data.sort_values(by='praise_num', ascending=False)
    praise_num = data_sort3['title'].head(100)
    df3 = pd.DataFrame()
    df3['praise_num_highest'] = praise_num
    df3['praise_num'] = data_sort3['praise_num'].head(100)
    df3.to_excel("praise_num_highest.xlsx", index=None)

    # 按打赏量排序，reward_num_highest.xlsx, 保存标题和打赏量
    data_sort4 = data.sort_values(by='reward_num', ascending=False)
    reward_num = data_sort4['title'].head(100)
    df4 = pd.DataFrame()
    df4['reward_num_highest'] = reward_num
    df4['reward_num'] = data_sort4['reward_num'].head(100)
    df4.to_excel("reward_num_highest.xlsx", index=None)


def get_title_long(inputs):
    try:
        return len(inputs)
    except:
        return 0


def get_week(inputs):
    try:
        anyday = datetime.datetime.strptime(inputs, '%Y-%m-%d')
        return anyday.weekday()
    except:
        return inputs


"""
看 周几 发文量最高：这个发文量应该是需要 用代码计算一下 对应周一到周日每天的总发文量，然后降序排列
看 周几 阅读量最高：这个阅读量应该是需要 用代码计算一下 对应周一到周日每天的总阅读量，然后降序排列
"""


def week_sort():
    data = pd.read_excel("./datasets/woshipm.xlsx")
    data['read_um_pro'] = data['read_um'].apply(read_num_process)
    data['title_long'] = data['title'].apply(get_title_long)
    data['mainbody_long'] = data['main_body'].apply(get_title_long)
    data['time_day'] = data['time'].apply(get_week)
    data.to_excel('./datasets/woshipm_加3列.xlsx')
    df = []
    for x, y in data.groupby(by='time_day'):
        rows = OrderedDict()
        rows['week'] = x
        rows['fawen'] = y.shape[0]
        rows['read'] = y['read_um_pro'].sum()
        df.append(rows)
    pd.DataFrame(df).to_excel('week_sort.xlsx', index=None)


def hulianwang():
    data = pd.read_excel("./datasets/woshipm.xlsx")
    print(data.columns.tolist())

    ss = []
    with open('userdict.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            ss.append(line.replace("\n", ""))

    def cuts(inputs):
        try:
            out = []
            res = jieba.lcut(inputs)
            for i in res:
                if i in ss:
                    out.append(i)
            return " ".join(out)
        except:
            return ""

    data['mainbody_cut'] = data['main_body'].apply(cuts)

    save = []
    for x, y in Counter(data['mainbody_cut'].values.tolist()).most_common():
        rows = {"name": x, "times": y}
        print(rows)
        save.append(rows)

    df = pd.DataFrame(save)
    # 保存前50名作者的名字和发文量到excel文件中。
    df.to_excel("word_list_count.xlsx", index=None)


def cuts(inputs):
    try:
        res = " ".join([i for i in jieba.lcut(inputs) if i not in st])
        return res
    except:
        return ""


def data_cuts():
    data = pd.read_excel("./datasets/woshipm.xlsx", )
    data['mainbody_cut'] = data['main_body'].apply(cuts)
    df = data['mainbody_cut']
    df.to_excel("mainbody_cut.xlsx", index=None)
    print(df.shape)


def build_tfidf():
    data = pd.read_excel("mainbody_cut.xlsx")
    corpus = []
    limit = 100
    num = 0
    for one in data['mainbody_cut'].values:
        num += 1
        if num > limit:
            break
        corpus.append(one)

    topn = 500
    max_df = 0.3
    vectorizer = TfidfVectorizer(max_df=max_df)
    matrix = vectorizer.fit_transform(corpus)
    feature_dict = {v: k for k, v in vectorizer.vocabulary_.items()}
    top_n_matrix = np.argsort(-matrix.todense())[:, :topn]
    df = pd.DataFrame(np.vectorize(feature_dict.get)(top_n_matrix))
    df.to_csv("build_tfidf.csv", index=None, header=None, sep=' ')


def plot_word_cloud(file_name, savename, bk='bk.png'):
    fp = 'C:\\Windows\\Fonts\\msyh.ttf'
    text = open(file_name, 'r', encoding='utf-8').read()
    alice_coloring = imread(bk)
    wc = wordcloud.WordCloud(background_color="white", width=918, height=978,
                             max_font_size=50,
                             mask=alice_coloring,
                             random_state=1,
                             max_words=80,
                             mode='RGBA',
                             font_path=fp)
    wc.generate(text)
    image_colors = wordcloud.ImageColorGenerator(alice_coloring)

    plt.axis("off")
    plt.figure()
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.axis("off")
    plt.figure(dpi=300)
    plt.axis("off")
    wc.to_file(savename)


if __name__ == '__main__':
    method = ''
    if method == 'content_type_pie':
        content_type_pie()

    if method == 'author_name_count':
        author_name_count()

    if method == 'pianhaofenxi':
        # 偏好分析
        pianhaofenxi()

    if method == 'week_sort':
        # 计算长度， 按周排序
        week_sort()

    if method =='hulianwang':
        # 互联网公司出现次数排名
        hulianwang()
    if method == 'data_cuts':
        # 分词
        data_cuts()

    if method == 'build_tfidf':
        # tf idf 获取关键词  top 500
        build_tfidf()

    if method == 'plot_word_cloud':
        # 词云图
        plot_word_cloud(file_name='build_tfidf.csv', savename='词云.png')

