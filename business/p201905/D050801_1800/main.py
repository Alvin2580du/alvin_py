import pandas as pd
import jieba
import matplotlib.pyplot as plt
import numpy as np
import datetime
import wordcloud
import urllib.request
from bs4 import BeautifulSoup
from tqdm import tqdm
from collections import Counter
from scipy.misc import imread

# 画图 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
"""
①对excel中的数据进行基本的清洗，去燥，去重
②对标题title 和帖子内容 main_body 分词-jieba 精确分词
"""

######### 分词 ##############
jieba.load_userdict("userdict.txt")

data_raw = pd.read_excel("./data/汽车之家论坛.xlsx")

print(data_raw.shape)
data_raw = data_raw.drop_duplicates()
data_raw = data_raw[~data_raw['main_body'].isin([None, " "])]
data_raw = data_raw.fillna(0)


def readLines(filename):
    # 读文件的方法
    out = []
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            out.append(line.replace("\n", ""))
    return out


stopwords = readLines('stopwords.txt')


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def sent2word(sentence):
    # 分词方法，输入句子，输出list
    if isinstance(sentence, str):
        segList = jieba.lcut(sentence)
        newSent = []
        for i in segList:
            if not i.replace(" ", ""):
                continue
            if i in stopwords:
                continue
            if not is_chinese(i):
                continue
            newSent.append(i)
        if len(newSent) > 0:
            return " ".join(newSent)
        else:
            return " "
    else:
        return " "


def question_one():
    data_raw['title_cut'] = data_raw['title'].apply(sent2word)
    data_raw['main_body_cut'] = data_raw['main_body'].apply(sent2word)
    print(data_raw.shape)
    data = data_raw[~data_raw['main_body_cut'].isin([None, " "])]
    data.to_excel("汽车之家论坛-分词.xlsx", index=None)
    print(data.shape)
    content = data['main_body_cut']
    fw = open('汽车之家论坛_cut.txt', 'w', encoding='utf-8')
    for one in content:
        fw.writelines(" ".join(one.split()) + "\n")

    """
    (26895, 8)
    (26886, 8)
    (26886, 10)
    (25373, 10)
    """


"""
2、描述性数据分析
（1）发帖者发文量统计
发帖者发文量统计 author_name，每个发帖者对应的发文量，降序排列，目的是看哪些作者比较活跃， 取前50名。
"""


def question_21():
    rows = {}
    for x, y in data_raw.groupby(by='author_name'):
        rows[x] = y.shape[0]
    save = []
    res = sorted(rows.items(), key=lambda item: item[1], reverse=True)[:50]
    for x in res:
        rows = {'author_name': x[0], 'num': x[1]}
        save.append(rows)
    df = pd.DataFrame(save)
    df.to_excel("question_21.xlsx", index=None)


"""

（2）发帖者地区统计
作者地区分布统计author_place，目的看下维修保养版块人群使用情况。降序排列
ps:不用精确到 城市分区，只是一级名称就行，比如：北京、河北、天津、湖南…,不需要北京 海淀。

"""


def get_p(inputs):
    try:
        return inputs.split()[0]
    except:
        return inputs


def question_22():
    rows = {}
    data_raw['author_province'] = data_raw['author_place'].apply(get_p)
    for x, y in data_raw.groupby(by='author_province'):
        rows[x] = y.shape[0]

    save = []
    res = sorted(rows.items(), key=lambda item: item[1], reverse=True)
    for x in res:
        rows = {'author_name': x[0], 'num': x[1]}
        save.append(rows)
    df = pd.DataFrame(save)
    df.to_excel("question_22.xlsx", index=None)


"""
（3） 读者阅读偏好分析
通过分析读者阅读偏好分析-为网站运营方向提供建议：
①哪些帖子点击量click_num较高-这个可以反应大家关注的热点，保留前50即可
②哪些帖子回复量reback_num较高-这个可以反映大家对哪些帖子比较有共识，产生话题，保留前50即可
"""


def question_23():
    data_sort = data_raw.sort_values(by='click_num', ascending=False)
    data_sort.head(50).to_excel("click_num_top50.xlsx", index=None)

    data_sort1 = data_raw.sort_values(by='reback_num', ascending=False)
    data_sort1.head(50).to_excel("reback_num_top50.xlsx", index=None)


"""

（4）热门帖子分析
①在原始数据基础上 添加新列 
第一个 新列 名为“title_long” 统计 帖子标题字数
第二个 新列 名为“mainbody_long” 统计 帖子正文字数
※	做关联性分析：
帖子的点击量、回复量和帖子标题字数、帖子正文字数是否存在统计学意义上的相关性关系 -可视化-想看出 标题字数多少、正文字数多少时 点击量、回复量最高。

②发文时间规律分析
第三个 新列 名为“time_day” 给“time”列的日期 附上 对应的 周一到周日
第四个 新列 名为“time_moment” 给“time”列的几分几秒 附上 对应的1 -24小时
 """


def get_title_long(inputs):
    try:
        return len(inputs)
    except:
        return 0


def get_time_day(inputs):
    # 2019-3-18 14:42:15
    try:
        res = datetime.datetime.strptime(inputs, '%Y-%m-%d %H:%M:%S').weekday()
        return res
    except:
        return inputs


def get_time_moment(inputs):
    try:
        res = datetime.datetime.strptime(inputs, '%Y-%m-%d %H:%M:%S').hour
        return res
    except:
        return inputs


def question_24():
    data_raw['title_long'] = data_raw['title'].apply(get_title_long)
    data_raw['mainbody_long'] = data_raw['main_body'].apply(get_title_long)
    x1 = data_raw['title_long'].values
    x2 = data_raw['mainbody_long'].values
    y1 = data_raw['click_num'].values
    y2 = data_raw['reback_num'].values

    ab = np.array([x1, y1])
    print(np.corrcoef(ab))
    print("- * -" * 10)

    ab1 = np.array([x1, y2])
    print(np.corrcoef(ab1))
    print("- * -" * 10)

    ab2 = np.array([x2, y1])
    print(np.corrcoef(ab2))
    print("- * -" * 10)

    ab3 = np.array([x2, y2])
    print(np.corrcoef(ab3))
    print("- * -" * 10)

    data_raw['time_day'] = data_raw['time'].apply(get_time_day)
    data_raw['time_moment'] = data_raw['time'].apply(get_time_moment)
    data_raw.to_excel("time_moment+day.xlsx", index=None)


"""
4、文本挖掘分析
（1）关键词提取
做词云图
"""


def question_41():
    data_name = '汽车之家论坛'

    text = open('{}_cut.txt'.format(data_name), 'r', encoding='utf-8').read()
    alice_coloring = imread("bk.jpg")

    wc = wordcloud.WordCloud(background_color="white", width=800, height=600,
                             mask=alice_coloring,
                             max_font_size=20,
                             random_state=1,
                             max_words=100,
                             font_path='C:\\Windows\\msyh.ttf')

    wc.generate(text)
    image_colors = wordcloud.ImageColorGenerator(alice_coloring)

    plt.axis("off")
    plt.figure()
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.axis("off")
    plt.figure(dpi=300)
    plt.axis("off")
    wc.to_file('{}_word_cloud.png'.format(data_name))


"""
（2）各类车品牌提及次数
统计所有帖子标题title 文本+帖子正文main_body中提及各类车品牌的次数-可以分析出哪种车型在维修保养时被提及的最多。根据提及频次的大小降序。 

（3）车部位提及次数
统计帖子标题 title +帖子文本中对应的车部件进行提取 
-为了看出汽车哪个部位遇到的维修保养问题最多，降序排列。
"""


def get_carnames():
    fw = open('all_carnames.txt', 'w', encoding='utf-8')

    for i in tqdm(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                   'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']):

        try:
            url = 'https://www.autohome.com.cn/grade/carhtml/{}.html'.format(i)
            req = urllib.request.Request(url)
            req.add_header("User-Agent",
                           "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36")
            req.add_header("Accept", "*/*")
            req.add_header("Accept-Language", "zh-CN,zh;q=0.8")
            data = urllib.request.urlopen(req)
            html = data.read()

            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('h4')
            for x in resp:
                try:
                    fw.writelines(x.text + "\n")
                except:
                    continue
        except:
            continue


def question_423():
    data = pd.read_excel("汽车之家论坛-分词.xlsx")
    names = readLines('all_carnames.txt')
    peijian = readLines('配件.txt')
    print(len(names), len(peijian))

    names_find = []
    peijian_find = []

    for x, y in zip(data['main_body_cut'], data['title_cut']):
        res = [x, y]
        for i in res:
            for w in i.split():
                if w in names:
                    names_find.append(w)
                if w in peijian:
                    peijian_find.append(w)

    print(len(names_find), len(peijian_find))

    names_find_save = []
    peijian_find_save = []

    for name, f in Counter(names_find).most_common(100000):
        names_find_save.append({"名称": name, "数量": f})

    for pj, f in Counter(peijian_find).most_common(100000):
        peijian_find_save.append({"名称": pj, "数量": f})

    df = pd.DataFrame(names_find_save)
    df.to_excel("汽车名称-数量.xlsx", index=None)

    df1 = pd.DataFrame(peijian_find_save)
    df1.to_excel("汽车配件名称-数量.xlsx", index=None)


"""
（4）维修保养问题分别提及次数
再添加新列“type”, 识别出对应的哪个词 
如果是 维修类词汇 则 识别为 维修问题，
如果对应的是保养类词汇 则 识别为 保养问题 
---目的想看维修为题多还是保养遇到的问题多。
"""
weixiu = ['维修', '损', '损坏', '受损', '更换', '修复', '启动', '断电', '发动机', '引擎', '轮毂', '电压', '启动机', '刹车',
          '打不着', '坏', '异', '异响', '去除', '除去', '除', '不稳定', '锁死', '熄火', '拆除', '拆', '毁', '失灵', '拆卸',
          '卸', '断气', '补胎', '补', '打蜡', '撞', '短路', '断路', '烧', '烧毁', '磨损', '剐蹭', '刮坏', '刮', '缺',
          '故障', '不能', '不', '问题', '事故', '响', '不动', '补漆', '划痕', '划', '漆', '怠速', '抖动', '换', '减震',
          '保险杠', '修', '割', '困难', '引擎舱', '引擎', '刹车', '助力器', '刮伤', '挡风玻璃', '马达', '白金', '电池',
          '飞轮齿轮', '马达齿轮', '起动杆', '低压线路', '电容器', '高压线路', '油箱', '喷油嘴', '汽油帮浦', '化油器', '怠速油',
          '水箱', '风扇皮带', '水管', '冷却水', '水帮浦', '节温器', '动力辅助装置', '前轮', '煞车踏板', '拖曳车轮', '煞车',
          '油帮浦', '煞车器', '储油箱', '车胎', '排气管', '易熄火', '怠速', '窒油', '加速力', '点火时间', '化油器', '失常']

baoyang = ['养', '保养', '机油', '防盗', '轮胎', '保', '不好', '色差', '油耗', '用', '打蜡', '美容', '过审', '过年审',
           '换胎', '空气滤清器', '空调滤清器', '汽油滤清器', '前刹车片', '后刹车片', '雨刮片', '防冻液', '火花塞', '刹车油',
           '自动变速箱油', '蓄电池']


def is_weixiu(inputs):
    for i in inputs:
        if i in weixiu:
            return '1'
    return '0'


def is_baoyang(inputs):
    for i in inputs:
        if i in baoyang:
            return '1'
    return '0'


def question_44():
    data = pd.read_excel("汽车之家论坛-分词.xlsx")
    data['维修'] = data['main_body_cut'].apply(is_weixiu)
    data['保养'] = data['main_body_cut'].apply(is_baoyang)
    weixiu_data = data[data['维修'].isin(['1'])]
    baoyang_data = data[data['保养'].isin(['1'])]
    rows = {"维修": weixiu_data.shape[0], '保养': baoyang_data.shape[0]}
    print(rows)
    df = pd.DataFrame([rows])
    df.to_excel("汽车之家论坛-维修-保养.xlsx", index=None)


""" 基础的分析
（5） 文本聚类分析
（6）帖子情感分析
（7）LDA主体模型建立
"""

if __name__ == "__main__":

    method = 'snownlp'

    if method == 'question_one':
        question_one()

    if method == 'question_21':
        question_21()

    if method == 'question_22':
        question_22()

    if method == 'question_23':
        question_23()

    if method == 'question_24':
        question_24()

    if method == 'question_41':
        question_41()

    if method == 'question_423':
        question_423()

    if method == 'question_44':
        question_44()

    if method == 'LdaModel':
        # LDA 主题模型，输出文件ldaOutput.txt 表示输出的文档主题。
        from gensim.corpora import Dictionary
        from gensim.models import LdaModel
        from gensim import models

        data = pd.read_excel("汽车之家论坛-分词.xlsx")
        # LDA 主题模型
        # 构建训练语料
        Listdata = data['main_body_cut'].values.tolist()
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


        data = pd.read_excel("汽车之家论坛-分词.xlsx")
        save = []
        num = 0
        for one in data['main_body_cut']:
            num += 1
            senti = get_sentiment_cn(one)
            save.append(senti)
            if num % 1000 == 1:
                print(num)

        data['senti'] = save
        data.to_excel("汽车之家论坛_senti.xlsx", index=None)

    if method == 'cluster':
        # 聚类分析
        from gensim.models.doc2vec import Doc2Vec
        from sklearn.cluster import KMeans
        import gensim

        # 利用gensim将doc转换为向量
        x_train = []
        train_data = pd.read_excel("汽车之家论坛-分词.xlsx")
        print(train_data.shape)
        poster_id = train_data['author_name']
        TaggededDocument = gensim.models.doc2vec.TaggedDocument
        limit = 80000
        for i, y in enumerate(train_data['main_body_cut'].values):
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
        n_cluster = 20

        print("============= Kmeans 聚类 =============")
        kmean_model = KMeans(n_clusters=n_cluster, init='k-means++', n_init=10,
                             max_iter=1000, tol=1e-4, precompute_distances='auto',
                             verbose=0, random_state=None, copy_x=True,
                             n_jobs=None, algorithm='auto')
        kmean_model.fit(infered_vectors_list)
        cluster_label = kmean_model.labels_

        df = pd.DataFrame()
        df['user'] = poster_id
        df['label'] = cluster_label
        df.to_excel("聚类结果.xlsx", index=None)
        print(df.shape)
