import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec
from sklearn.cluster import KMeans
import jieba
import numpy as np
from sklearn.mixture import GaussianMixture

# 已完成


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
    newSent = [i for i in segList if i not in stopwords]
    return newSent


def cut_content():
    # 对发的微博进行分词，并保存到文件
    postfile = 'microblogPCU/user_post.csv'
    # repost_num:转发数量
    # commnet_num: 评论数量
    post_data = pd.read_csv(postfile)
    save = []

    for x, y in post_data.groupby(by='poster_id'):
        content = " ".join(sent2word(" ".join([i for i in y['content'].values if isinstance(i, str)])))
        repost_num = y['repost_num'].sum()
        comment_num = y['comment_num'].sum()
        rows = {"user": x, "content": content, 'repost_num': repost_num, 'comment_num': comment_num}
        save.append(rows)
    df = pd.DataFrame(save)
    df.to_excel("user_content.xlsx", index=None)


def cluster_user(method='soft'):
    # 利用gensim将doc转换为向量
    x_train = []
    train_data = pd.read_excel("user_content.xlsx")
    print(train_data.shape)
    poster_id = train_data['user']
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    limit = 800
    for i, y in enumerate(train_data['content'].values):
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

    if method == 'hard':
        print("============= Kmeans 硬聚类 =============")
        kmean_model = KMeans(n_clusters=n_cluster, init='k-means++', n_init=10,
                             max_iter=1000, tol=1e-4, precompute_distances='auto',
                             verbose=0, random_state=None, copy_x=True,
                             n_jobs=None, algorithm='auto')
        kmean_model.fit(infered_vectors_list)
        cluster_label = kmean_model.labels_

        df = pd.DataFrame()
        df['user'] = poster_id
        df['label'] = cluster_label
        df.to_excel("硬聚类结果.xlsx", index=None)
        print(df.shape)
    else:
        print("=============GaussianMixture 软聚类 =============")
        clf = GaussianMixture(n_components=n_cluster, covariance_type='full', tol=1e-3,
                              reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                              weights_init=None, means_init=None, precisions_init=None,
                              random_state=None, warm_start=False,
                              verbose=0, verbose_interval=10)
        clf.fit(np.array(infered_vectors_list))
        cluster_label = clf.predict(np.array(infered_vectors_list))
        print(len(cluster_label), len(poster_id))
        df = pd.DataFrame()
        df['user'] = poster_id
        df['label'] = cluster_label
        df.to_excel("软聚类结果.xlsx", index=None)
        print(df.shape)


def final():
    # 合并聚类结果和用户信息
    file_name = "硬聚类结果.xlsx"
    data_results = pd.read_excel(file_name)

    def get_class(inputs):
        try:
            return data_results[data_results['user'].isin([inputs])]['label'].values[0]
        except:
            return "没有发微博的用户"

    weibo_user = 'microblogPCU/weibo_user.csv'
    weibo_user_data = pd.read_csv(weibo_user)
    labels = []
    for x, y in weibo_user_data.iterrows():
        label = get_class(y['user_id'])
        labels.append(label)

    weibo_user_data['label'] = labels
    weibo_user_data.to_excel("weibo_user_{}.xlsx".format(file_name.replace(".xlsx", "")), index=None)


if __name__ == "__main__":

    method = 'cluster_user'

    if method == 'cut_content':
        cut_content()

    if method == 'cluster_user':
        method = 'soft'
        cluster_user(method=method)

    if method == 'final':
        final()
