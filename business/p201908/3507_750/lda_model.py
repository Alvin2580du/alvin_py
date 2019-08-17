import pandas as pd
import jieba
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import models


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def build_data():
    save = []
    num = 0
    for file in ['水果', '猪瘟', '猪肉', '鸡蛋']:
        num += 1
        data = pd.read_excel("{}.xlsx".format(file))
        print(data.shape)
        print(data.columns)
        for x, y in data.iterrows():
            cont = "{} {}".format(y['weibos'], y['zhuanfa'])
            cont_cut = " ".join([i for i in jieba.lcut(cont) if len(i) > 1 and is_chinese(i)])
            if len(cont_cut.split()) > 2:
                rows = {"content": cont_cut, 'target_names': file, 'target': num}
                save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("dataAll.xlsx", index=None)
    print(df.shape)


if __name__ == '__main__':
    method = 'LDA'

    if method == 'build_data':
        build_data()

    if method == 'LDA':
        data = pd.read_excel("dataAll.xlsx")
        train_set = []
        lines = data['content'].values

        for line in lines:
            train_set.append([i for i in line.split()])

        dictionary = Dictionary(train_set)
        corpus = [dictionary.doc2bow(text) for text in train_set]  # 构建稀疏向量
        tfidf = models.TfidfModel(corpus)  # 统计tfidf
        corpus_tfidf = tfidf[corpus]  # 得到每个文本的tfidf向量，稀疏矩阵
        num_topics = 12
        lda_model = LdaModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics, iterations=1)
        top_topics = lda_model.top_topics(corpus)
        saves = []
        averages = []
        print_topics = []
        fw = open('lda model topicRestlts.txt', 'w', encoding='utf-8')

        for topic_id, top_topic in enumerate(top_topics):
            tc = sum([t[0] for t in top_topic[0]])
            averages.append(tc)
            rows = {}
            rows['# of topic'] = topic_id
            rows['topic coherence'] = "{:0.3f}".format(tc)
            saves.append(rows)
            fw.writelines(" ".join(["{}*{:0.3f}".format(t[1], t[0]) for t in top_topic[0]]) + "\n")
            print(" ".join(["{}*{:0.3f}".format(t[1], t[0]) for t in top_topic[0]]))

        df = pd.DataFrame(saves)
        df.to_csv("lda coherence.csv", index=None)

        print(df.shape)
        print("LDA coherence 平均值：{}".format(sum(averages) / num_topics))


