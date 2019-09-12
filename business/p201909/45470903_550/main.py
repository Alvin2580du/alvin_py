import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec
import jieba
from collections import OrderedDict

jieba.load_userdict("userdict.csv")

TaggededDocument = gensim.models.doc2vec.TaggedDocument


def cut_data():
    data = pd.read_excel("News.xlsx")

    print(data.columns)
    fw = open('content.csv', 'w', encoding='utf-8')

    for line in data['Content']:
        line_cut = [i for i in jieba.lcut(line) if len(i) > 1]
        fw.writelines(" ".join(line_cut) + "\n")


def get_datasest():
    with open("content.csv", 'r', encoding='utf-8') as cf:
        docs = cf.readlines()
        x_train = []
        for i, text in enumerate(docs):
            word_list = text.split(" ")
            l = len(word_list)
            word_list[l - 1] = word_list[l - 1].strip()
            document = TaggededDocument(word_list, tags=[i])
            x_train.append(document)
        return x_train


def train(x_train, size=200):
    print("模型训练中，请等待...")
    model_dm = Doc2Vec(x_train, min_count=3, window=5, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=100, report_delay=1.0)
    model_dm.save('model.m')
    return model_dm


if __name__ == '__main__':

    method = 'cut_data'

    if method == 'cut_data':
        cut_data()

    if method == 'train':
        x_train = get_datasest()
        model_dm = train(x_train)

    if method == 'one':
        model_dm = Doc2Vec.load("model.m")
        word = u'自动驾驶'
        save = []
        for i in model_dm.wv.most_similar(word, topn=30):
            rows = OrderedDict({"words": word, 'most_similar': i[0], 'degree': i[1]})
            save.append(rows)
            print(rows)

        df = pd.DataFrame(save)
        df.to_csv("one_results.csv", index=None)
        print(df.shape)

    if method == 'two':
        """
        从2017年7月至9月的领域实体数据库中各选取前30个高频领域实体，
        即共计60个目标领域实体，利用Word2vec模型输出与目标领域实体最相关的50个词汇，
        """

        data = pd.read_csv("ner.csv")
        # word,label,date
        save = []
        model_dm = Doc2Vec.load("model.m")

        for i, date in data.groupby(by='label'):
            for j, label in date.groupby(by='date'):
                words = label['word'].values
                for w in words:
                    try:
                        for ps in model_dm.wv.most_similar(u"{}".format(w), topn=50):
                            rows = OrderedDict({"date": i,
                                                "label": j,
                                                "words": w,
                                                'most_similar': ps[0],
                                                'degree': ps[1]})
                            save.append(rows)
                    except:
                        continue

        df = pd.DataFrame(save)
        df.to_csv('two_results.csv', index=None)
        print(df.shape)

    if method == 'three':
        """
        首先，利用分析源训练的Word2vec模型寻找与“海康威视”最相关的30个词汇，
        再分寻找与这30个词汇最相关的50个词汇，将输出结果转化成边文件格式得到30*50共1500条数据
        """
        save = []

        model_dm = Doc2Vec.load("model.m")
        num = 1
        root_word = u'海康威视'
        for i in model_dm.wv.most_similar(root_word, topn=100):
            try:
                for j in model_dm.wv.most_similar(u"{}".format(i[0]), topn=50):
                    rows = OrderedDict({"words": i[0], 'most_similar': j[0], 'degree': j[1]})
                    save.append(rows)
                num += 1

            except:
                continue

            if num > 30:
                break

        df = pd.DataFrame(save)
        df.to_csv('three_results_{}.csv'.format(root_word), index=None)
        print(df.shape)

    if method == 'four':
        """
        首先，以“百度”、“腾讯”、“阿里巴巴”、“阿里”（由于很多媒体习惯用阿里代替阿里巴巴进行报道，
        此处将将阿里加入目标关键词序列）为目标关键词，通过Word2vec模型分别输出200个最相关词汇，
        """
        save = []

        model_dm = Doc2Vec.load("model.m")
        num = 1
        root_word = [u'百度', u'腾讯', u'阿里巴巴', u'阿里']
        for rw in root_word:
            for i in model_dm.wv.most_similar(rw, topn=200):
                rows = OrderedDict({"words": rw, 'most_similar': i[0], 'degree': i[1]})
                save.append(rows)

        df = pd.DataFrame(save)
        df.to_csv('four_results.csv', index=None)
        print(df.shape)
