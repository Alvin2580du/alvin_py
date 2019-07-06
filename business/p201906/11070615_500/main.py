import pandas as pd
import os
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import models
import jieba


def get_month(inputs):
    return inputs[:6]

sy = '。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑•¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼❝❞'

save = []

for file in os.listdir('./hot'):
    with open(os.path.join('./hot', file), 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line_sp = line.split('	')
            rows = {}
            rows['date'] = file.replace('.txt', '')
            rows['content'] = " ".join([i for i in jieba.lcut(line_sp[3]) if i not in sy])
            save.append(rows)

df = pd.DataFrame(save)
df.to_excel("train.xlsx", index=None)

# LDA 主题模型
# 构建训练语料
Listdata = df['content'].values.tolist()
train_set = [listi.replace("\n", "").strip().split(' ') for listi in Listdata if isinstance(listi, str)]
dictionary = Dictionary(train_set)
corpus = [dictionary.doc2bow(text) for text in train_set]  # 构建稀疏向量
tfidf = models.TfidfModel(corpus)  # 统计tfidf
corpus_tfidf = tfidf[corpus]  # 得到每个文本的tfidf向量，稀疏矩阵
lda = LdaModel(corpus_tfidf, id2word=dictionary, num_topics=50, iterations=100)


def get_topic(test_doc):
    test_doc = list(jieba.cut(test_doc))  # 新文档进行分词
    doc_bow = dictionary.doc2bow(test_doc)  # 文档转换成bow
    doc_lda = lda[doc_bow]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    out_topic = []
    prob_list = []
    for topic in doc_lda:
        res = lda.print_topic(topic[0])
        out_topic.append(res)
        prob_list.append(topic)
    return " ".join(out_topic), prob_list


results = []
df['month'] = df['date'].apply(lambda x: x[:6])
for x, y in df.groupby(by='month'):
    topic, prob = get_topic("".join(y['content'].values.tolist()))
    rows = {}
    rows['month'] = x
    rows['topic'] = topic
    rows['prob'] = prob
    results.append(rows)

df = pd.DataFrame(results)
df.to_excel("results.xlsx", index=None)
print(df.shape)

