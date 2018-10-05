import pandas as pd
from collections import Counter
import numpy as np
import os
import time

stop_words = pd.read_csv("stopwords.csv", header=None).values.tolist()
stop_words = [i for j in stop_words for i in j]
tfidf_matrix_csv = 'tfidf_matrix.csv'
dataset = None


def get_top_1000():
    data = dataset  # pd.read_csv("datasets.csv", usecols=['content'])
    mydoclist = []

    for doc in data.values.tolist():
        mydoclist.append(doc[0])

    words_list = []
    for x in mydoclist:
        for i in x.split():
            if i not in stop_words:
                words_list.append(i)
    rows = {}

    # for x, y in Counter(words_list).most_common(1000):
    index = 0
    for x, _ in Counter(words_list).most_common(1000):
        rows[x] = index
        index += 1
    return rows


def cosine_similarities(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0:
        return 0
    num = float(vector_a * vector_b.T)
    # cos = num / denom
    # sim = 0.5 + 0.5 * cos
    # return sim
    return num / denom


def tf_of_doc(top_1000, word_list):
    '''
    return :  a vector of 1000 dimensions representing a document

    '''
    # a dict used for save word's count
    doc_tf = {}.fromkeys(top_1000.keys(), 0)

    # all words count in this doc
    count = len(word_list)

    # tf=word_count/all_word_count
    # word_counter=Counter(word_list)
    for word in word_list:
        doc_tf[word] += 1

    # convert dict.values to numpy array
    return np.array(list(doc_tf.values())) / count


def build_vec2():
    # 保存语料库的，所有doc词频
    # 这是一个numpy数组
    # [[0,0.2,0,0,...],(numpy数组第1维) 1000列，代表词袋中的词的个数
    # [0,0,0.3,0,...],
    # ...
    # [0,0,0,0.1,...],
    # [0,0.1,0,0,...]]
    # numpy数组第0维,共1400行 ,表示1400个文档
    start_time = time.time()
    corpus_tf = np.zeros((1400, 1000))
    docsList = get_doc_list()
    end_time = time.time()
    print("load doc list cost:", end_time - start_time)

    start_time = time.time()
    top_1000 = get_top_1000()
    end_time = time.time()
    print("comptue top  1000 cost:", end_time - start_time)

    start_time = time.time()
    # get  words freq  in corpus
    for i, doc in enumerate(docsList):
        if doc != "":
            tf_doc = tf_of_doc(top_1000, doc.split(' '))
            corpus_tf[i, :] = tf_doc

    end_time = time.time()
    print("corpus_tf cost:", end_time - start_time)
    # compute idf and build svm
    start_time = time.time()
    vsm = np.zeros(corpus_tf.shape)
    for i, doc in enumerate(docsList):
        # Statistically appeard words for this doc
        # 利用 set 去重。对于idf来说，只要词在文档中只现就可以了。词频在上面已经计算完了
        doc_words = set(doc.split())

        # get indxis in top1000,use for find which doc contain words
        # 词在top1000 的序号
        word_indices = [top_1000[k] for k in doc_words]
        # check all doc appeard words
        contain_words_docs = corpus_tf[:, word_indices] > 0
        # compute words nums, by numpy array's  axis=0
        word_nums = np.sum(contain_words_docs, axis=0)
        # compute idf
        idf = np.log(1400 / (word_nums + 1))
        # for this doc compute tf.idf=tf*idf
        vsm[i, word_indices] = idf * corpus_tf[i, word_indices]
    end_time = time.time()
    print("vsm cost:", end_time - start_time)

    tfidf_matrix_df = pd.DataFrame(vsm)
    tfidf_matrix_df.columns = list(top_1000.keys())

    tfidf_matrix_df.to_csv(tfidf_matrix_csv, index=None)
    print(tfidf_matrix_df.shape)


def get_doc_list():
    mydoclist = []
    data = dataset  # pd.read_csv("datasets.csv", usecols=['content'])
    top_1000_words = get_top_1000()
    for doc in data.values.tolist():
        doc_tmp = []
        for w in doc[0].split():
            if w in list(top_1000_words.keys()):
                doc_tmp.append(w)
        mydoclist.append(" ".join(doc_tmp))
    return mydoclist


def get_doc_vec(top1000, word_list):
    tf = tf_of_doc(top1000, word_list)
    doc_words = set(word_list)
    word_indices = [top1000[k] for k in doc_words]
    contain_words_docs = tf[word_indices] > 0
    word_nums = np.sum(contain_words_docs, axis=0)
    # the query as a new doc ,so all document count is 1401
    idf = np.log(1401 / (word_nums + 1))
    return tf * idf


def find_top10_doc_task2(words_list, top1000):
    tfidf_matrix_df = pd.read_csv(tfidf_matrix_csv)
    vec = get_doc_vec(top1000, words_list)
    tmp_cos = {}
    # tfidf_matrix_df.shape[0]=1400 ,all docs
    for row in range(tfidf_matrix_df.shape[0]):
        res = cosine_similarities(tfidf_matrix_df.values[row], vec)
        tmp_cos[row] = res
    tmp_cos_sorted = sorted(tmp_cos.items(), key=lambda x: x[1], reverse=True)
    top_10 = tmp_cos_sorted[:10]
    return top_10


def find_top10_doc_task3(words_list, top1000):
    tfidf_matrix_df = pd.read_csv(tfidf_matrix_csv)
    vec = get_doc_vec(top1000, words_list)
    # find which doc contain words
    words_indices = [top1000[k] for k in words_list]
    check_special_words = tfidf_matrix_df.values[:, words_indices] > 0
    # axis=1 means compute sum by row.select the doc which contain any query word
    docs_contain_words = np.sum(check_special_words, axis=1)
    docsid = np.where(docs_contain_words > 0)[0]

    tmp_cos = {}
    for row in docsid:
        res = cosine_similarities(tfidf_matrix_df.values[row], vec)
        tmp_cos[row] = res
    tmp_cos_sorted = sorted(tmp_cos.items(), key=lambda x: x[1], reverse=True)
    top_10 = tmp_cos_sorted[:10]
    return top_10


def search_words(words):
    top1000 = get_top_1000()
    print("Your query words: {}".format(" ".join(words.split(" "))))
    words = words.split(" ")
    words_hit_top1000 = [w for w in words if w in top1000]

    if not words_hit_top1000:
        return None
    top_10 = find_top10_doc_task2(words, top1000)
    print("The top 10 similarities documents :")
    for docId, cos in top_10:
        print("             docId: cranfield%04d " % docId)


def main():
    global dataset
    dataset = pd.read_csv("datasets.csv", usecols=['content'])

    if not os.path.exists(tfidf_matrix_csv):
        build_vec2()
    words = 'method'  # transfer equations  free problem case
    search_words(words)


if __name__ == '__main__':
    main()
