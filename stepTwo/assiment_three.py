import numpy as np
import math
import pandas as pd
from collections import Counter

stop_words = ['the', 'of', 'and', 'a', 'in']


def get_top_1000():
    data = pd.read_csv("datasets.csv", usecols=['content'])
    mydoclist = []

    for doc in data.values.tolist():
        mydoclist.append(doc[0])

    words_list = []
    for x in mydoclist:
        for i in x.split():
            if i not in stop_words:
                words_list.append(i)
    rows = {}
    for x, y in Counter(words_list).most_common(1000):
        rows[x] = y
    return rows

rows = get_top_1000()
print(rows)
exit(1)

def tf(term, document):
    return freq(term, document)


def freq(term, document):
    return document.split().count(term)


def l2_normalizer(vec):
    denom = np.sum([el ** 2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]


def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return lexicon


def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount += 1
    return doccount


def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1 + df)


def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat


def get_doc_term_matrix_l2():
    vocabulary = build_lexicon(mydoclist)

    my_idf_vector = [idf(word, mydoclist) for word in vocabulary]

    my_idf_matrix = build_idf_matrix(my_idf_vector)

    doc_term_matrix = []
    for doc in mydoclist:
        tf_vector = [tf(word, doc) for word in vocabulary]
        tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
        doc_term_matrix.append(tf_vector)

    doc_term_matrix_tfidf = []
    for tf_vector in doc_term_matrix:
        doc_term_matrix_tfidf.append(np.dot(tf_vector, my_idf_matrix))

    doc_term_matrix_tfidf_l2 = []
    for tf_vector in doc_term_matrix_tfidf:
        doc_term_matrix_tfidf_l2.append(l2_normalizer(tf_vector))

    doc_term_matrix_l2 = []
    for vec in doc_term_matrix:
        doc_term_matrix_l2.append(l2_normalizer(vec))

    return np.matrix(doc_term_matrix_l2)


doc_term_matrix_l2 = get_doc_term_matrix_l2()
print(doc_term_matrix_l2.shape)
