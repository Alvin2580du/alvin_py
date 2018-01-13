from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans

from sklearn.datasets import fetch_20newsgroups

use_hashing = False
use_idf = False
minibatch = True
verbose = False
n_features = 10000
n_components = 10
true_k = 10

if use_hashing:
    if use_idf:
        hasher = HashingVectorizer(n_features=n_features, stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=n_features, stop_words='english', alternate_sign=False,
                                       norm='l2', binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=2, stop_words='english',
                                 use_idf=use_idf)

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
print(dataset.data)
exit(1)
X = vectorizer.fit_transform(dataset.data)

svd = TruncatedSVD(n_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

if minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000, verbose=verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, verbose=verbose)

km.fit(X)

if use_hashing:
    if n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        for ind in order_centroids[i, :10]:
            x1 = terms[ind]

if not use_hashing:
    if n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        for ind in order_centroids[i, :10]:
            x1 = terms[ind]
            print(x1)


# 问题识别

