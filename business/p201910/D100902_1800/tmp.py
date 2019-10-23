from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def trans(inputs):
    if inputs == '积极':
        return 1
    elif inputs == '中性':
        return 0
    else:
        return -1
data = pd.read_excel("情感分析结果.xlsx")

data['senti2vec'] = data['senti'].apply(trans)

train_data = data.loc[:data.shape[0] * 0.7, :]
test_data = data.loc[data.shape[0] * 0.7:, :]

train_x, train_y, test_x, test_y = train_data['content'], train_data['senti2vec'], test_data['content'], test_data[
    'senti2vec']
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

con = CountVectorizer(binary=True)
train_x = con.fit_transform(train_x)
test_x = con.transform(test_x)

test_y = label_binarize(test_y, classes=[0, 1, -1])
print(test_y.shape)
train_y = label_binarize(train_y, classes=[0, 1, -1])
n_classes = 3

# classifier
clf = OneVsRestClassifier(GaussianNB())
clf.fit(train_x.toarray(), train_y)
y_score = clf.predict(test_x.toarray())

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])






