from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import numpy as np
import docx
import os

stopwords = []
doc_file = docx.Document('暂停词或介词.docx')
for paragraph in doc_file.paragraphs:
    stopwords.append(paragraph.text.strip())

fumian = []
doc_file = docx.Document('负面词.docx')
for paragraph in doc_file.paragraphs:
    fumian.append(paragraph.text.strip())


def get_label(inputs):
    for i in fumian:
        if i in inputs:
            # 负面
            return 0
    # 正面
    return 1


def train():
    text_cut_train = []
    text_cut_label = []
    for file in os.listdir("./data"):
        file_name = os.path.join('./data', file)
        doc_file = docx.Document(file_name)
        k = 0
        for paragraph in doc_file.paragraphs:
            labels = get_label(paragraph.text)
            k += 1
            text_cut = [i for i in paragraph.text.split() if i not in stopwords]
            text_cut_train.append(" ".join(text_cut))
            text_cut_label.append(labels)

    tfidf2 = TfidfVectorizer()
    data = tfidf2.fit_transform(text_cut_train)
    x_train, x_test, y_train, y_test = train_test_split(data, np.array(text_cut_label), test_size=0.3)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    joblib.dump(model, 'model.m')
    y_test_ = model.predict(x_test)
    acc = accuracy_score(y_test, y_test_)
    print('准确率：')
    print(acc)
