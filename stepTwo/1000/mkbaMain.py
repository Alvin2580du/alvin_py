import sqlite3 as db
import time
import pandas as pd
import pickle

from sklearn.decomposition import FastICA, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

t0 = time.time()
# WRITE YOUR CODE HERE
#####################2.1###############
# How many tweets in the training set contain at least one of the strings “AT&T”, “Verizon” or “T-Mobile” (1.5 Marks)
YOURSTUDENTID = '客户ID'


def readFronSqllite(db_path, exectCmd):
    conn = db.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(exectCmd)
    rows = cursor.fetchall()
    return rows


churn = readFronSqllite('tweets.db', exectCmd="select * from churn")
tweets = readFronSqllite('tweets.db', exectCmd="select * from tweets")


def find_spacial_symbol(inputs):
    if "AT&T" or "Verizon" or "T-Mobile" in inputs.split():
        return True
    else:
        return False


def find_train_id():
    train_ids = []
    hidden_ids = []
    for x in churn:
        if x[2] == 'training':
            train_ids.append(x[0])
        else:
            hidden_ids.append(x[0])
    return train_ids, hidden_ids


train_ids, hidden_ids = find_train_id()


def write_txt(student_id, part_number, data):
    file = open("ID_{0}_Q_{1}.txt".format(student_id, part_number), 'w')
    file.write(data)
    file.close()


def get_num():
    number_tweets = 0
    for x in tweets:
        if isinstance(x[0], float):
            if x[0] in train_ids:
                if "AT&T" or "Verizon" or "T-Mobile" in x[3]:
                    number_tweets += 1
    return number_tweets


number_tweets = str(get_num())
print(number_tweets)
# This will save your answer to a .txt file
write_txt(YOURSTUDENTID, "2_1", number_tweets)


#####################2.2###############
#  Find all tweets in the training set containing the strings “AT&T”, “Verizon” and “switch” (2.5 Marks)

fw = open("ID_{}_Q_2_2.csv".format(YOURSTUDENTID), 'w')
for x in tweets:
    if x[0] in train_ids:
        if isinstance(x[0], float):
            rows = "| {}  | {}  |".format(int(x[0]), x[3])
            fw.writelines(rows + '\n')


########################2.3########################
#  Identify Churning Customers via Logistic Regression
# 2.3.1 Transform Features (1.5 Marks)
def get_label(tweet_id):
    for x in churn:
        if isinstance(x[0], float):
            if int(x[0]) == tweet_id:
                label = x[1]
                return label


texts = []
with open("ID_{}_Q_2_2.csv".format(YOURSTUDENTID), 'r') as fr:
    lines = fr.readlines()
    for line in lines:
        t_id = int("{}".format(line.split(' | ')[0].replace(" ", "").replace("|", "")))
        label = get_label(t_id)
        rows = {'text': line.replace("\n", "").split(" | ")[1], 'label': label}
        texts.append(rows)

train_df = pd.DataFrame(texts)

tfidf_transformer = TfidfVectorizer()
tfidf_transformer.fit_transform(train_df['text'])
dense_result = tfidf_transformer.transform(train_df['text']).todense()
print(dense_result.shape)

# This code will save your Transformer/Vectoriser object to a file

filename = "ID_{0}_Q_2_3_1.pickle".format(YOURSTUDENTID)

# MYTRANSFORMEROBJECT must be a sklearn transformer or vectoriser
s = pickle.dump(tfidf_transformer, open(filename, 'wb'))

################# 2.3.2 Dimension Reduction (1.5 Marks)
transformerDR = FastICA(n_components=5)
X_transformed = transformerDR.fit_transform(dense_result)
print("X_transformed shape:{}".format(X_transformed.shape))

# #### 2.3.3 Tuning (2 Marks)
param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
              'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
              }
lr_grid = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
lr_grid.fit(X_transformed, train_df['label'].values)
par = lr_grid.best_params_
sco = lr_grid.best_score_
print("par:{}, sco:{}".format(par, sco))

# This code will save your GridSearchCV or RandomisedSearchCV to a file
filename = "ID_{0}_Q_2_3_3.pickle".format(YOURSTUDENTID)
# MYGRIDSEARCHOBJECT must be GridSearchCV or RandomisedSearchCV
s = pickle.dump(lr_grid, open(filename, 'wb'))

################ 2.3.4 Output Model (1 Marks)
lr_model = LogisticRegression(C=0.001, solver='lbfgs')
lr_model.fit(X_transformed, train_df['label'].values)

# This code will save your LogisticRegression to a file
filename = "ID_{0}_Q_2_3_4.pickle".format(YOURSTUDENTID)
# MYLOGISTICREGRESSION must be of type sklearn.linear_model.LogisticRegression
s = pickle.dump(lr_model, open(filename, 'wb'))

################# 2.3.5 Predicting Churn for the Hidden Customers (4 Marks)
print("================*=========================")

test_set = []
for x in tweets:
    if x[0] in hidden_ids:
        test_set.append({'text': x[3]})

test_set_df = pd.DataFrame(test_set)
test_sparse = tfidf_transformer.transform(test_set_df['text']).toarray()

transformerDR = FastICA(n_components=5)
transformerDR.fit_transform(test_sparse)
test_X = transformerDR.transform(test_sparse)
pre_result = lr_model.predict(test_X)
print("pre_result.shape:{}".format(pre_result.shape))
test_set_df['pre'] = pre_result
test_set_df.to_csv("ID_{}_Q_2_3_5.csv".format(YOURSTUDENTID), index=None)

############### 2.4 Prediction Competition (Total 10 Marks)
pip_model = Pipeline([("pca", PCA(n_components=5)),
                      ("clf", LogisticRegression(C=0.001, solver='lbfgs'))])
pip_model.fit(X_transformed, y=train_df['label'].values)
filename = "ID{0}Q_2_4_1.pickle".format(YOURSTUDENTID)
# MYLOGISTICREGRESSION must be of type sklearn.linear_model.LogisticRegression
s = pickle.dump(pip_model, open(filename, 'wb'))
# predict model
pre = pip_model.predict(test_X)
test_set_df['pred'] = pre
del test_set_df['pre']
test_set_df.to_csv("ID{0}Q_2_4_1.csv".format(YOURSTUDENTID), index=None)
print(test_set_df.shape)

######### END
print("time cost:{}".format(time.time() - t0))
