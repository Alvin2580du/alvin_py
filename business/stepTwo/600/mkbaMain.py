import sqlite3 as db
import time
import pandas as pd
import pickle

from sklearn.decomposition import KernelPCA, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

t0 = time.time()
# WRITE YOUR CODE HERE
#####################2.1###############
# How many tweets in the training set contain at least one of the strings “AT&T”, “Verizon” or “T-Mobile” (1.5 Marks)
YOURSTUDENTID = 470265589

# Use this area to load the data
data = open('churn_survey.json').read()
df_cust = pd.read_json('churn_survey.json')
df_cust.head()

seven_data = df_cust[['Gender', 'OnlineBackup', 'OnlineSecurity', 'StreamingMovies', 'StreamingTV', 'TechSupport']]
list_res = []
for g in ['Female', 'Male']:
    gen_res = [g]
    for addon in ['OnlineBackup', 'OnlineSecurity', 'StreamingMovies', 'StreamingTV', 'TechSupport']:
        total = seven_data.loc[seven_data[addon] == 'Yes', addon].count()
        buy_yes = seven_data.loc[(seven_data['Gender'] == g) &
                                 (seven_data[addon] == 'Yes'), addon].count()
        rate = buy_yes / total
        gen_res.append(rate)
result_1_7 = pd.DataFrame(list_res).round(4)
result_1_7.to_csv("ID_{0}_Q_{1}.csv".format(YOURSTUDENTID, '1_7'), index=None)


# This function is used to save answers with a non-tabular output
def write_txt(student_id, part_number, data):
    file = open("ID_{0}_Q_{1}.txt".format(student_id, part_number), 'w')
    file.write(data)
    file.close()


# Use this area to load the data
import numpy as np

data = open('churn_survey.json').read()
df_cust = pd.read_json('churn_survey.json')
df_cust.head()

# WRITE YOUR CODE HERE
method_name = df_cust["PaymentMethod"].value_counts().index[0]
# This will save your answer to a .txt file
write_txt(YOURSTUDENTID, "1_1", method_name)

# WRITE YOUR CODE HERE
cust_groups = df_cust.groupby("Churn")
charge_mean = cust_groups.mean()['MonthlyCharges'].round(4).reset_index()
charge_mean.to_csv("ID_{0}_Q_{1}.csv".format(YOURSTUDENTID, "1_2"), index=None)
# WRITE YOUR CODE HERE
amountspent_sd = cust_groups.std()['MonthlyCharges'].round(4).reset_index()
amountspent_sd.to_csv("ID_{0}_Q_{1}.csv".format(YOURSTUDENTID, "1_3"), index=None)

Churn_k = df_cust.groupby(['Churn', 'Contract']).size().reset_index()
Churn_k.columns = ['Churn', 'Contract', 'Number']
Churn_k1 = Churn_k.pivot(columns='Contract', index='Churn', values='Number').reset_index()
Churn_k1.head()

churn_count = cust_groups.count().iloc[:, 0].reset_index()
combine = pd.concat([churn_count, Churn_k1], axis=1)
combine.head()

Churn_k1['Month-to-month'] = combine['Month-to-month'] * 100 / combine['Contract']
Churn_k1['One year'] = combine['One year'] * 100 / combine['Contract']
Churn_k1['Two year'] = combine['Two year'] * 100 / combine['Contract']
Churn_k1 = Churn_k1.round(2)
Churn_k1.to_csv("ID_{0}_Q_{1}.csv".format(YOURSTUDENTID, "1_4"), index=None)

# WRITE YOUR CODE HERE
cor = df_cust.replace(['No', 'Yes', 'Female', 'Male', 'No internet service', 'No phone service', ], [0, 1, 0, 1, 0, 0])
cor_dum = pd.get_dummies(cor, drop_first=True)
cor.head()

cor_1 = cor_dum.corr().loc['MonthlyCharges']
cor_abs = cor_1.abs().sort_values(ascending=False)
cor_abs.head()

sort_cor = pd.DataFrame(cor_abs).drop(['MonthlyCharges'])
most_cor = sort_cor.iloc[0:1, 0:1].reset_index()
most_cor.columns = ['Feature', 'Value']
result_1_5 = pd.DataFrame(most_cor)
result_1_5.to_csv("ID_{0}_Q_{1}.csv".format(YOURSTUDENTID, "1_5"), index=None)

# WRITE YOUR CODE HERE
df_inf = df_cust['Tenure'].describe().round(1)
df_1_6 = pd.DataFrame(df_inf)
df_1_6.to_csv("ID_{0}_Q_{1}.csv".format(YOURSTUDENTID, "1_6"))

######## 1.7



############ 1.8



# WRITE YOUR CODE HERE 2.1
import re
import sqlite3

con = sqlite3.connect("tweets.db")
df_tweets_sql = pd.read_sql("select * from tweets", con)
df_churn_sql = pd.read_sql("select * from churn", con)
churn_tr = df_churn_sql[~df_churn_sql['set'].isin(['hidden'])]
df_merged = (pd.merge(churn_tr, df_tweets_sql, how='left', on='tid'))
df_merged = df_merged.dropna()
print(df_merged.columns.tolist())
# ['tid', 'churn', 'set', 'uid', 'date', 'text']
df_merged.to_csv("df_merged.csv", index=None)
matches = df_merged.text.str.contains('AT&T|Verizon|T-Mobile', flags=re.IGNORECASE, na=False)
result = matches.value_counts()
print(result)
number_tweets = '2492'
# This will save your answer to a .txt file
write_txt(YOURSTUDENTID, "2_1", number_tweets)
######### 2.2
# WRITE YOUR CODE HERE
import re

matches_1 = df_merged.loc[
    df_merged.text.str.contains('AT&T|Verizon|T-Mobile', flags=re.IGNORECASE, na=False), ['tid', 'text']]

# 2.3
# WRITE YOUR CODE HERE
from sklearn.feature_extraction.text import TfidfVectorizer

merged_text = list(df_merged['text'].dropna(axis=0, how='any'))
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(merged_text)
# 2.3.2 Dimension Reduction (1.5 Marks)¶

# WRITE YOUR CODE HERE
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=5)
X_transformed = svd.fit_transform(X)

# #### 2.3.3 Tuning (2 Marks)
param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
              'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
              }
lr_grid = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
lr_grid.fit(X_transformed, df_merged['churn'].values)
par = lr_grid.best_params_
sco = lr_grid.best_score_
print("par:{}, sco:{}".format(par, sco))

# This code will save your GridSearchCV or RandomisedSearchCV to a file
filename = "ID_{0}_Q_2_3_3.pickle".format(YOURSTUDENTID)
# MYGRIDSEARCHOBJECT must be GridSearchCV or RandomisedSearchCV
s = pickle.dump(lr_grid, open(filename, 'wb'))

################ 2.3.4 Output Model (1 Marks)
lr_model = LogisticRegression(C=0.001, solver='lbfgs')
lr_model.fit(X_transformed, df_merged['churn'].values)

# This code will save your LogisticRegression to a file
filename = "ID_{0}_Q_2_3_4.pickle".format(YOURSTUDENTID)
# MYLOGISTICREGRESSION must be of type sklearn.linear_model.LogisticRegression
s = pickle.dump(lr_model, open(filename, 'wb'))

################# 2.3.5 Predicting Churn for the Hidden Customers (4 Marks)
print("================*=========================")

churn_tr_test = df_churn_sql[df_churn_sql['set'].isin(['hidden'])]
df_mergedTest = (pd.merge(churn_tr_test, df_tweets_sql, how='left', on='tid'))
df_mergedTest_copy = df_mergedTest.copy()
print("df_mergedTest.shape:{}".format(df_mergedTest.shape))
tfidf_transformer = TfidfVectorizer()
tfidf_transformer.fit_transform(df_mergedTest['text'])
test_sparse = tfidf_transformer.transform(df_mergedTest['text']).toarray()
print(test_sparse)
transformerDR = KernelPCA(n_components=5)
transformerDR.fit_transform(test_sparse)
test_X = transformerDR.transform(test_sparse)
pre_result = lr_model.predict(test_X)
print("pre_result.shape:{}".format(pre_result.shape))
df_mergedTest['Churn'] = pre_result
del df_mergedTest['churn']
del df_mergedTest['set']
del df_mergedTest['uid']
del df_mergedTest['date']
del df_mergedTest['text']

df_mergedTest.to_csv("ID_{}_Q_2_3_5.csv".format(YOURSTUDENTID), index=None)
############### 2.4 Prediction Competition (Total 10 Marks)
Pipelinemodel = Pipeline([("pca", PCA(n_components=5)),
                          ("clf", LogisticRegression(C=0.001, solver='lbfgs'))])
Pipelinemodel.fit(X_transformed, y=df_merged['churn'].values)
filename = "ID{0}Q_2_4_1.pickle".format(YOURSTUDENTID)
# MYLOGISTICREGRESSION must be of type sklearn.linear_model.LogisticRegression
s = pickle.dump(Pipelinemodel, open(filename, 'wb'))
# predict model
pre = Pipelinemodel.predict(test_X)
df_mergedTest_copy['Churn'] = pre

del df_mergedTest_copy['churn']
del df_mergedTest_copy['set']
del df_mergedTest_copy['uid']
del df_mergedTest_copy['date']
del df_mergedTest_copy['text']
df_mergedTest_copy.to_csv("ID{0}Q_2_4_1.csv".format(YOURSTUDENTID), index=None)
df_mergedTest.shape
######### END
print("time cost:{}".format(time.time() - t0))
