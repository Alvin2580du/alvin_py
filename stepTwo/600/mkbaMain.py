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
YOURSTUDENTID = 470265589


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
import sqlite3

con = sqlite3.connect("tweets.db")
df_tweets_sql = pd.read_sql("select * from tweets", con)
df_churn_sql = pd.read_sql("select * from churn", con)
churn_tr = df_churn_sql[~df_churn_sql['set'].isin(['hidden'])]
df_merged = (pd.merge(churn_tr, df_tweets_sql, how='left', on='tid'))
matches = df_merged.text.str.contains('AT&T|Verizon|T-Mobile', flags=re.IGNORECASE, na=False)
result = matches.value_counts()
number_tweets = '2492'
# This will save your answer to a .txt file
write_txt(YOURSTUDENTID, "2_1", number_tweets)


######### 2.2
# WRITE YOUR CODE HERE
import re
matches_1 = df_merged.loc[df_merged.text.str.contains('AT&T|Verizon|T-Mobile',flags=re.IGNORECASE,na=False), ['tid', 'text']]
matches_1.to_csv("ID_{0}_Q_{1}.csv".format(YOURSTUDENTID, "2_2"),index = None)
matches_1.head()

# 2.3
# WRITE YOUR CODE HERE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
merged_text = list(df_merged['text'].dropna(axis=0, how='any'))
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(merged_text)
# 2.3.2 Dimension Reduction (1.5 Marks)¶

# WRITE YOUR CODE HERE
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=5)
svd.fit(X).transform(X)