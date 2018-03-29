import pandas as pd
from datetime import datetime, timedelta
import os
from tqdm import tqdm, trange
from sklearn.linear_model import LinearRegression
import numpy as np


def getday(y=2017, m=8, d=15, n=0):
    the_date = datetime(y, m, d)
    result_date = the_date + timedelta(days=n)
    d = result_date.strftime('%Y/%m/%d')
    return d


def sub356(inputs):
    return (inputs - timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')


def sub90(inputs):
    return (inputs - timedelta(days=90)).strftime('%Y-%m-%d %H:%M:%S')


def convert(inputs):
    return datetime.strptime(inputs.replace("\n", ""), '%Y/%m/%d').strftime("%Y-%m-%d %H:%M:%S")


def timestamp2list(timestamps):
    datelist = timestamps.to_pydatetime()
    datelists = []
    for d in datelist:
        d = d.strftime("%Y-%m-%d %H:%M:%S")
        datelists.append(d)
    return datelists


def str2timedate(inputs):
    return datetime.strptime(inputs, '%Y-%m-%d %H:%M:%S')


def ind_new():
    data = pd.read_csv("./data_dir/train.csv")
    inds = {}
    data['日期'] = data['日期'].apply(convert)
    for index, row in data.iterrows():
        i = str2timedate(row['日期'])
        name = row['股票简称']
        c = row['行业代码']
        t1sub365 = sub356(i)
        t1sub90 = sub90(i)
        datelist = timestamp2list(pd.date_range(start=t1sub365, end=t1sub90, freq='D'))
        company = data[data['日期'].isin(datelist) & data['行业代码'].isin([c])]
        means = company.mean()
        inds[name] = means
    df = pd.DataFrame(inds)
    if not os.path.exists('./data_dir'):
        os.makedirs("./data_dir")
    df.to_csv("./data_dir/ind.csv", index=None, encoding='utf-8')
    print(df.shape)


def rec_new():
    recs = {}
    data = pd.read_csv("./data_dir/train.csv", encoding='utf-8')
    for index, row in data.iterrows():
        name = row['股票简称']
        i = str2timedate(row['日期'])
        t1sub90 = sub90(i)
        datelist = timestamp2list(pd.date_range(start=t1sub90, end=i, freq='D'))
        company = data[data['日期'].apply(convert).isin(datelist)]
        del company['股票简称']
        del company['日期']
        del company['行业代码']
        means = company.mean().values
        recs[name] = means
    df = pd.DataFrame(recs)
    if not os.path.exists('./data_dir'):
        os.makedirs("./data_dir")
    df.to_csv("./data_dir/rec.csv", index=None, encoding='utf-8')
    print(df.shape)


def buildmodel_new():
    rec = pd.read_csv("./data_dir/rec.csv")
    ind = pd.read_csv("./data_dir/ind.csv")
    data = pd.read_csv("./data_dir/train.csv")
    del data['行业代码']
    del data['日期']
    savedf = []
    print(rec.shape, ind.shape, data.shape)

    rec = rec.fillna(0)
    ind = ind.fillna(0)
    data = data.fillna(0)
    for index, row in data.iterrows():
        name = row['股票简称']
        y_train = row.values[1:]
        x1 = rec[name]
        x2 = ind[name]
        print(y_train.shape, x1.shape, x2.shape)
        X_train = pd.DataFrame([x1, x2]).T.values
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        u = linreg.intercept_
        a1, a2 = linreg.coef_[0], linreg.coef_[1]
        save = {'a1': a1, 'a2': a2, 'u': u, 'name': name}
        savedf.append(save)

    df = pd.DataFrame(savedf)
    df.to_csv("results.csv", encoding='utf-8', index=None, sep=",")


if __name__ == '__main__':
    # 这里依次修改 下面的每种method，就可以出结果了。

    method = 'buildmodel_new'

    if method == 'rec_new':
        rec_new()

    if method == 'buildmodel_new':
        buildmodel_new()

    if method == 'ind_new':
        ind_new()
