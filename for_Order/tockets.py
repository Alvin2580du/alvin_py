"""
需求：

    1、筛选并提取该公司日期之前，90天之内的所有公司数据，将筛选后公司数值数据列求平均，得到一个新向量，命名rec；例如：14行，
    贵人鸟，日期2014-01-14，则选取2013-10-14至2014-01-13间的公司，则筛选出2-13行所有公司，将简称之后的所有数据列求平均，
    得到一个同等长度的新向量，命名rec13，名字主要用于匹配该公司，用于下部计算。

    2、筛选公司日期之前，365天至91天之内的所有同行业公司数据，筛选后数值同理得到一个新向量，命名ind；例如100行，高能环境，
    日期2014-12-18，行业代码N，则取2013-12-18至2014-9-17间所有公司，再筛选同行业代码公司，筛选后同上方式得到新向量，命名ind。

    3、经过上述处理，每个公司样本将分别得出两个新向量rec、ind加上原向量记为一组，有478组，共计1434个向量。

    4、令Y=原始公司数据，逐个将公司Y、rec、ind代入Y=a1*rec+a2*ind+u，回归得出a1、a2、u的值，无需回归其他数值。

    最终将得到478组a1、a2、u的值，这是最终需要的结果。
"""
import pandas as pd
from datetime import datetime, timedelta
import os
from tqdm import tqdm, trange
from sklearn.linear_model import LinearRegression
import numpy as np
from glob import glob

"""
1. 删除无用列，得到(478, 2685)维大的矩阵,文件名为train.csv。
2. 把每个公司90天内的相关公司的数据，分组到不同的文件中，然后对每个文件计算平均值
3.按行业分组，然后对每个行业内的365天到91天内的数据统计，计算平均值
4. 建立回归模型，求出系数。

"""


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


def groupbycompany():
    data = pd.read_csv("./data_dir/train.csv", encoding='utf-8')
    datacopy = data.copy()
    datacopy['date'] = pd.to_datetime(data['日期'])
    df = datacopy.set_index('date')
    k = 0
    for i in tqdm(df.index):
        select_index = pd.date_range(end=i, periods=90, freq='D')
        company = df.loc[select_index, :]
        NONE_VIN = company[~company["股票简称"].isnull()]
        NONE_VIN = NONE_VIN.drop(NONE_VIN.index[-1], axis=0)
        if not os.path.exists("./company"):
            os.makedirs("./company")
        NONE_VIN.to_csv("./company/{}.csv".format(k), encoding='utf-8')
        k += 1
    print(k)


def rec_compute():
    recs = {}
    k = 0
    for i in trange(478):
        file = os.path.join("./company", "{}.csv".format(i))
        data = pd.read_csv(file)
        name = data['股票简称']
        del data['date']
        del data['行业代码']
        del data['日期']
        del data['股票简称']
        meanrec = data.mean().values
        recs[name] = meanrec
        k += 1
    df = pd.DataFrame(recs)
    df = df.fillna(0)
    df.to_csv("./data_dir/rec.csv", index=None, encoding='utf-8')
    print(k, df.shape)


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


def groupbyhangye():
    data = pd.read_csv("./data_dir/train.csv", encoding='utf-8')
    datagroup = data.groupby(by='行业代码')
    for i, j in datagroup:
        if not os.path.exists("./hangye"):
            os.makedirs("./hangye")
        j.to_csv("./hangye/{}.csv".format(i), index=None, encoding='utf-8')


def timestamp2list(timestamps):
    datelist = timestamps.to_pydatetime()
    datelists = []
    for d in datelist:
        d = d.strftime("%Y-%m-%d %H:%M:%S")
        datelists.append(d)
    return datelists


def str2timedate(inputs):
    return datetime.strptime(inputs, '%Y-%m-%d %H:%M:%S')


def getind():
    inds = {}
    k = 0
    for file in tqdm(os.listdir('./hangye')):
        filename = os.path.join("./hangye", file)
        data = pd.read_csv(filename)
        datacopy = data.copy()
        datacopy['日期'] = data['日期'].apply(convert)
        limit = 3

        for index, row in datacopy.iterrows():
            name = row['股票简称']
            i = str2timedate(row['日期'])
            t1sub365 = sub356(i)
            t1sub90 = sub90(i)
            datelist = timestamp2list(pd.date_range(start=t1sub365, end=t1sub90, freq='D'))
            company = datacopy[datacopy['日期'].isin(datelist)]
            k += 1
            del company['股票简称']
            del company['日期']
            del company['行业代码']
            means = company.mean().values
            inds[name] = means
            if k > limit:
                break
    df = pd.DataFrame(inds)
    if not os.path.exists('./data_dir'):
        os.makedirs("./data_dir")
    df.to_csv("./data_dir/ind.csv", index=None, encoding='utf-8')
    print(df.shape)


def buildmodel():
    rec = pd.read_csv("./data_dir/rec.csv")
    ind = pd.read_csv("./data_dir/ind.csv")
    print(rec.shape, ind.shape)
    data = pd.read_csv("./data_dir/train.csv")
    del data['行业代码']
    del data['日期']
    gupiao = data['股票简称']
    del data['股票简称']
    length = len(data)
    print("length:{}".format(length))
    rec = rec.fillna(0)
    ind = ind.fillna(0)
    data = data.fillna(0)

    savedf = []
    for i in range(length):
        y_train = data.loc[i, :]
        recvalue = rec.loc[i, :]
        indvalue = ind.loc[i, :]
        X_train = np.array([recvalue, indvalue]).reshape((2682, 2))
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        u = linreg.intercept_
        a1, a2 = linreg.coef_[0], linreg.coef_[1]
        save = {'a1': a1, 'a2': a2, 'u': u}
        savedf.append(save)

    df = pd.DataFrame(savedf)
    df['股票简称'] = gupiao
    df.to_csv("results.csv", encoding='utf-8', index=None, sep=",")


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
        print(np.sum(means))
    df = pd.DataFrame(inds)
    if not os.path.exists('./data_dir'):
        os.makedirs("./data_dir")
    df.to_csv("./data_dir/ind.csv", index=None, encoding='utf-8')
    print(df.shape)

if __name__ == '__main__':
    # 这里依次修改 下面的每种method，就可以出结果了。

    method = 'ind_new'  # 0.groupbyhangye  1. getind, 2. rec_new, 3. buildmodel_new
    # if method == 'groupbycompany':
    #     groupbycompany()
    #
    # if method == 'rec_compute':
    #     rec_compute()

    if method == 'groupbyhangye':
        groupbyhangye()

    if method == 'getind':
        getind()  # (478, 2682)

    # if method == 'buildmodel':
    #     buildmodel()

    if method == 'rec_new':
        rec_new()

    if method == 'buildmodel_new':
        buildmodel_new()
    if method == 'ind_new':
        ind_new()