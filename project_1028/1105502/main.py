import pandas as pd
import os


def get_month(inputs):
    s = '06-30'
    if s in inputs:
        return inputs
    else:
        return None


def del_col(data):
    for x in data.columns:
        if '证券名称' in x:
            continue
        if '证券代码' in x:
            continue
        res = get_month(x)
        if res:
            continue
        else:
            del data[x]
    return data


# 总市值，账面市值比(净资产/总市值），市盈率倒数
def zm_shizhibi(x1, x2):
    return x1 / x2


def data_preprocess():
    zichan = pd.read_excel("A股净资产.xlsx").dropna()
    shiyinlv = pd.read_excel("全部A股半年市盈率ttm06-16.xls").dropna()
    shizhi = pd.read_excel("全部A股月总市值06-16.xls").dropna()
    shouyilv = pd.read_excel("全部A股月收益率06-16.xls").dropna()
    print(zichan.shape, shiyinlv.shape, shizhi.shape, shouyilv.shape)

    zichan_stocks = list(zichan['证券名称'].values)
    shiyinlv_stocks = list(shiyinlv['证券名称'].values)
    chanzhi_stocks = list(shizhi['证券名称'].values)
    shouyilv_stocks = list(shouyilv['证券名称'].values)

    name1 = [i for i in zichan_stocks if i in shiyinlv_stocks]
    name2 = [i for i in chanzhi_stocks if i in shouyilv_stocks]
    name = [i for i in name1 if i in name2]

    zichan_save = del_col(zichan)
    shiyinlv_save = del_col(shiyinlv)
    shizhi_save = del_col(shizhi)
    shouyilv_save = del_col(shouyilv)

    zichan_save = zichan_save[zichan_save['证券名称'].isin(name)]
    shiyinlv_save = shiyinlv_save[shiyinlv_save['证券名称'].isin(name)]
    shizhi_save = shizhi_save[shizhi_save['证券名称'].isin(name)]
    shouyilv_save = shouyilv_save[shouyilv_save['证券名称'].isin(name)]
    if not os.path.exists("./results"):
        os.makedirs('./results')

    zichan_save.columns = ['证券代码', '证券名称'] + ['zichan_20{0:02d}'.format(i) for i in range(5, 19)]
    shiyinlv_save.columns = ['证券代码', '证券名称'] + ['shiyinlv_20{0:02d}'.format(i) for i in range(6, 17)]
    shizhi_save.columns = ['证券代码', '证券名称'] + ['shizhi_20{0:02d}'.format(i) for i in range(6, 17)]
    shouyilv_save.columns = ['证券代码', '证券名称'] + ['shouyilv_20{0:02d}'.format(i) for i in range(6, 17)]

    print(zichan_save.shape, shiyinlv_save.shape, shizhi_save.shape, shouyilv_save.shape)
    df1 = pd.merge(left=zichan_save, right=shiyinlv_save, how='left', on=['证券代码', '证券名称'])
    df2 = pd.merge(left=shizhi_save, right=shouyilv_save, how='left', on=['证券代码', '证券名称'])
    df = pd.merge(left=df1, right=df2, how='left', on=['证券代码', '证券名称'])
    df['zm_shizhibi_2006'] = df.apply(lambda row: zm_shizhibi(row['zichan_2006'], row['shizhi_2006']), axis=1)
    df['zm_shizhibi_2007'] = df.apply(lambda row: zm_shizhibi(row['zichan_2007'], row['shizhi_2007']), axis=1)
    df['zm_shizhibi_2008'] = df.apply(lambda row: zm_shizhibi(row['zichan_2008'], row['shizhi_2008']), axis=1)
    df['zm_shizhibi_2009'] = df.apply(lambda row: zm_shizhibi(row['zichan_2009'], row['shizhi_2009']), axis=1)
    df['zm_shizhibi_2010'] = df.apply(lambda row: zm_shizhibi(row['zichan_2010'], row['shizhi_2010']), axis=1)
    df['zm_shizhibi_2011'] = df.apply(lambda row: zm_shizhibi(row['zichan_2011'], row['shizhi_2011']), axis=1)
    df['zm_shizhibi_2012'] = df.apply(lambda row: zm_shizhibi(row['zichan_2012'], row['shizhi_2012']), axis=1)
    df['zm_shizhibi_2013'] = df.apply(lambda row: zm_shizhibi(row['zichan_2013'], row['shizhi_2013']), axis=1)
    df['zm_shizhibi_2014'] = df.apply(lambda row: zm_shizhibi(row['zichan_2014'], row['shizhi_2014']), axis=1)
    df['zm_shizhibi_2015'] = df.apply(lambda row: zm_shizhibi(row['zichan_2015'], row['shizhi_2015']), axis=1)
    df['zm_shizhibi_2016'] = df.apply(lambda row: zm_shizhibi(row['zichan_2016'], row['shizhi_2016']), axis=1)

    del df['zichan_2005']
    del df['zichan_2017']
    del df['zichan_2018']

    for i in range(6, 17):
        del df['zichan_20{0:02d}'.format(i)]
    df.to_excel("数据.xlsx", index=None)
    print(df.shape)


def group_data(data):
    x1 = data[:129].index.values
    x2 = data[129:129 * 2].index.values
    x3 = data[129 * 2:129 * 3].index.values
    x4 = data[129 * 3:129 * 4].index.values
    x5 = data[129 * 4:129 * 5].index.values
    x6 = data[129 * 5:129 * 6].index.values
    x7 = data[129 * 6:129 * 7].index.values
    x8 = data[129 * 7:129 * 8].index.values
    x9 = data[129 * 8:129 * 9].index.values
    x10 = data[129 * 9:].index.values
    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10


def get_year(col):
    return col.split("_")[-1]


data = pd.read_excel("数据.xlsx", index_col='证券代码').head(1290)


def get_totoal_shouyilv(x, col):
    total = 0

    for i in x:
        chanzhi = data['shizhi_{}'.format(get_year(col))].loc[i]
        total += chanzhi

    out = 0
    for i in x:
        chanzhi = data['shizhi_{}'.format(get_year(col))].loc[i]
        shouyi = data['shouyilv_{}'.format(get_year(col))].loc[i]
        out += (chanzhi / total) * shouyi
    return out


def build_last():
    fw = open("results.txt", 'w', encoding='utf-8')
    for col in data.columns:
        if '证券名称' in col:
            continue
        res = data[col].sort_values(ascending=False)
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = group_data(res)
        x1_total = get_totoal_shouyilv(x1, col)
        x2_total = get_totoal_shouyilv(x2, col)
        x3_total = get_totoal_shouyilv(x3, col)
        x4_total = get_totoal_shouyilv(x4, col)
        x5_total = get_totoal_shouyilv(x5, col)
        x6_total = get_totoal_shouyilv(x6, col)
        x7_total = get_totoal_shouyilv(x7, col)
        x8_total = get_totoal_shouyilv(x8, col)
        x9_total = get_totoal_shouyilv(x9, col)
        x10_total = get_totoal_shouyilv(x10, col)
        save = "{},{},{},{},{},{},{},{},{},{},{}".format(col, x1_total, x2_total, x3_total, x4_total, x5_total,
                                                         x6_total,
                                                         x7_total, x8_total, x9_total, x10_total)
        print(save)
        fw.writelines(save + "\n")


for method in ['data_preprocess', 'build_last']:
    if method == 'data_preprocess':
        data_preprocess()

    print("= " * 20)
    if method == 'build_last':
        build_last()
