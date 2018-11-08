# 附件一：构造自变量

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import pickle

# ！！！简称：eh:年中个股总权益，也就是size；b/m：年末账面价值/市值；rim:个股收益率

pl = pd.read_pickle('panel1.pickle')

rim = pd.read_pickle('rim.pickle')
# 把个股每年的年中总权益(eh)，以及年末权益市值比(b/m)，处理好转换为panel（一种三维数据格式），有items,major axis,
# minor axis三个维度，items是eh和b/m，major维度是年份，minor是个股代码
df = pd.DataFrame(columns=['smb', 'hml'])  # 创建一个空的df，用来保存分组计算出的每组的个股收益


def get_sp(sp):
    sp1 = sp.drop('b/m', axis=1)
    return sp1


def six_equal(year):  # 六等分function
    nf = pl.major_xs(year)  # df
    nf.dropna(inplace=True, axis=0)
    bm = np.array(nf['b/m'])
    sz = np.array(nf['eh'])
    small = nf[nf['eh'] < np.percentile(sz, 50)]  # dataframe, 分位数percentile
    big = nf[nf['eh'] >= np.percentile(sz, 50)]
    sg = small[small['b/m'] < np.percentile(np.array(small['b/m']), 30)]
    sn = small[np.logical_and((small['b/m'] >= np.percentile(np.array(small['b/m']), 30)),
                              (small['b/m'] < np.percentile(np.array(small['b/m']), 70)))]
    sv = small[small['b/m'] >= np.percentile(np.array(small['b/m']), 70)]
    bg = big[big['b/m'] < np.percentile(np.array(big['b/m']), 30)]
    bn = big[np.logical_and((big['b/m'] >= np.percentile(np.array(big['b/m']), 30)),
                            (big['b/m'] < np.percentile(np.array(big['b/m']), 70)))]
    bv = big[big['b/m'] >= np.percentile(np.array(big['b/m']), 70)]
    arr0 = [sg, sn, sv, bg, bn, bv]
    arr1 = list(map(get_sp, arr0))
    result = pd.concat(arr1, keys=['sg', 'sn', 'sv', 'bg', 'bn', 'bv'])
    return result


def get_r(fraction, rim0, ns):
    fr = ns.loc[fraction]
    fr = pd.merge(fr, rim0.to_frame('r'), how='left', left_index=True, right_index=True)
    r = ((fr['eh'] * fr['r']).sum()) / (fr['eh'].sum())
    return r


def get_rlist(rim0):
    arr = ['sg', 'sn', 'sv', 'bg', 'bn', 'bv']
    rim0 = [rim0] * 6
    rl = list(map(get_r, arr, rim0))  # 求加权收益率
    smb = (1 / 3) * (rl[0] + rl[1] + rl[2]) - (1 / 3) * (rl[3] + rl[4] + rl[5])
    hml = (1 / 2) * (rl[2] + rl[5]) - (1 / 2) * (rl[0] + rl[3])
    return [smb, hml]


def get_rmth(year, month):
    rim0 = rim.loc[str(year) + '-' + str(month).zfill(2)]  # 加零
    return rim0


def get_ryear(year):
    for i in range(7, 13):
        rim0 = get_rmth(year, i)
        arr0 = get_rlist(rim0)
        df.loc[str(year) + '-' + str(i).zfill(2)] = arr0  # loc用于创建
    for i in range(1, 7):
        rim0 = get_rmth(year + 1, i)
        arr0 = get_rlist(rim0)
        df.loc[str(year + 1) + '-' + str(i).zfill(2)] = arr0

for j in range(1996, 2016):
    ns = six_equal(j)  # 六等分
    get_ryear(j)


# 附件二：构造资产组合，用作因变量

# rim=pd.read_pickle('rim.pickle')
# rim=rim['rim']


def get_quintile(year):
    mt = pl.major_xs(year)
    mt = mt.dropna()
    qt = mt['eh'].quantile([.2, .4, .6, .8])
    gp1 = mt[mt['eh'] < qt.iloc[0]]
    gp2 = mt[(mt['eh'] >= qt.iloc[0]) & (mt['eh'] < qt.iloc[1])]
    gp3 = mt[(mt['eh'] >= qt.iloc[1]) & (mt['eh'] < qt.iloc[2])]
    gp4 = mt[(mt['eh'] >= qt.iloc[2]) & (mt['eh'] < qt.iloc[3])]
    gp5 = mt[mt['eh'] >= qt.iloc[3]]
    gp = [gp1, gp2, gp3, gp4, gp5]
    lst = []
    for g in gp:
        qtg = g['b/m'].quantile([.2, .4, .6, .8])
        lst.append(g[g['b/m'] < qtg.iloc[0]])
        lst.append(g[(g['b/m'] >= qtg.iloc[0]) & (g['b/m'] < qtg.iloc[1])])
        lst.append(g[(g['b/m'] >= qtg.iloc[1]) & (g['b/m'] < qtg.iloc[2])])
        lst.append(g[(g['b/m'] >= qtg.iloc[2]) & (g['b/m'] < qtg.iloc[3])])
        lst.append(g[g['b/m'] >= qtg.iloc[3]])

    k = []
    for i in range(1, 6):
        for j in range(1, 6):
            k.append('gp' + str(i) + str(j))

    result = pd.concat(lst, keys=k)
    result.drop('b/m', axis=1, inplace=True)
    return result


def build_last():
    k = []
    for i in range(1, 6):
        for j in range(1, 6):
            k.append('gp' + str(i) + str(j))

    df = pd.DataFrame(columns=k)

    for h in range(1996, 2016):
        ns = get_quintile(h)
        get_ryear(h)

    nrr = pd.read_pickle('r.pickle')
    for joe in k:
        df[joe] = df[joe] - (pd.to_numeric(nrr['rmm']) / 100)

    df.to_pickle('groups.pickle')

    df = pd.DataFrame(
        columns=['r^2', 'itc', 't(itc)', 'p(itc)', 'ermm', 't(ermm)', 'p(ermm)', 'smb', 't(smb)', 'p(smb)', 'hml',
                 't(hml)', 'p(smb)'])
    facs = pd.read_pickle('facs.pickle')
    gp = pd.read_pickle('groups.pickle')

    k = []
    for i in range(1, 6):
        for j in range(1, 6):
            k.append('gp' + str(i) + str(j))
    for h in range(25):
        model = pd.merge(gp[k[h]].to_frame(), facs, how='outer', left_index=True, right_index=True)
        result = sm.ols(formula=k[h] + ' ~ ermm + smb + hml', data=model).fit()
        pv = list(result.pvalues)  # 导出p值
        tv = list(result.tvalues)  # 导出p值
        co = list(result.params)  # 导出系数
        df.loc[k[h]] = [result.rsquared, co[0], tv[0], pv[0], co[1], tv[1], pv[1], co[2], tv[2], pv[2], co[2], tv[2],
                        pv[2]]
    df.to_pickle('result.pickle')
    df.to_excel('result.xls')
    # print(result.summary())
