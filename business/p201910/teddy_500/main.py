import pandas as pd
import Levenshtein
from collections import OrderedDict


def get_simm(str1, str2):
    # 1.计算莱文斯坦比
    sim2 = Levenshtein.ratio(str1, str2)
    # 2.计算jaro距离
    sim3 = Levenshtein.jaro(str1, str2)
    return (sim2 + sim3) / 2


"""
1、用“baa.xlsx”里的title1, title2, and title3, ProductionTitle_June2016Master_1, 
ProductionTitle1, ProductionTitle2, ProductionTitle3, ProductionTitle4
以及我新发给你的title.basics.tsv里的 primarytitle和 orginal title （先matchKS_IMDB）
去match the “ProductionTitle” in FlimLA data.xlsx.

2、 是baa里面的ProductionCompany1 ProductionCompany2 ProductionCompany3 ProductionCompany4和
FlimLA data.xlsx里的production company计算相似度

3、 年份 判断 1 0
"""

data_baa = pd.read_excel("./data/Baa.xlsx")
flimla = pd.read_csv("./data/filmla.csv", sep=',')
title_basics = pd.read_csv("./data/title.basics.tsv", sep='\t', low_memory=False)
FilmLA_title = flimla['ProductionTitle'].values
FilmLA_CompanyName = flimla['CompanyName'].values


def get_most_similar(inputs, base):
    scores = 0
    out = None
    num = 0
    ids = None
    for i in base:
        num += 1
        i2lower = str(i).lower()
        ratio = get_simm(i2lower, inputs)
        if ratio > scores:
            scores = ratio
            out = i2lower
            ids = num
    return scores, out, inputs, ids


# KS_IMDB
def get_primaryTitle(ids):
    try:
        res1 = title_basics[title_basics['tconst'].isin([ids])]['primaryTitle'].values[0]
        return res1
    except:
        return None


def get_originalTitle(ids):
    try:
        res2 = title_basics[title_basics['tconst'].isin([ids])]['originalTitle'].values[0]
        return res2
    except:
        return None


def get_year(inputs):
    inputs = str(inputs)
    if "A" in inputs:
        return 2009
    elif "B" in inputs:
        return 2011
    elif "C" in inputs:
        return 2012
    elif "D" in inputs:
        return 2013
    elif "E" in inputs:
        return 2014
    elif "F" in inputs:
        return 2015
    elif "G" in inputs:
        return 2016
    else:
        return inputs


num = 0
save = []
for x, y in data_baa.iterrows():
    num += 1
    KS_IMDB = str(y['KS_IMDB']).replace("???", '')
    primaryTitle = get_primaryTitle(KS_IMDB)
    originalTitle = get_originalTitle(KS_IMDB)
    title = "{}={}={}={}={}={}={}={}={}={}".format(primaryTitle, originalTitle, y['Title'], y['title1'],
                                                   y['title2'],
                                                   y['ProductionTitle_June2016Master_1'],
                                                   y['ProductionTitle1'], y['ProductionTitle2'],
                                                   y['ProductionTitle3'],
                                                   y['ProductionTitle4']).lower()
    title2set = [i for i in list(set(title.split('='))) if i != 'nan']
    if not title2set:
        continue
    try:
        scores_ = {}
        k = 0
        for one in title2set:
            if str(one) == 'none':
                continue
            scores, out, inputs, ids = get_most_similar(one, base=FilmLA_title)
            if scores > k:
                k = scores
                scores_['scores'] = scores
                scores_['out'] = out
                scores_['inputs'] = inputs
                scores_['ids'] = ids

        rows = OrderedDict()
        rows['Baa line'] = num
        rows['FlimLA line'] = scores_['ids']
        rows['Baa title'] = scores_['inputs']
        rows['FlimLA title'] = scores_['out']
        rows['title scores'] = scores_['scores']
    except:
        continue

    if rows['title scores'] < 0.7:
        continue

    com_title = "{}={}={}={}".format(y['ProductionCompany1'], y['ProductionCompany2'],
                                     y['ProductionCompany3'], y['ProductionCompany4']).lower()
    com_title2set = [i for i in list(set(com_title.split('='))) if i != 'nan']
    if not com_title2set:
        rows['Baa Company'] = ''
        rows['FlimLA Company'] = ''
        rows['Company scores'] = ''
    else:
        com_scores_ = {}
        k = 0
        for one in com_title2set:
            scores, out, inputs, ids = get_most_similar(one, base=FilmLA_title)
            if scores > k:
                k = scores
                com_scores_['scores'] = scores
                com_scores_['out'] = out
                com_scores_['inputs'] = inputs
                com_scores_['ids'] = ids

        rows['Baa Company'] = com_scores_['inputs']
        rows['FlimLA Company'] = com_scores_['out']
        rows['Company scores'] = com_scores_['scores']

    try:
        # Queue_reapply1
        Queue_reapply1 = int(get_year(y['Queue_reapply1']))
        FiscalYear = int(y['FiscalYear'].split('/')[0])
        Activity = int(flimla.iloc[rows['FlimLA line']]['Activity'].split("/")[-1])

        if Queue_reapply1 <= Activity:
            Queue_reapply1_scores = '1'
        else:
            Queue_reapply1_scores = '0'

        if FiscalYear <= Activity:
            FiscalYear_scores = '1'
        else:
            FiscalYear_scores = '0'

        rows['Activity'] = Activity
        rows['FiscalYear'] = FiscalYear
        rows['FiscalYear scores'] = FiscalYear_scores
        rows['Queue_reapply1'] = Queue_reapply1
        rows['Queue_reapply1 scores'] = Queue_reapply1_scores
    except:
        rows['Activity'] = ''
        rows['FiscalYear'] = ''
        rows['FiscalYear scores'] = ''
        rows['Queue_reapply1'] = ''
        rows['Queue_reapply1 scores'] = ''

    save.append(rows)
    print(rows)


df = pd.DataFrame(save)
df.to_csv('结果.csv', index=None)

