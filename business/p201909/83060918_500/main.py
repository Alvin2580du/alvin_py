# 星期天晚上， 两个文件比对
import pandas as pd
from collections import OrderedDict
import difflib
import Levenshtein


def get_simm(str1, str2):
    # 1. difflib
    seq = difflib.SequenceMatcher(None, str1, str2)
    sim1 = seq.ratio()
    # 2.计算莱文斯坦比
    sim2 = Levenshtein.ratio(str1, str2)
    # 3.计算jaro距离sa764 ZzREsa
    sim3 = Levenshtein.jaro(str1, str2)
    # 4. Jaro–Winkler距离
    sim4 = Levenshtein.jaro_winkler(str1, str2)
    return (sim1 + sim2 + sim3 + sim4) / 4


Baa = pd.read_excel("Baa.xlsx")
FilmLA = pd.read_excel("FlimLA data.xlsx")
FilmLA_title = FilmLA['title'].values


def get_most_similar(inputs):
    scores = 0
    out = None
    num = 0
    ids = None
    for i in FilmLA_title:
        num += 1
        i2lower = str(i).lower()
        ratio = get_simm(i2lower, inputs)
        if ratio > scores:
            scores = ratio
            out = i2lower
            ids = num
    return scores, out, inputs, ids

save = []
num = 0
for x, y in Baa.iterrows():
    num += 1
    title = "{}={}={}".format(y['Title'], y['title1'], y['title2']).lower()
    title2set = [i for i in list(set(title.split('='))) if i != 'nan']

    if len(title2set) == 1:
        scores, out, inputs, ids = get_most_similar(title2set[0])
        rows = OrderedDict()
        rows['Baa line'] = num
        rows['FlimLA line'] = ids
        rows['Baa'] = inputs
        rows['FlimLA data'] = out
        rows['scores'] = scores
        rows['label'] = 1
        save.append(rows)
        print("{}, {}".format(len(title2set), rows))

    if len(title2set) == 2:
        scores1, out1, inputs1, ids1 = get_most_similar(title2set[0])
        scores2, out2, inputs2, ids2 = get_most_similar(title2set[1])
        if scores1 > scores2:
            rows = OrderedDict()
            rows['Baa line'] = num
            rows['FlimLA line'] = ids1
            rows['Baa'] = inputs1
            rows['FlimLA data'] = out1
            rows['scores'] = scores1
            rows['label'] = 2
            save.append(rows)
            print("{}, {}".format(len(title2set), rows))
        else:
            rows = OrderedDict()
            rows['Baa line'] = num
            rows['FlimLA line'] = ids2
            rows['Baa'] = inputs2
            rows['FlimLA data'] = out2
            rows['scores'] = scores2
            rows['label'] = 2
            save.append(rows)
            print("{}, {}".format(len(title2set), rows))

    if len(title2set) == 3:
        scores1, out1, inputs1, ids1 = get_most_similar(title2set[0])
        scores2, out2, inputs2, ids2 = get_most_similar(title2set[1])
        scores3, out3, inputs3, ids3 = get_most_similar(title2set[2])
        if scores1 > scores2:
            if scores1 > scores3:
                rows = OrderedDict()
                rows['Baa line'] = num
                rows['FlimLA line'] = ids1
                rows['Baa'] = inputs1
                rows['FlimLA data'] = out1
                rows['scores'] = scores1
                rows['label'] = 3
                save.append(rows)
                print("{}, {}".format(len(title2set), rows))
            else:
                rows = OrderedDict()
                rows['Baa line'] = num
                rows['FlimLA line'] = ids3
                rows['Baa'] = inputs3
                rows['FlimLA data'] = out3
                rows['scores'] = scores3
                rows['label'] = 3
                save.append(rows)
                print("{}, {}".format(len(title2set), rows))
        else:
            if scores2 > scores3:
                rows = OrderedDict()
                rows['Baa line'] = num
                rows['FlimLA line'] = ids2
                rows['Baa'] = inputs2
                rows['FlimLA data'] = out2
                rows['scores'] = scores2
                rows['label'] = 3
                save.append(rows)
                print("{}, {}".format(len(title2set), rows))
            else:
                rows = OrderedDict()
                rows['Baa line'] = num
                rows['FlimLA line'] = ids3
                rows['Baa'] = inputs3
                rows['FlimLA data'] = out3
                rows['scores'] = scores3
                rows['label'] = 3
                save.append(rows)
                print("{}, {}".format(len(title2set), rows))

df = pd.DataFrame(save)
df.to_csv("结果文件.csv", index=None)
print(df.shape[0])



