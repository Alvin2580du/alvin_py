import pandas as pd
import jieba
from collections import Counter

data = pd.read_excel("自然语言处理职位表.xlsx")
print(data.shape)
alls = []
for one in data.iterrows():
    zhize = one[1]['职责'].split()
    for x in zhize:
        xc = jieba.lcut(x)
        for i in xc:
            alls.append(i)


for x, y in Counter(alls).most_common(1000):
    print(y, x)

