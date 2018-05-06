from antcolony.utils.utils import replace_symbol
import pandas as pd
import jieba
from collections import Counter
all = []
data = pd.read_csv("./datasets/data00418.csv", usecols=['标题／微博内容']).values
for one in data:
    msg = one[0]
    msgcut = jieba.lcut(replace_symbol(msg))
    for w in msgcut:
        all.append(w)


for x, y in Counter(all).most_common(1000):
    print(x, y)