import numpy as np
import pandas as pd


def cosine_similarities(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0:
        return 0
    num = float(vector_a * vector_b.T)
    # cos = num / denom
    # sim = 0.5 + 0.5 * cos
    # return sim
    return num / denom


def build(dataname="yinyong.csv"):
    data = pd.read_csv(dataname, sep=',')
    print(data.head())
    colmu = data.columns.tolist()
    save = []
    length = len(colmu)
    for i in range(length):
        for j in range(i, length):
            try:
                x = data[colmu[i]]
                y = data[colmu[j]]
                res = cosine_similarities(x, y)
                rows = {'作者1': colmu[i], '作者2': colmu[j], 'simlar': res}
                save.append(rows)
            except Exception as e:
                continue
    df_save = pd.DataFrame(save)
    df_save.to_csv("相似度_{}".format(dataname), index=None)

for file in ['yinyong.csv', 'hezuo.csv', 'yinyong87.csv', 'hezuo87.csv']:
    build(file)



