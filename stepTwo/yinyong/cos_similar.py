import numpy as np
import pandas as pd
import os

save_dir = "./xiangsidu"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def cosine_similarities(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0:
        return 0
    num = float(vector_a * vector_b.T)
    return num / denom


def build_(flag='1', sheet='引用'):
    if flag == '1':
        dataname = '数据2.xlsx'
        if sheet == '引用':
            data = pd.read_excel(dataname, sheet_name='459引用矩阵')
            del data['Author']
        else:
            data = pd.read_excel(dataname, sheet_name='459合作矩阵')
            del data['Author']

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
        df_save.to_excel("./xiangsidu/{}_{}".format(sheet, dataname), index=None)
    else:
        dataname = '数据.xlsx'
        if sheet == '引用':
            data = pd.read_excel(dataname, sheet_name='引用矩阵')
            del data['Author']
        else:
            data = pd.read_excel(dataname, sheet_name='合作矩阵')
            del data['Author']

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
        df_save.to_excel("./xiangsidu/{}_{}".format(sheet, dataname), index=None)


for f in ['1', '2']:
    for s in ['引用', '合作']:
        build_(flag=f, sheet=s)

print("1完成")
