import numpy as np
import pandas as pd

#
# def cosine_similarities(vector_a, vector_b):
#     vector_a = np.mat(vector_a)
#     vector_b = np.mat(vector_b)
#     denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
#     if denom == 0:
#         return 0
#     num = float(vector_a * vector_b.T)
#     # cos = num / denom
#     # sim = 0.5 + 0.5 * cos
#     # return sim
#     return num / denom
#
#
# def build_(dataname="yinyong.csv"):
#     data = pd.read_csv(dataname, sep=',')
#     print(data.head())
#     colmu = data.columns.tolist()
#     save = []
#     length = len(colmu)
#     for i in range(length):
#         for j in range(i, length):
#             try:
#                 x = data[colmu[i]]
#                 y = data[colmu[j]]
#                 res = cosine_similarities(x, y)
#                 rows = {'作者1': colmu[i], '作者2': colmu[j], 'simlar': res}
#                 save.append(rows)
#             except Exception as e:
#                 continue
#     df_save = pd.DataFrame(save)
#     df_save.to_csv("相似度_{}".format(dataname), index=None)


# for file in ['yinyong.csv', 'hezuo.csv', 'yinyong87.csv', 'hezuo87.csv']:
#     build(file)


# 作者1	作者2	合作次数	1-2单向引用次数	2-1单向引用次数	互引次数


yinyong = pd.read_csv("yingyong_87_erweibiao.csv")


def get_2to1_yingyong(x1, x2):
    cishu = 0
    for x, y in yinyong.iterrows():
        a1 = y['作者1']
        a2 = y['作者2']
        if a1 == x2 and a2 == x1:
            cishu = y['引用次数']
    return cishu


def build_one():
    hezuo = pd.read_csv("hezuo_87_erweibiao.csv")
    hezuo['2-1单向引用次数'] = hezuo.apply(lambda row: get_2to1_yingyong(row['作者1'], row['作者2']), axis=1)
    hezuo.to_csv('hezuo_87new_.csv', index=None)


def build_last(hezuo, yinyong):
    out_df = pd.DataFrame()
    out_df['作者1'] = hezuo['作者1']
    out_df['作者2'] = hezuo['作者2']
    out_df['合作次数'] = hezuo['合作次数']
    out_df['1-2单向引用次数'] = yinyong['引用次数']
    out_df['2-1单向引用次数'] = hezuo['2-1单向引用次数']
    out_df['互引次数'] = out_df.apply(lambda row: min(row['1-2单向引用次数'], row['2-1单向引用次数']), axis=1)
    out_df.to_csv("Shett1_.csv", index=None)


def build():
    h = pd.read_csv("hezuo_87new_.csv")
    yinyong = pd.read_csv("yingyong_87_erweibiao.csv")
    build_last(h, yinyong)


build_one()
print("第一阶段完成，开始第二阶段数据处理")
build()