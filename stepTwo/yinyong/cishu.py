import pandas as pd

import os

save_dir = "./cishu"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

flag = '2'  # 修改这里。1,或者2

if flag == '1':
    yinyong = pd.read_excel("数据.xlsx", sheet_name='引用二维表')
    hezuo = pd.read_excel("数据.xlsx", sheet_name='合作二维表')
    print(yinyong.shape, hezuo.shape)
else:
    yinyong = pd.read_excel('数据2.xlsx', sheet_name="459引用二维表")
    hezuo = pd.read_excel('数据2.xlsx', sheet_name="459合作二维表")
    print(yinyong.shape, hezuo.shape)

rows1 = {}
rows2 = {}
for x, y in yinyong.iterrows():
    a1 = y['作者1']
    a2 = y['作者2']
    cishu = y['引用次数']
    rows1[a1] = cishu
    rows2[a2] = cishu


def get_2to1_yingyong(x1, x2):
    if x1 in rows1.keys() and x2 in rows2.keys():
        res = rows1[x2]
        return res
    return 0

# def get_2to1_yingyong(x1, x2):
#     cishu = 0
#     for x, y in yinyong.iterrows():
#         a1 = y['作者1']
#         a2 = y['作者2']
#         if a1 == x2 and a2 == x1:
#             cishu = y['引用次数']
#     print(cishu)
#     return cishu


def build_one():
    hezuo['2-1单向引用次数'] = hezuo.apply(lambda row: get_2to1_yingyong(row['作者1'], row['作者2']), axis=1)
    hezuo.to_csv('./cishu/hezuo_87new_.csv', index=None)
    print(hezuo.shape)


def build_last(hezuo, yinyong):
    out_df = pd.DataFrame()
    out_df['作者1'] = hezuo['作者1']
    out_df['作者2'] = hezuo['作者2']
    out_df['合作次数'] = hezuo['合作次数']
    out_df['1-2单向引用次数'] = yinyong['引用次数']
    out_df['2-1单向引用次数'] = hezuo['2-1单向引用次数']
    out_df['互引次数'] = out_df.apply(lambda row: min(row['1-2单向引用次数'], row['2-1单向引用次数']), axis=1)
    out_df.to_csv("./cishu/Shett1_.csv", index=None)


def build():
    h = pd.read_csv("./cishu/hezuo_87new_.csv")
    build_last(h, yinyong)


build_one()
print("第一阶段完成，开始第二阶段数据处理")
build()
