import pandas as pd
import os

nro = pd.read_csv("NRO.csv")
pt_id = nro['pt_id'].values.tolist()


def xiaoshu(inputs):
    return int(inputs)


def get_root(inputs):
    return inputs[:5]


if not os.path.exists("./data"):
    os.makedirs("./data")
nro_pm = pd.read_csv("NRO-PM(7,9,10).csv", usecols=['CB_ID', 'CB_ND1', 'CB_ND2', 'CB_CAPAFO', 'CB_LONG'])
print(nro_pm.head())
nro_pm.insert(0, 'NRO', nro_pm['CB_ID'].apply(get_root))
nro_pm.insert(1, '线缆名称', nro_pm['CB_ID'])
nro_pm.insert(2, '线缆规格', nro_pm['CB_CAPAFO'])
nro_pm.insert(3, '线缆长度', nro_pm['CB_LONG'].apply(xiaoshu))
"""
CB_ID ：线缆名称
CB_ND1：这条线缆的头连接的东西的名称
CB_ND2：这条线缆的尾连接的东西的名称
CB_CAPAFO：这条线缆的规格
CB_LONG：这条线缆的长度
"""
del nro_pm['CB_ID']
del nro_pm['CB_CAPAFO']
del nro_pm['CB_LONG']
only_ones = []
save = []
for x, y in nro_pm.groupby(by='NRO'):
    only_ones.append(x)
    data_ = y.sort_values(by='CB_ND1')
    for x1, data_sort in data_.iterrows():
        rows = {}
        rows['NRO'] = data_sort['NRO']
        rows['线缆名称'] = data_sort['线缆名称']
        rows['线缆规格'] = data_sort['线缆规格']
        rows['线缆长度'] = data_sort['线缆长度']
        rows['这条线缆的头连接的东西的名称'] = data_sort['CB_ND1']
        rows['这条线缆的尾连接的东西的名称'] = data_sort['CB_ND2']
        save.append(rows)

df = pd.DataFrame(save)
fw = open('结果文件New.txt', 'w', encoding='utf-8')
for row, one in df.groupby(by='NRO'):
    res = "================================================================"
    fw.writelines(res+'\n')
    for x1, y in one.iterrows():
        res = "{},{},{},{},{},{}".format(y['NRO'], y['线缆名称'], y['线缆规格'],
                                         y['线缆长度'], y['这条线缆的头连接的东西的名称'],
                                         y['这条线缆的尾连接的东西的名称'])
        fw.writelines(res+'\n')
        print(res)

not_in = [i for i in pt_id if i not in only_ones]
print(len(not_in), not_in)
nro_notin = nro[nro['pt_id'].isin(only_ones)]
nro_notin.to_csv("有链接.csv", index=None)
nro_notin = nro[~nro['pt_id'].isin(only_ones)]
nro_notin.to_csv("无链接.csv", index=None)
