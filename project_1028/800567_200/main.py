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


get_csv()
exit(1)

"""
NRO,    线缆名称,       线缆规格,     线缆长度,    CB_ND1,             CB_ND2
86DAN,  86DAN_TR_00002,288,         537,        86DAN,              86DAN_PEP_TR_00003
86DAN,  86DAN_TR_00001,72,          9,          86DAN,              86DAN_00013
86DAN,  86DAN_TR_00003,144,         312,        86DAN,              86DAN_00014
86DAN,  86DAN_TR_00004,72,          1827,       86DAN,              86DAN_90009
86DAN,  86DAN_TR_00006,144,         32,         86DAN_PEP_TR_00003, 86DAN_00012
86DAN,  86DAN_TR_00005,144,         0,          86DAN_PEP_TR_00003, 86DAN_90000


NRO,线缆名称,线缆规格,线缆长度,CB_ND1,CB_ND2
86CGS,86CGS_TR_00013,72,2304,86CGS_PEP_TR_00008,86CGS_PEP_TR_00018
86CGS,86CGS_TR_00005,72,2308,86CGS,86CGS_PEP_TR_00008
86CGS,86CGS_TR_00014,72,115,86CGS_PEP_TR_00018,86CGS_00002



"""

for file in os.listdir("./data"):
    file_name = os.path.join("./data", file)
    data = pd.read_csv(file_name)
    data_sort = data.sort_values(by='CB_ND1')
    length = data_sort.shape[0]
    cb_dn1_length = len(list(set(data_sort['CB_ND1'].values.tolist())))
    data_group = data_sort.groupby(by='CB_ND1')
    # NRO,线缆名称,线缆规格,线缆长度,CB_ND1,CB_ND2
    fw = open("./results/{}".format(file), 'w', encoding='utf-8')
    columns = ['NRO'] + ['线缆名称', '线缆规格', '线缆长度', '链接的东西名称'] * cb_dn1_length
    print(columns)
    line = ",".join(columns)
    fw.writelines(line + "\n")
    for i in range(length):
        for j in range(i + 1, length):
            first = data_sort.loc[[i]]
            second = data_sort.loc[[j]]
            first_dn1 = first['CB_ND1'].values.tolist()[0]
            first_dn2 = first['CB_ND2'].values.tolist()[0]
            second_dn1 = second['CB_ND1'].values.tolist()[0]
            second_dn2 = second['CB_ND2'].values.tolist()[0]
            if first_dn1 == second_dn2:
                res = "{},{},{},{} "
                print("second:{}".format(second))
            if first_dn2 == second_dn1:
                print("first:{}".format(first))

    exit(1)

    # for x, y in data_sort.iterrows():
    #     nro = y['NRO']
    #     name = y['线缆名称']
    #     guige = y['线缆规格']
    #     changdu = y['线缆长度']
    #     dn1 = y['CB_ND1']
    #     dn2 = y['CB_ND2']
    #     str0 = ",,,"
    #     str1 = "{},{},{},{}".format(name, guige, changdu, dn2)
    #     if len(dn1) > 5:
    #         str_w = "{},{},{}".format(nro, str0, str1)
    #     else:
    #         str_w = "{},{},{}".format(nro, str1, str0)
    #
    #     print(str_w)
    #     fw.writelines(str_w + "\n")
