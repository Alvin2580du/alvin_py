import pandas as pd
import os
from tqdm import tqdm
from collections import OrderedDict
import numpy as np


def add_month():
    data_path = './datasets/临时数据/'
    save_path = './datasets/data_merge'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in tqdm(os.listdir(data_path)):
        filename = os.path.join(data_path, file)
        data = pd.read_csv(filename, sep=';')
        data['month'] = file[:7]
        data.to_csv(os.path.join(save_path, file), index=None, sep=',')


def merge_data():
    save = []
    save_path = './datasets/data_merge'
    for file in tqdm(os.listdir(save_path)):
        filename = os.path.join(save_path, file)
        with open(filename, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                if "证券代码" in line:
                    continue
                linesp = line.split(",")
                rows = OrderedDict()
                rows['证券代码'] = linesp[0]
                rows['绝对收益率'] = linesp[1]
                rows['规模'] = linesp[2]
                rows['BP'] = "{}".format(linesp[3])
                rows['ROE'] = "{}".format(linesp[4])
                rows['资产变化率'] = linesp[5]
                rows['month'] = linesp[6].replace('\n', '')
                save.append(rows)

    df = pd.DataFrame(save)
    df.columns = ['证券代码', '绝对收益率', '规模', 'BP', 'ROE', '资产变化率', 'month']
    df_group = df.groupby(by='证券代码')
    save_dir = './datasets/data_id'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for x, y in df_group:
        y.to_csv(os.path.join(save_dir, "{}.csv".format(x)), index=None)


def means(inputs):
    return np.mean(inputs.dropna())


def compute_mean():
    save_path = './datasets/data_id'
    save_list = []
    for file in os.listdir(save_path):
        filename = os.path.join(save_path, file)
        data = pd.read_csv(filename)
        data_group = data.groupby(by='month')

        for x, y in data_group:
            # 证券代码,绝对收益率,规模,BP,ROE,资产变化率,month
            rows = OrderedDict({'证券代码': "%06d" % int(file.replace(".csv", "")),
                                '月份': x,
                                '绝对收益率': means(y['绝对收益率']),
                                '规模': means(y['规模']),
                                'BP': means(y['BP']),
                                'ROE': means(y['ROE']),
                                '资产变化率': means(y['资产变化率'])})
            save_list.append(rows)
    df = pd.DataFrame(save_list)
    df_group = df.groupby(by='月份')
    for x, y in df_group:
        del y['月份']
        y_save = y.sort_values(by="证券代码")
        save_dir = './datasets/huizong'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # y_save = y_save.fillna("null")
        y_save.to_excel(os.path.join(save_dir, '汇总_{}.xlsx'.format(x)), index=None)


def build():
    print("start")
    add_month()
    print("add month success ")
    merge_data()
    print("merge data success")
    compute_mean()
    print("compute_mean success")


build()
