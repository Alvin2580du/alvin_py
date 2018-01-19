import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

from pyduyp.logger.log import log


def data_split_by_day():
    data = pd.read_csv("yancheng/datasets/train_20171215.txt", sep='\t')
    data_grouped = data.groupby(by='date')
    for i, j in tqdm(data_grouped):
        if len(j) == 0:
            continue
        j.to_csv("yancheng/datasets/results/train/{}.csv".format(i), index=None)


def get_date_features():
    root = 'yancheng/datasets/results/train/'
    log.info("Total files :{}".format(len(os.listdir(root))))
    res = []
    for file in tqdm(os.listdir(root)):
        rows = OrderedDict()
        file_name = os.path.join(root, file)
        data = pd.read_csv(file_name)
        cnt = np.sum(data['cnt'].values)
        rows['date'] = data['date'].values[0]
        rows['week'] = data['day_of_week'].values[0]
        rows['cnt'] = cnt
        res.append(rows)
    df = pd.DataFrame(res).sort_values(by='date')
    df.to_csv(os.path.join("yancheng/datasets/results", "total_by_day.csv"), index=None)


def plot_cnt():
    data = pd.read_csv(os.path.join("yancheng/datasets/results", "total_by_day.csv"), usecols=['cnt']).values
    plt.figure()
    plt.plot(data)
    plt.savefig(os.path.join("yancheng/datasets/results", "total_by_day.png"), dpi=300)


def get_features():
    data = pd.read_csv(os.path.join("yancheng/datasets/results", "total_by_day.csv"))
    X = data['cnt']
    y = data['week']
    corr = np.corrcoef(X, y)
    log.info("{}".format(corr))
    data_grouped = data.groupby(by='week')
    for i, j in data_grouped:
        j.to_csv("yancheng/datasets/results/{}.csv".format(i), index=None)


def ex_data_find():
    root = 'yancheng/datasets/results/train/week'
    save_root = 'yancheng/datasets/results/train/week_ex'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for f in os.listdir(root):
        f_name = os.path.join(root, f)
        data = pd.read_csv(f_name)

        if int(f.split(".")[0]) < 6:
            data = data[data['cnt'] >= 200]
            data.to_csv(os.path.join(save_root, "ex_{}".format(f)), index=None, header=None)
        else:
            data = data[data['cnt'] <= 1000]
            data.to_csv(os.path.join(save_root, "ex_{}".format(f)), index=None, header=None)


def conact():
    df = []
    save_root = 'yancheng/datasets/results/train/week_ex'
    for f in os.listdir(save_root):
        f_name = os.path.join(save_root, f)
        with open(f_name, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                rows = {'date': line.strip().split(",")[0], 'week': line.strip().split(",")[1],
                        'cnt': line.strip().split(",")[2]}
                df.append(rows)

    df = pd.DataFrame(df)
    df.columns = ['date', 'week', 'cnt']
    df = df.sort_values(by='date')
    df.to_csv("yancheng/datasets/results/train/total_by_day_ex.csv", index=None)


def sorted_df(by='cnt'):
    data = pd.read_csv("yancheng/datasets/results/train/total_by_day_ex.csv")

    data = data.sort_values(by=by)
    data.to_csv("yancheng/datasets/results/train/total_by_day_ex_sorted.csv", index=None)


def get_features_corr():
    data = pd.read_csv(os.path.join("yancheng/datasets/results", "total_by_day_ex_sorted.csv"))
    X = data['cnt']
    y = data['week']
    corr = np.corrcoef(X, y)
    log.info("{}".format(corr))

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        exit(1)

    method = sys.argv[1]

    if method == 'first':
        data_split_by_day()

    if method == 'second':
        get_date_features()
    if method == 'plot':
        plot_cnt()
    if method == 'third':
        get_features()
    if method == 'four':
        ex_data_find()
    if method == 'five':
        sorted_df()
    if method == 'six':
        get_features_corr()
