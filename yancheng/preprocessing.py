import pandas as pd
import os
from glob import glob

from collections import Counter, OrderedDict
import matplotlib.pyplot as plt


def fun1():
    # date  day_of_week  brand   cnt
    """
  date  	     int   	  日期，经过脱敏，用数字来表示  
  day_of_week  	 int  	  表示星期几  
  brand  	    int  	  汽车品牌  
  cnt  	        int  	  上牌数
    :return: 
    """
    data = pd.read_csv("./datasets/train_20171215.txt", sep='\t')
    brand_1 = data[data['brand'].isin([1])]
    brand_2 = data[data['brand'].isin([2])]
    brand_3 = data[data['brand'].isin([3])]
    brand_4 = data[data['brand'].isin([4])]
    brand_5 = data[data['brand'].isin([5])]

    brand = data['brand']
    brand_total = []
    for x in brand.values.tolist():
        if x not in brand_total:
            brand_total.append(x)

    brand_total_name = [brand_1, brand_2, brand_3, brand_4, brand_5]
    i = 0
    for b in brand_total_name:
        i += 1
        b.to_csv("./datasets/results/brand_{}.csv".format(i))


def fun2():
    path = "./datasets/results/"
    files = glob(os.path.join(path, "*.csv"))
    out = []
    for file in files:
        rows = {}
        data = pd.read_csv(file, usecols=['date', 'day_of_week', 'cnt'])
        weeks = data['day_of_week']

        weeks_list = []
        for x in weeks.values.tolist():
            weeks_list.append(x)

        res = Counter(weeks_list)
        for x, y in res.most_common(7):
            rows[x] = y
        out.append(rows)

    df = pd.DataFrame(out)
    df.to_csv("./datasets/results/freq.csv", index=None)


def fun3():
    path = "yancheng/datasets/results/"
    files = glob(os.path.join(path, "*.csv"))
    i = 0
    plt.figure(figsize=(1200, 900), dpi=300)
    for file in files:
        print(file)
        data = pd.read_csv(file, usecols=['date', 'day_of_week', 'cnt'])
        cnt = data['cnt']
        plt.subplot(511 + i)
        plt.plot(cnt)
        plt.subplots_adjust(hspace=2)
        i += 1

    plt.legend(labels="cnt", loc='best')
    plt.savefig("yancheng/datasets/results/cnt.png")


def fun4():
    data = pd.read_csv("./datasets/train_20171215.txt", sep='\t', usecols=['date', 'day_of_week', 'cnt'])
    df = pd.DataFrame()
    df.insert(0, "date", None)
    df.insert(1, "week", None)
    df.insert(2, "cnt", None)

    lines_number = 0
    for i in range(1, 1033):
        cnt = data[data['date'].isin([i])]['cnt']
        total = 0
        for x in cnt:
            total += x

        weeks = data[data['date'].isin([i])]['day_of_week']
        week = weeks.values.tolist()[0]

        df.loc[lines_number, 'date'] = int(i)
        df.loc[lines_number, 'week'] = int(week)
        df.loc[lines_number, 'cnt'] = int(total)
        lines_number += 1

    df.to_csv("./datasets/results/data_train.csv", index=None)


def plot_total():
    data = pd.read_csv("./datasets/results/data_train.csv", usecols=['cnt'])
    plt.figure(dpi=300)
    plt.plot(data.values)
    plt.savefig("./datasets/results/total.png")


if __name__ == "__main__":
    plot_total()


