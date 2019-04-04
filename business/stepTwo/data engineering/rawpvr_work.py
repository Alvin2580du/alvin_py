import numpy as np
import pandas as pd
import time
import datetime


def get_date(date, sep=1):
    # 格式化时间，并计算日期是星期几
    x1 = datetime.datetime.fromtimestamp(time.mktime(time.strptime(date, "%Y-%m-%d %H:%M:%S")))
    week = x1.weekday()
    if week == sep:
        return '1'
    else:
        return '0'


def get_time(x):
    # 取日期的小时那一位数字
    if 17 <= int(x.split(":")[0].split()[-1]) < 18:
        return "yes"
    else:
        return "no"


def compute_average_speeds():
    print("compute_average_speeds")
    x1083 = pd.read_csv("rawpvr_2018-02-01_28d_1083 TueFri.csv")  # 读取数据
    x1083['week'] = x1083['Date'].apply(get_date)  # 获取星期的值
    x1083['Hour'] = x1083['Date'].apply(get_time)  # 获取小时的值
    x1083 = x1083.dropna(subset=['Speed (mph)'])

    data1083_north = x1083[x1083['week'].isin(['0']) & x1083['Direction'].isin(['1']) & x1083['Hour'].isin(['yes'])]
    print(data1083_north.shape)
    Ts_1083 = {}
    for k1, v1 in data1083_north.sort_values(by='Hour').groupby(by='Lane'):
        Ts_1083[k1] = v1['Speed (mph)'].mean()
    x1415 = pd.read_csv("rawpvr_2018-02-01_28d_1415 TueFri.csv")  # 读取数据
    x1415['week'] = x1415['Date'].apply(get_date)  # 获取星期的值
    x1415['Hour'] = x1415['Date'].apply(get_time)  # 获取小时的值
    x1415 = x1415.dropna(subset=['Speed (mph)'])
    data1415_north = x1415[x1415['week'].isin(['0']) & x1415['Direction'].isin(['1']) & x1415['Hour'].isin(['yes'])]
    Ts_1415 = {}
    for k2, v2 in data1415_north.sort_values(by='Hour').groupby(by='Lane'):
        Ts_1415[k2] = v2['Speed (mph)'].mean()

    d3 = {}
    counter = 0
    for k, v in Ts_1083.items():
        counter += 1
        d3[counter] = v
    for k1, v1 in Ts_1415.items():
        counter += 1
        d3[counter] = v1
    average_mean = (3.019864 / np.mean(np.array(list(d3.values())))) * 60
    print(average_mean)


# task 6
compute_average_speeds()
