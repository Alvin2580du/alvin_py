import numpy as np
import time
import datetime
import pandas as pd


def getWdata(d, w=1):
    # 格式化时间，并计算日期是星期几
    x1 = time.mktime(time.strptime(d, "%Y-%m-%d %H:%M:%S"))
    week = datetime.datetime.fromtimestamp(x1).weekday()
    if week == w:
        return 'T'
    else:
        return 'F'


def Is_seventeen(x):
    # 取日期的小时那一位数字
    t = x.split(":")[0].split()[-1]
    if 17 <= int(t) < 18:
        return "s"
    else:
        return "f"


def buildAvgSpeeds():
    # 1415
    data2 = pd.read_csv("rawpvr_2018-02-01_28d_1415 TueFri.csv")  # 读取数据
    data2['week'] = data2['Date'].apply(getWdata)  # 获取星期的值
    data2['Hour'] = data2['Date'].apply(Is_seventeen)  # 获取小时的值
    data2 = data2.dropna(subset=['Speed (mph)'])
    data2North = data2[data2['week'].isin(['F']) & data2['Direction'].isin(['1']) & data2['Hour'].isin(['s'])]

    Speed_1415 = []
    for x, y in data2North.sort_values(by='Hour').groupby(by='Lane'):
        Speed_1415.append(y['Speed (mph)'].mean())

    data1 = pd.read_csv("rawpvr_2018-02-01_28d_1083 TueFri.csv")  # 读取数据
    data1['week'] = data1['Date'].apply(getWdata)  # 获取星期的值
    data1['Hour'] = data1['Date'].apply(Is_seventeen)  # 获取小时的值
    data1 = data1.dropna(subset=['Speed (mph)'])

    data1083_north = data1[data1['week'].isin(['F']) & data1['Direction'].isin(['1']) & data1['Hour'].isin(['s'])]
    Speed_1083 = []
    for x, y in data1083_north.sort_values(by='Hour').groupby(by='Lane'):
        Speed_1083.append(y['Speed (mph)'].mean())

    res = Speed_1083 + Speed_1415
    average_mean = (3.019864 / np.mean(np.array(res))) * 60
    print(average_mean)


def main():
    print("[!] Start runing ...")
    print("Task 6")
    buildAvgSpeeds()


if __name__ == "__main__":
    main()
