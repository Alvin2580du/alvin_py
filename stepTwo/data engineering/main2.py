import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict


def time2week(date, w=1):
    week = datetime.datetime.fromtimestamp(time.mktime(time.strptime(date, "%Y-%m-%d %H:%M:%S"))).weekday()
    if week == w:
        return 'Tuesday'
    else:
        return 'Friday'


def getHour_17(x):
    # 取日期的小时那一位数字
    res = int(x.split(":")[0].split()[-1])
    if 17 <= res < 18:
        return "1"
    else:
        return "0"


def getHour(x):
    # 取日期的小时那一位数字
    return int(x.split(":")[0].split()[-1])


def data_plot(datainputs, file_name):
    # 获取画图数据，并按照时间分组，保存到字典中
    total_traffic_volume = {}
    for x, y in datainputs.sort_values(by='Hour').groupby(by='Hour'):
        total_traffic_volume[x] = y['Speed (mph)'].mean()
    print("{}, {}".format(file_name, list(total_traffic_volume.values())))
    # 画图函数
    objects = total_traffic_volume.keys()  # 获取字典的键作为条形图的x轴的对象
    plt.figure(figsize=(15, 10))  # 新建画板
    plt.barh(np.arange(len(list(objects))), list(total_traffic_volume.values()), alpha=0.5, color='red')  # 画条形图
    plt.yticks(np.arange(len(objects)), list(objects))  # x轴的刻度
    plt.title("{} the average traffic volume for each hour of Tuesday".format(file_name))  # 标题
    plt.ylabel("Average traffic volume")  # y轴的标签
    plt.xlabel("Each hour of {}".format(file_name))  # x轴的标签
    plt.savefig("{}.png".format(file_name))  # 保存图片
    plt.close()  # 关闭画板


def build_task5():
    data_1083s = pd.read_csv("rawpvr_2018-02-01_28d_1083 TueFri.csv")  # 读取数据
    data_1083s['week'] = data_1083s['Date'].apply(time2week)  # 获取星期的值
    data_1083s['Hour'] = data_1083s['Date'].apply(getHour)  # 获取小时的值
    data_1083s = data_1083s.dropna(subset=['Speed (mph)'])

    # North  选择周二北边的数据
    data1083_north = data_1083s[data_1083s['week'].isin(['Tuesday']) & data_1083s['Direction'].isin(['1'])]
    data_plot(data1083_north, file_name='Tuesday North direction')
    # South 选择周二南边的数据
    data1083_South = data_1083s[data_1083s['week'].isin(['Tuesday']) & data_1083s['Direction'].isin(['2'])]
    data_plot(data1083_South, file_name='Tuesday South direction')

    # -------------------------------------------------
    # North 选择周五北边的数据
    data1083_north = data_1083s[data_1083s['week'].isin(['Friday']) & data_1083s['Direction'].isin(['1'])]
    data_plot(data1083_north, file_name='Friday North direction')
    # South 选择周五南边的数据
    data1083_South = data_1083s[data_1083s['week'].isin(['Friday']) & data_1083s['Direction'].isin(['2'])]
    data_plot(data1083_South, file_name='Friday South direction')


def build_task6():
    data_1083s = pd.read_csv("rawpvr_2018-02-01_28d_1083 TueFri.csv")  # 读取数据
    data_1083s['week'] = data_1083s['Date'].apply(time2week)  # 获取星期的值
    data_1083s['Hour'] = data_1083s['Date'].apply(getHour_17)  # 获取小时的值
    data_1083s = data_1083s.dropna(subset=['Speed (mph)'])

    data1083_north = data_1083s[data_1083s['week'].isin(['Friday']) &
                                data_1083s['Direction'].isin(['1']) &
                                data_1083s['Hour'].isin(['1'])]
    total_traffic_speed_1083 = {}
    for x, y in data1083_north.sort_values(by='Hour').groupby(by='Lane'):
        mean_v = y['Speed (mph)'].mean()
        total_traffic_speed_1083[x] = mean_v
    # 1415
    data_1415s = pd.read_csv("rawpvr_2018-02-01_28d_1415 TueFri.csv")  # 读取数据
    data_1415s['week'] = data_1415s['Date'].apply(time2week)  # 获取星期的值
    data_1415s['Hour'] = data_1415s['Date'].apply(getHour_17)  # 获取小时的值
    data_1415s = data_1415s.dropna(subset=['Speed (mph)'])

    data1415_north = data_1415s[data_1415s['week'].isin(['Friday']) &
                                data_1415s['Direction'].isin(['1']) &
                                data_1415s['Hour'].isin(['1'])]

    total_traffic_speed_1415 = {}
    for x, y in data1415_north.sort_values(by='Hour').groupby(by='Lane'):
        mean_v = y['Speed (mph)'].mean()
        total_traffic_speed_1415[x] = mean_v

    res = list(total_traffic_speed_1083.values()) + list(total_traffic_speed_1415.values())
    average_mean = (3.019864 / np.mean(np.array(res))) * 60
    print(average_mean)


if __name__ == '__main__':
    method = 'build_task6'

    # 要运行第五题，就把method的值改成build_task5， 运行第六题改成build_task6

    if method == 'build_task5':
        build_task5()

    if method == 'build_task6':
        build_task6()
