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


def get_nine(inputs):
    if inputs.split()[-1][0] == '9':
        return 'nine'
    else:
        return 'other'


def get_day(inputs):
    return inputs.split()[0]


def get_hour(inputs):
    return inputs.split(":")[0]


def build_part2():
    data1083 = pd.read_csv("rawpvr_2018-02-01_28d_1083 TueFri.csv")
    data1083['week'] = data1083['Date'].apply(time2week)
    data1083['DateNew'] = data1083['Date'].apply(get_day)
    data1083['Nine'] = data1083['Date'].apply(get_nine)
    data1083_north_lanes_sample = data1083[data1083['week'].isin(['Tuesday']) &
                                           data1083['Direction'].isin(['1']) &
                                           data1083['Nine'].isin(['nine'])]
    print(data1083_north_lanes_sample.shape)

    vel_counts = []
    for x, y in data1083_north_lanes_sample.groupby(by='DateNew'):
        rows = {'date': x, 'num': y.shape[0]}
        vel_counts.append(rows)

    vel_counts_df = pd.DataFrame(vel_counts)
    desc_df = vel_counts_df.sort_values(by='num')['num']
    desc_df_ = pd.DataFrame(desc_df.describe()).T

    desc_df_copy = pd.DataFrame()
    desc_df_copy['Range'] = desc_df_['max'] - desc_df_['min']
    desc_df_copy['1st Quartile'] = desc_df_['25%']
    desc_df_copy['2nd Quartile'] = desc_df_['50%']
    desc_df_copy['3rd Quartile'] = desc_df_['75%']
    desc_df_copy['Interquartile range'] = desc_df_['75%'] - desc_df_['25%']
    desc_df_copy.to_csv("desc_df_copy.csv", index=None)
    print(desc_df_copy)


def plotss(datainputs, name):
    total_traffic_volume = []
    # 首先按照小时分组，这样可以分组统计每小时的平均车流量
    for x, y in datainputs.groupby(by='Hour'):
        rows = OrderedDict()
        rows['date'] = x.split()[0]
        rows['hour'] = int(x.split()[1])
        rows['num'] = y.shape[0]
        total_traffic_volume.append(rows)

    df = pd.DataFrame(total_traffic_volume)
    df_sort = df.sort_values(by='hour')  # 按照小时来排序
    res = {}
    for x, y in df_sort.groupby(by='hour'):
        # 统计每小时的平均车流量
        res[x] = y['num'].mean()
    objects = list(res.keys())  # 对象是时间
    performance = list(res.values())  # 条形图的高度表示平均车流量
    y_pos = np.arange(len(objects))
    plt.figure(figsize=(10, 8))  # 新建一个画布
    plt.barh(y_pos, performance, align='center', alpha=0.5)  # 画条形图
    plt.yticks(y_pos, objects)

    plt.ylabel("Average traffic volume")  # x轴和y轴的标签
    plt.xlabel("Each hour of Tuesday")
    plt.title("{} the average traffic volume for each hour of Tuesday".format(name))  # 标题
    plt.savefig("{}.png".format(name))  # 保存到图片
    plt.show()
    plt.close()


def build_task3():
    data1083 = pd.read_csv("rawpvr_2018-02-01_28d_1083 TueFri.csv")
    data1083['week'] = data1083['Date'].apply(time2week)
    data1083['Hour'] = data1083['Date'].apply(get_hour)
    # North  选择周二北边的数据
    data1083_north = data1083[data1083['week'].isin(['Tuesday']) & data1083['Direction'].isin(['1'])]
    plotss(data1083_north, name='North direction')
    # South 选择周二南边的数据
    data1083_South = data1083[data1083['week'].isin(['Tuesday']) & data1083['Direction'].isin(['2'])]
    plotss(data1083_South, name='South direction')


def get_hour_(inputs):
    output = inputs.split()[-1][:2].replace(":", "")
    return int(output)


def build_task4():
    # 先按小时分组， 然后按照gaps排序，求中位数，然后填充缺失值
    data1083 = pd.read_csv("rawpvr_2018-02-01_28d_1083 TueFri.csv")
    data1083['week'] = data1083['Date'].apply(time2week)
    data1083['Hour'] = data1083['Date'].apply(get_hour_)
    data1083_north = data1083[data1083['week'].isin(['Tuesday']) &
                              data1083['Direction'].isin(['1']) &
                              data1083['Lane'].isin(['2'])]

    data1083_north_group = data1083_north.sort_values(by='Hour', ascending=True).groupby(by='Hour')
    save = []
    # 首先按照小时分组，这样可以分组统计每小时的中位数
    for x, y in data1083_north_group:
        if 7 <= x <= 19:
            y_sorted_copy = y.sort_values(by='Gap (s)')
            g_m = y['Gap (s)'].median()
            h_m = y['Headway (s)'].median()
            y_sorted_copy['Gap (s)'] = y['Gap (s)'].fillna(g_m)
            y_sorted_copy['Headway (s)'] = y['Headway (s)'].fillna(h_m)
            print("time: %02d" % x)
            s = '      g: %0.3f, h: %0.3f' % (g_m, h_m)
            print(s)
            print("---------------------------------------")
            for x1, y1 in y_sorted_copy.iterrows():
                rows = OrderedDict()
                rows['Date'] = pd.to_datetime(y1['Date'], format="%Y-%m-%d %H:%M:%S")
                rows['Lane'] = y1['Lane']
                rows['Lane Name'] = y1['Lane Name']
                rows['Direction'] = y1['Direction']
                rows['Direction Name'] = y1['Direction Name']
                rows['Speed (mph)'] = y1['Speed (mph)']
                rows['Headway (s)'] = y1['Headway (s)']
                rows['Gap (s)'] = y1['Gap (s)']
                rows['Flags'] = y1['Flags']
                rows['Flag Text'] = y1['Flag Text']
                save.append(rows)

    df = pd.DataFrame(save)
    df.sort_values(by='Date', ascending=True).to_csv("rawpvr_2018-02-01_28d_1083 TueFri_FillNa.csv", index=None)


if __name__ == '__main__':
    method = 'build_task4'

    if method == 'build_part2':
        build_part2()

    if method == 'build_task3':
        build_task3()

    if method == 'build_task4':
        build_task4()
