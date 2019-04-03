"""
需求：
    1、不同学位及不同性别的学生在不同时间的消费情况。
    2、学生在不同地点的消费情况与性别和学位的关联。
    3、不同地方消费金额数量的变化。
    4、在某一特定的商户（例如公共浴室或者独立浴室）进行消费的人数量与时间之间的联系。

auth : Alvin_2580
"""

import pandas as pd
import os
import numpy as np
from datetime import datetime
import time
from matplotlib import pyplot as plt
import matplotlib.dates as dates
import matplotlib.ticker as ticker
data = pd.read_csv("./datasets/shuju.csv")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def compare_time(l_time, start_t, end_t):
    s_time = datetime.strptime(start_t, "%H:%M")
    e_time = datetime.strptime(end_t, "%H:%M")
    log_time = datetime.strptime(l_time, "%H:%M")
    if (log_time >= s_time) and (log_time <= e_time):
        return True
    return False


def gettimerange(x):
    x2date = datetime.strptime(x, '%Y/%m/%d %H:%M').strftime("%H:%M")
    one = "00:00"
    two = "07:59"
    three = "15:59"
    if compare_time(x2date, one, two):
        return "早上"
    elif compare_time(x2date, two, three):
        return "中午"
    else:
        return "晚上"


def chagetime2Three(inputs):
    x2date = datetime.strptime(inputs, '%Y/%m/%d %H:%M').strftime("%Y/%m/%d ")
    return x2date


def chagetime2hour(inputs):
    x2date = datetime.strptime(inputs, '%Y/%m/%d %H:%M').strftime("%Y/%m/%d %H")
    return x2date


def groupdata():
    groupedbydegree = data.groupby(by='学生类型')
    for i, j in groupedbydegree:
        j.to_csv("./datasets/{}.csv".format(i), index=None, encoding='utf-8')

    groupedbysex = data.groupby(by='学生性别')
    for i, j in groupedbysex:
        j.to_csv("./datasets/{}.csv".format(i), index=None, encoding='utf-8')

    groupedbysex = data.groupby(by='商户名称')
    for i, j in groupedbysex:
        j.to_csv("./datasets/{}.csv".format(i), index=None, encoding='utf-8')


def analsisage(name='本科'):
    """
    不同学位及不同性别的学生在不同时间的消费情况。
    :param name:
    :return:
    """
    maxmoney = 1500000
    f = open("./datasets/{}.csv".format(name), 'rb')
    boy = pd.read_csv(f, usecols=['交易日期时间', '交易金额'], encoding='utf-8', sep=',')
    xiaofeishijian = boy['交易日期时间']
    print(len(xiaofeishijian), type(xiaofeishijian), xiaofeishijian[0], type(xiaofeishijian[0]))
    boynew = pd.DataFrame()
    boynew['time'] = boy['交易日期时间'].apply(chagetime2Three)
    boynew['money'] = boy['交易金额']
    boygrouped = boynew.groupby(by='time')
    label = []
    total = []
    for i, j in boygrouped:
        costsum = sum(j['money'])
        label.append(i)
        total.append(costsum)

    plt.figure(dpi=200)
    ax1 = plt.subplot(111)
    width = 0.5
    x_bar = np.array(label)
    data = np.array(total)
    rect = ax1.bar(left=x_bar, height=data, width=width, color="lightblue")

    for rec in rect:
        x = rec.get_x()
        height = rec.get_height()
        ax1.text(x + 0.1, 1.02 * height, str(height))
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(("早上", "中午", "晚上"))
    ax1.set_ylabel("sales")
    ax1.set_title("{}生消费总值和时间的关系".format(name))
    ax1.grid(True)
    ax1.set_ylim(0, maxmoney)
    plt.savefig("./results/{}cost.png".format(name))


def analsisaddress(name='独立浴室'):
    """
    学生在不同地点的消费情况与性别和学位的关联。
    :param name:
    :return:
    """
    # 公共浴室二, 公共浴室三, 公共浴室一
    maxmoney = 1500000
    types = '学生类型'  # 性别， 学生类型
    xlabels = ("博士", "硕士", "本科")  # ("博士", "硕士", "本科")， ("男", "女")
    f = open("./datasets/{}.csv".format(name), 'rb')
    sex = pd.read_csv(f, usecols=['学生性别', '交易金额'], encoding='utf-8', sep=',')

    f1 = open("./datasets/{}.csv".format(name), 'rb')
    degree = pd.read_csv(f1, usecols=['学生类型', '交易金额'], encoding='utf-8', sep=',')
    if types == '性别':
        datagrouped = sex.groupby(by='学生性别')
    else:
        datagrouped = degree.groupby(by='学生类型')
    label = []
    total = []
    for i, j in datagrouped:
        costsum = sum(j['交易金额'])
        label.append(i)
        total.append(costsum)

    plt.figure(dpi=200)
    ax1 = plt.subplot(111)
    width = 0.5
    x_bar = np.array(label)
    data = np.array(total)
    rect = ax1.bar(left=x_bar, height=data, width=width, color="lightblue")

    for rec in rect:
        x = rec.get_x()
        height = rec.get_height()
        ax1.text(x + 0.1, 1.02 * height, str(height))
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(xlabels)
    ax1.set_ylabel("sales")
    ax1.set_title("{}生消费总值和{}的关系".format(name, types))
    ax1.grid(True)
    ax1.set_ylim(0, maxmoney)
    plt.savefig("./results/{}_{}.png".format(name, types))


def addresstotalcost():
    """
    不同地方消费金额数量的变化。
    :return:
    """
    maxmoney = 1500000
    data = pd.read_csv("./datasets/shuju.csv", usecols=['商户名称', '交易金额'])
    groupedbysex = data.groupby(by='商户名称')

    label = []
    total = []
    for i, j in groupedbysex:
        costsum = sum(j['交易金额'])
        label.append(i)
        total.append(costsum)

    plt.figure(dpi=200)
    ax1 = plt.subplot(111)
    width = 0.5
    x_bar = np.array(label)
    data = np.array(total)
    rect = ax1.bar(left=x_bar, height=data, width=width, color="lightblue")

    for rec in rect:
        x = rec.get_x()
        height = rec.get_height()
        ax1.text(x + 0.1, 1.02 * height, str(height))
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(('独立浴室', '公共浴室二', '公共浴室一', '公共浴室三'))
    ax1.set_ylabel("sales")
    ax1.set_title("不同浴室消费总值对比图")
    ax1.grid(True)
    ax1.set_ylim(0, maxmoney)
    plt.savefig("./results/不同浴室.png")


def analysislast():
    maxmoney = 10000
    from collections import Counter
    # 在某一特定的商户（例如公共浴室或者独立浴室）进行消费的人数量与时间之间的联系。
    data = pd.read_csv("./datasets/shuju.csv", usecols=['商户名称', '交易日期时间'])
    gonggongweiyu = data[~data['商户名称'].isin(["独立浴室"])]['交易日期时间'].apply(gettimerange)
    Threecount = Counter([i for j in gonggongweiyu.tolist() for i in j])

    label = []
    total = []
    for i, j in Threecount.most_common(3):
        label.append(i)
        total.append(j)

    plt.figure(dpi=200)
    ax1 = plt.subplot(111)
    width = 0.5
    x_bar = np.array(label)
    data = np.array(total)
    rect = ax1.bar(left=x_bar, height=data, width=width, color="lightblue")

    for rec in rect:
        x = rec.get_x()
        height = rec.get_height()
        ax1.text(x + 0.1, 1.02 * height, str(height))
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(('早上', '中午', '晚上'))
    ax1.set_ylabel("人数")
    ax1.set_title("消费的人数量与时间之间的联系")
    ax1.grid(True)
    ax1.set_ylim(0, maxmoney)
    plt.savefig("./results/公共浴室.png")


def moreanalysis():
    data = pd.read_csv("./datasets/shuju.csv", usecols=['交易金额', '学生性别'])
    datagrouped = data.groupby(by='学生性别')
    for i, j in datagrouped:
        desci = j.describe()
        desci.to_csv("./results/描述性统计分析-交易金额_{}.csv".format(i), header=None, encoding='utf-8')


def analsisage_v2(name='本科'):
    """
    不同学位及不同性别的学生在不同时间的消费情况。
    :param name:
    :return:
    """
    import matplotlib.dates as dates
    import matplotlib.ticker as ticker

    f = open("./datasets/{}.csv".format(name), 'rb')
    boy = pd.read_csv(f, usecols=['交易日期时间', '交易金额'], encoding='utf-8', sep=',')
    xiaofeishijian = boy['交易日期时间']
    boynew = pd.DataFrame()
    boynew['time'] = boy['交易日期时间'].apply(chagetime2Three)
    boynew['money'] = boy['交易金额']
    boygrouped = boynew.groupby(by='time')
    label = []
    total = []
    for i, j in boygrouped:
        costavergae = sum(j['money']) / len(j['money'])
        label.append(i)
        total.append(costavergae)
    plt.figure()
    plt.plot(label, total)
    plt.xlabel("时间")
    plt.ylabel("平均消费")
    plt.title("{}每天平均消费情况变化图".format(name))
    plt.savefig("./results_2/{}.png".format(name))


def analsisage_by_hour(name='本科'):
    """
    不同学位及不同性别的学生在不同时间的消费情况。
    :param name:
    :return:
    """

    f = open("./datasets/{}.csv".format(name), 'rb')
    boy = pd.read_csv(f, usecols=['交易日期时间', '交易金额'], encoding='utf-8', sep=',')
    xiaofeishijian = boy['交易日期时间']
    boynew = pd.DataFrame()
    boynew['time'] = boy['交易日期时间'].apply(chagetime2hour)
    boynew['money'] = boy['交易金额']
    boygrouped = boynew.groupby(by='time')
    label = []
    total = []
    for i, j in boygrouped:
        costavergae = sum(j['money']) / len(j['money'])
        if i[:10] != "2014/10/10":
            continue
        else:
            label.append(i)
            total.append(costavergae)
    plt.figure()
    plt.plot(label, total)
    plt.xlabel("时间")
    plt.ylabel("平均消费")
    plt.title("{}每天平均消费情况变化图".format(name))
    plt.savefig("./results_2/{}_hour.png".format(name))


if __name__ == "__main__":
    method = 'analsisage_v2'
    if method == 'groupdata':
        groupdata()

    if method == 'analsisage':
        analsisage()

    if method == 'analsisaddress':
        for names in ['公共浴室二', '公共浴室一', '公共浴室三']:
            analsisaddress(name=names)

    if method == 'addresstotalcost':
        addresstotalcost()

    if method == 'analysislast':
        analysislast()

    if method == 'moreanalysis':
        moreanalysis()

    if method == 'analsisage_v2':
        names = ["博士", "硕士", "本科", "男", "女", '公共浴室二', '公共浴室一', '公共浴室三', '独立浴室']
        for n in names:
            analsisage_v2(name=n)

    if method == 'analsisage_by_hour':

        names = ["博士", "硕士", "本科", "男", "女", '公共浴室二', '公共浴室一', '公共浴室三', '独立浴室']
        for n in names:
            analsisage_by_hour(name=n)