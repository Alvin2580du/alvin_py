""" 需求

1. 根据附件1中的用户观看信息数据，试分析用户的收视偏好，并给出附件2的产品的营销推荐方案

2. 为了更好的维用户服务，扩大营销范围，利用附件1到附件3的数据，试对相似偏好的用户进行分类（用户标签），
    对产品进行分类打包（产品标签），并给出营销推荐方案


附件1： 用户收视信息，用户回看信息，用户点播信息，用户单片点播信息。
附件2： 电视产品信息数据
附件3：用户基本信息
"""

import pandas as pd
import os
from datetime import datetime


def compare_time(l_time, start_t, end_t):
    s_time = datetime.strptime(start_t, "%H:%M")
    e_time = datetime.strptime(end_t, "%H:%M")
    log_time = datetime.strptime(l_time, "%H:%M")
    if (log_time >= s_time) and (log_time <= e_time):
        return True
    return False


def gettimerange(x):
    try:
        x2date = datetime.strptime(x, '%Y/%m/%d %H:%M').strftime("%H:%M")
    except:
        x2date = datetime.strptime("00:00", "%H:%M").strftime("%H:%M")

    one = "00:00"
    two = "07:59"
    three = "15:59"
    if compare_time(x2date, one, two):
        return 0
    elif compare_time(x2date, two, three):
        return 1
    else:
        return 2


def convert(inputs):
    try:
        out = datetime.strptime(inputs, '%Y/%m/%d %H:%M')
    except:
        out = datetime.strptime(inputs, '%Y/%m/%d')
    return out


def timecost(starttime, endtime):
    t1 = convert(starttime)
    t2 = convert(endtime)
    return (t1 - t2).seconds


def compute_time_cost():
    shoushi = pd.read_csv("./datasets/tv_data/shoushi.csv")
    shoushicopy = shoushi.copy()
    shoushicopy['时长'] = shoushi.apply(lambda row: timecost(row['收看开始时间'], row['收看结束时间']), axis=1)
    shoushicopy['时间段'] = shoushi['收看开始时间'].apply(gettimerange)
    shoushicopy.to_csv("./datasets/tv_data/shoushiNew.csv")


def group_user():
    # 1
    shoushi = pd.read_csv("./datasets/tv_data/shoushiNew.csv")
    huikan = pd.read_csv("./datasets/tv_data/huikan.csv")
    dianbo = pd.read_csv("./datasets/tv_data/dianbo.csv")
    danpiandianbo = pd.read_csv("./datasets/tv_data/danpiandianbo.csv")

    # 2
    chanpin = pd.read_csv("./datasets/tv_data/chanpinxinxi.csv", encoding='utf-8')

    # 3
    jiben = pd.read_csv("./datasets/tv_data/jibenxinxi.csv")

    shoushigroup = shoushi.groupby(by='机顶盒设备号')
    huikangroup = huikan.groupby(by='设备号')
    dianbogroup = dianbo.groupby(by='设备号')
    danpiandianbogroup = danpiandianbo.groupby(by='设备号')

    for x, y in shoushigroup:
        save_path = os.path.join("./datasets/tv_data/results", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_shoushi.csv".format(x))
        y.to_csv(save_name, index=None)

    for x, y in huikangroup:
        save_path = os.path.join("./datasets/tv_data/results", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_huikan.csv".format(x))
        y.to_csv(save_name, index=None)

    for x, y in dianbogroup:
        save_path = os.path.join("./datasets/tv_data/results", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_dianbo.csv".format(x))
        y.to_csv(save_name, index=None)

    for x, y in danpiandianbogroup:
        save_path = os.path.join("./datasets/tv_data/results", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_dianbodanpian.csv".format(x))
        y.to_csv(save_name, index=None)


hao2mingdata = pd.read_csv("./datasets/tv_data/shoushi.csv", usecols=['频道号', '频道名'])


def hao2ming(hao):
    for one in hao2mingdata.values:
        if hao == one[0]:
            return one[1]
    return -1


def make_huikan():
    huikan = pd.read_csv("./datasets/tv_data/huikan.csv")
    huikancopy = huikan.copy()
    huikancopy['频道号'] = huikan['频道'].apply(hao2ming)
    huikancopy['时长'] = huikan.apply(lambda row: timecost(row['回看开始时间'], row['回看结束时间']), axis=1)
    huikancopy['时间段'] = huikan['回看开始时间'].apply(gettimerange)
    huikancopy.to_csv("./datasets/tv_data/huikanNew.csv", index=None)


# chanpin = pd.read_csv("./datasets/tv_data/chanpinxinxi.csv", encoding='utf-8', usecols=['连续剧分类']).values.tolist()
# chanpin = [i for j in chanpin for i in j]
