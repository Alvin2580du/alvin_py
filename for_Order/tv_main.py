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
from tqdm import tqdm


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

    one = "12:00"
    two = "19:59"
    three = "23:59"
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


hao2mingdata = pd.read_csv("./datasets/tv_data/shoushi.csv", usecols=['频道号', '频道名'])
datas = {}
for one in hao2mingdata.values:
    datas[one[1]] = one[0]


def hao2ming(hao):
    return datas[hao]


def make_huikan():
    huikan = pd.read_csv("./datasets/tv_data/huikan.csv")
    huikancopy = huikan.copy()
    huikancopy['频道号'] = huikan['频道'].apply(hao2ming)
    huikancopy['时长'] = huikan.apply(lambda row: timecost(row['回看开始时间'], row['回看结束时间']), axis=1)
    huikancopy['时间段'] = huikan['回看开始时间'].apply(gettimerange)
    huikancopy.to_csv("./datasets/tv_data/huikanNew.csv", index=None)


def group_user():
    # 1
    shoushi = pd.read_csv("./datasets/tv_data/shoushiNew.csv")
    huikan = pd.read_csv("./datasets/tv_data/huikanNew.csv")
    dianbo = pd.read_csv("./datasets/tv_data/dianbo.csv")
    danpiandianbo = pd.read_csv("./datasets/tv_data/danpiandianbo.csv")

    # 2
    # chanpin = pd.read_csv("./datasets/tv_data/chanpinxinxi.csv", encoding='utf-8')
    #
    # # 3
    # jiben = pd.read_csv("./datasets/tv_data/jibenxinxi.csv")

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


def get_shoushionehot():
    pindaohao = pd.read_csv("./datasets/tv_data/shoushi.csv", usecols=['频道号']).values
    pindaohao2list = [i for j in pindaohao for i in j]

    path = './datasets/tv_data/results/'
    shoushi_list = []
    feature_columns = []

    for dirpath, dirnames, filenames in (os.walk(path)):
        for file in tqdm(filenames):
            rows = {}
            fullpath = os.path.join(dirpath, file)
            userid = fullpath.split("/")[-2]
            rows['userid'] = userid

            if "shoushi" in fullpath:
                shoushi = pd.read_csv(fullpath, usecols=['频道号', '时长', '时间段'])
                pingdao = shoushi['频道号']
                features = [v for v in pingdao if v in pindaohao2list]
                if len(features) > 0:
                    for w in features:
                        rows[w] = w
                        if w not in feature_columns:
                            feature_columns.append(w)
                shoushi_list.append(rows)

    msg_list2df = pd.DataFrame(shoushi_list)
    output = './datasets/tv_data/shoushionehot.csv'
    df_ohe = pd.get_dummies(msg_list2df, columns=feature_columns, dummy_na=False)
    df_ohe.to_csv(output, index=None, encoding='utf-8')


def get_dianbodanpinonehot():
    pindaohao = pd.read_csv("./datasets/tv_data/danpiandianbo.csv", usecols=['二级目录']).values
    pindaohao2list = [i for j in pindaohao for i in j]

    path = './datasets/tv_data/results/'
    shoushi_list = []
    feature_columns = []

    for dirpath, dirnames, filenames in (os.walk(path)):
        for file in tqdm(filenames):
            rows = {}
            fullpath = os.path.join(dirpath, file)
            userid = fullpath.split("/")[-2]
            rows['userid'] = userid
            if "danpian" in fullpath:
                huikan = pd.read_csv(fullpath, usecols=['二级目录', '用户号'])
                pingdao = huikan['二级目录']
                features = [v for v in pingdao if v in pindaohao2list]
                if len(features) > 0:
                    for w in features:
                        rows[w] = w
                        if w not in feature_columns:
                            feature_columns.append(w)
                shoushi_list.append(rows)

    msg_list2df = pd.DataFrame(shoushi_list)
    output = './datasets/tv_data/danpian_onehot.csv'
    df_ohe = pd.get_dummies(msg_list2df, columns=feature_columns, dummy_na=False)
    df_ohe.to_csv(output, index=None, encoding='utf-8')


def classifiy_user():
    from sklearn.cluster import KMeans
    clusters_number = 10
    output = './datasets/tv_data/shoushionehot.csv'
    data = pd.read_csv(output)
    user = data['userid']
    del data['userid']
    X = data.values
    k_means = KMeans(n_clusters=clusters_number, init='k-means++', n_init=10,
                     max_iter=1000, tol=1e-4, precompute_distances='auto',
                     verbose=0, random_state=None, copy_x=True,
                     n_jobs=1, algorithm='auto')
    k_means.fit(X)
    labels = k_means.labels_
    rows = {'user': user, "label": labels}

    df = pd.DataFrame(rows)
    df.to_csv("./datasets/tv_data/kmeans_labels.csv", index=None)


def labels_group():
    path = "./datasets/tv_data/kmeans_labels.csv"
    data = pd.read_csv(path)
    datagroup = data.groupby(by='label')
    for x, y in datagroup:
        del y['label']
        y.to_csv("./datasets/tv_data/cluseter/{}.csv".format(x), index=None)


def replaces(inputs):
    try:
        return str(inputs).replace("\\", "/").split("/")[-1]
    except:
        return inputs


def make_gifts():
    chanpin = pd.read_csv("./datasets/tv_data/chanpinxinxi.csv", encoding='utf-8',
                          usecols=['正题名', '内容描述', '分类名称', '连续剧分类'])
    chanpincopy = chanpin.copy()
    chanpincopy['分类名称.1'] = chanpin['分类名称'].apply(replaces)
    for x, y in chanpincopy.groupby(by='分类名称.1'):
        print(x)


if __name__ == '__main__':
    method = "make_gifts"

    if method == 'compute_time_cost':
        compute_time_cost()

    if method == 'make_huikan':
        make_huikan()

    if method == 'group_user':
        group_user()

    if method == 'get_shoushionehot':
        get_shoushionehot()

    if method == 'get_dianbodanpinonehot':
        get_dianbodanpinonehot()

    if method == 'classifiy_user':
        classifiy_user()

    if method == 'labels_group':
        labels_group()

    if method == 'make_gifts':
        make_gifts()
