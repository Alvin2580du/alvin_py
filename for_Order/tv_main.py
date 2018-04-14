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
from tqdm import tqdm, trange
import numpy as np
import re
import random


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
        return "下午"
    elif compare_time(x2date, two, three):
        return "晚上"
    else:
        return "深夜"


def convert(inputs):
    try:
        out = datetime.strptime(inputs, '%Y/%m/%d %H:%M')
    except:
        out = datetime.strptime(inputs, '%Y/%m/%d')
    return out


def timecost(starttime, endtime):
    t1 = convert(starttime)
    t2 = convert(endtime)
    return (t1 - t2).seconds / 3600.


data = pd.read_csv("./datasets/tv_data/jibenxinxi.csv", usecols=['用户号', '套餐', '机顶盒编号'])

taocan2list = list(set(data['套餐'].values.tolist()))
taocan2list.remove(np.nan)

y2j = {}  # 机顶盒号:用户号
for one in data.values:
    y2j[one[2]] = one[0]

y2taocan = {}  # 用户号: 套餐
for one in data.values:
    y2taocan[one[0]] = one[1]

y2j = {}  # 用户号：机顶盒号
for one in data.values:
    y2j[one[0]] = one[2]

j2c = {}  # 机顶盒号:套餐
for one in data.values:
    j2c[one[2]] = one[1]


def get_taocan(userid):
    return y2taocan[userid]


def get_userid(jidinghehao):
    return y2j[jidinghehao]


def get_jidinghehao(userid):
    return y2j[userid]


def get_taocan_by_jidinghe(jidinghe):
    return j2c[jidinghe]


hao2mingdata = pd.read_csv("./datasets/tv_data/shoushi.csv", usecols=['频道号', '频道名'])
pindao2list = hao2mingdata['频道名'].values
datas = {}
for one in hao2mingdata.values:
    datas[one[1]] = one[0]


def hao2ming(hao):
    return datas[hao]


def make_shoushi():
    shoushi = pd.read_csv("./datasets/tv_data/shoushi.csv", usecols=['机顶盒设备号', '频道名', '收看开始时间', '收看结束时间'])
    shoushicopy = pd.DataFrame()
    shoushicopy['timecost_shoushi'] = shoushi.apply(lambda row: timecost(row['收看开始时间'], row['收看结束时间']), axis=1)
    shoushicopy['timerange_shoushi'] = shoushi['收看开始时间'].apply(gettimerange)
    shoushicopy['userid'] = shoushi['机顶盒设备号'].apply(get_userid)
    shoushicopy['pindaoming_shoushi'] = shoushi['频道名']
    shoushicopy['taocan'] = shoushi['机顶盒设备号'].apply(get_taocan_by_jidinghe)
    shoushicopy.to_csv("./datasets/tv_data/shoushiNew.csv", index=None)


def make_huikan():
    huikan = pd.read_csv("./datasets/tv_data/huikan.csv", usecols=['用户号', '频道', '回看时长.时.', '回看开始时间', '回看结束时间'])
    huikancopy = pd.DataFrame()
    huikancopy['pindaoming_huikan'] = huikan['频道']
    huikancopy['timecost_huikan'] = huikan.apply(lambda row: timecost(row['回看开始时间'], row['回看结束时间']), axis=1)
    huikancopy['timerange_huikan'] = huikan['回看开始时间'].apply(gettimerange)
    huikancopy['userid'] = huikan['用户号']
    huikancopy.to_csv("./datasets/tv_data/huikanNew.csv", index=None)


def make_dianbo():
    dianbo = pd.read_csv("./datasets/tv_data/dianbo.csv", usecols=['用户号', '节目名称', '点播金额.元.', '二级目录'])
    dianbocopy = pd.DataFrame()
    dianbocopy['jiemumingchen_dianbo'] = dianbo['节目名称']
    dianbocopy['jine_dianbo'] = dianbo['点播金额.元.']
    dianbocopy['erjimulu_dianbo'] = dianbo['二级目录']
    dianbocopy['userid'] = dianbo['用户号']
    dianbocopy.to_csv("./datasets/tv_data/dianbo_New.csv", index=None)


def make_danpiandianbo():
    danpiandianbo = pd.read_csv("./datasets/tv_data/danpiandianbo.csv",
                                usecols=['用户号', '影片名称', '二级目录', '观看开始时间', '观看结束时间'])
    danpiandianbocopy = pd.DataFrame()
    danpiandianbocopy['userid'] = danpiandianbo['用户号']
    danpiandianbocopy['timecost_danpiandianbo'] = danpiandianbo.apply(
        lambda row: timecost(row['观看开始时间'], row['观看结束时间']), axis=1)
    danpiandianbocopy['erjimulu_danpiandianbo'] = danpiandianbo['二级目录']
    danpiandianbocopy['yingpianmingcheng_danpiandianbo'] = danpiandianbo['影片名称']
    danpiandianbocopy.to_csv("./datasets/tv_data/danpiandianboNew.csv", index=None)


def group_user():
    shoushi = pd.read_csv("./datasets/tv_data/shoushiNew.csv")
    huikan = pd.read_csv("./datasets/tv_data/huikanNew.csv")
    dianbo = pd.read_csv("./datasets/tv_data/dianbo_New.csv")
    danpiandianbo = pd.read_csv("./datasets/tv_data/danpiandianboNew.csv")

    shoushigroup = shoushi.groupby(by='userid')
    huikangroup = huikan.groupby(by='userid')
    dianbogroup = dianbo.groupby(by='userid')
    danpiandianbogroup = danpiandianbo.groupby(by='userid')

    for x, y in shoushigroup:
        save_path = os.path.join("./datasets/tv_data/groupbyuserid", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_shoushi.csv".format(x))
        y.to_csv(save_name, index=None)

    for x, y in huikangroup:
        save_path = os.path.join("./datasets/tv_data/groupbyuserid", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_huikan.csv".format(x))
        y.to_csv(save_name, index=None)

    for x, y in dianbogroup:
        save_path = os.path.join("./datasets/tv_data/groupbyuserid", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_dianbo.csv".format(x))
        y.to_csv(save_name, index=None)

    for x, y in danpiandianbogroup:
        save_path = os.path.join("./datasets/tv_data/groupbyuserid", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_dianbodanpian.csv".format(x))
        y.to_csv(save_name, index=None)


def get_numbers_of_time(inputs):
    a, b, c = 0, 0, 0
    for x in inputs:
        if x == '深夜':
            c += 1
        elif x == '晚上':
            b += 1
        else:
            a += 1

    return a, b, c


def get_sata_time(inputs):
    a, b, c = inputs.mean(), inputs.max(), inputs.min()
    return a, b, c


def get_stat_jine(inputs):
    return inputs.mean(), inputs.max(), inputs.min()


yingpianmingcheng_danpiandianbo = pd.read_csv("./datasets/tv_data/danpiandianboNew.csv",
                                              usecols=['yingpianmingcheng_danpiandianbo'])
yingpianmingcheng2list = [i for j in yingpianmingcheng_danpiandianbo.values for i in j]

jiemumingcheng = pd.read_csv("./datasets/tv_data/dianbo_New.csv",
                             usecols=['jiemumingchen_dianbo'])
jiemumingcheng2list = [i for j in jiemumingcheng.values for i in j]


def get_onehot():
    """
    先提取下列特征，得到一个大矩阵，计算每个用户的相似度，然后利用协同过滤，找到最相似的用户， 然后给对应的用户的推荐最相似用户的观看记录。并根据对应节目的文本相似度， 推荐相关产品。
    
    下面是特征：
    
    userid, 所有的频道onehot-深夜(收视、回看、点播)-晚上-下午-时间均值(收视、回看、点播)-时间最大值-时间最小值-套餐编号，点播的节目名称onehot，回看的频道名称onehot。
    
    11111, 0, 1，个数，个数，个数， val, val, val, 0,1,0
    :return: 
    """
    path = './datasets/tv_data/groupbyuserid/'
    shoushi_list = []
    feature_columns = []
    for dirpath, dirnames, filenames in tqdm(os.walk(path)):
        for file in filenames:
            fullpath = os.path.join(dirpath, file)
            userid = fullpath.split("/")[-1].split('\\')[0]
            rows = {'userid': userid}

            if "shoushi" in fullpath:
                shoushi = pd.read_csv(fullpath)

                a, b, c = get_numbers_of_time(shoushi['timerange_shoushi'])  # 下午，晚上，深夜
                t_avg, t_max, t_min = get_sata_time(shoushi['timecost_shoushi'])
                rows['xiawu_shoushi'] = a
                rows['wans_shoushi'] = b
                rows['shenye_shoushi'] = c
                rows['t_avg_shoushi'] = t_avg
                rows['t_max_shoushi'] = t_max
                rows['t_min_shoushi'] = t_min

                pingdao = shoushi['pindaoming_shoushi'].values
                pingdaofeatures = [v for v in pingdao if v in pindao2list]
                if len(pingdaofeatures) > 0:
                    for w1 in pingdaofeatures:
                        rows[w1] = w1
                        if w1 not in feature_columns:
                            feature_columns.append(w1)

                taocan = shoushi['taocan'].values
                taocanfeatures = [v for v in taocan if v in taocan2list]
                if len(taocanfeatures) > 0:
                    for w2 in taocanfeatures:
                        rows[w2] = w2
                        if w2 not in feature_columns:
                            feature_columns.append(w2)

            if "danpian" in fullpath:
                # userid, timecost_danpiandianbo, erjimulu_danpiandianbo, yingpianmingcheng_danpiandianbo
                danpian = pd.read_csv(fullpath)
                t_avg, t_max, t_min = get_sata_time(danpian['timecost_danpiandianbo'])
                rows['t_avg_danpian'] = t_avg
                rows['t_max_danpian'] = t_max
                rows['t_min_danpian'] = t_min

                yingpianmingcheng = danpian['yingpianmingcheng_danpiandianbo'].values
                yingpianmingchengfeatures = [v for v in yingpianmingcheng if v in yingpianmingcheng2list]
                if len(yingpianmingchengfeatures) > 0:
                    for w1 in yingpianmingchengfeatures:
                        rows[w1] = w1
                        if w1 not in feature_columns:
                            feature_columns.append(w1)

            if "huikan" in fullpath:
                # pindaoming_huikan, timecost_huikan, timerange_huikan, userid
                huikan = pd.read_csv(fullpath)
                a, b, c = get_numbers_of_time(huikan['timerange_huikan'])  # 下午，晚上，深夜
                t_avg, t_max, t_min = get_sata_time(huikan['timecost_huikan'])
                rows['xiawu_huikan'] = a
                rows['wans_huikan'] = b
                rows['shenye_huikan'] = c
                rows['t_avg_huikan'] = t_avg
                rows['t_max_huikan'] = t_max
                rows['t_min_huikan'] = t_min

                pindaoming_huikan = huikan['pindaoming_huikan'].values
                pindaoming_huikanfeatures = [v for v in pindaoming_huikan if v in pindao2list]
                if len(pindaoming_huikanfeatures) > 0:
                    for w1 in pindaoming_huikanfeatures:
                        rows[w1] = w1
                        if w1 not in feature_columns:
                            feature_columns.append(w1)

            if 'dianbo' in fullpath and len(fullpath.split("\\")[-1]) < 16:
                dianbo = pd.read_csv(fullpath)
                a, b, c = get_stat_jine(dianbo['jine_dianbo'])
                rows['jine_avg'] = a
                rows['jine_max'] = b
                rows['jine_min'] = c

                jiemumingchen_dianbo = dianbo['jiemumingchen_dianbo'].values
                jiemumingchen_dianbofeatures = [v for v in jiemumingchen_dianbo if v in jiemumingcheng2list]
                if len(jiemumingchen_dianbofeatures) > 0:
                    for w1 in jiemumingchen_dianbofeatures:
                        rows[w1] = w1
                        if w1 not in feature_columns:
                            feature_columns.append(w1)

            shoushi_list.append(rows)
    msg_list2df = pd.DataFrame(shoushi_list)
    output = './datasets/tv_data/shoushionehot.csv'
    df_ohe = pd.get_dummies(msg_list2df, columns=feature_columns, dummy_na=False)
    df_ohe.to_csv(output, index=None, encoding='utf-8')


def replaces_digits(inputs):
    out = re.sub('[0-9]', "",
                 str(inputs).replace("(", "").replace(")", "").replace(" ", "").strip().replace("月", "").replace("日",
                                                                                                                 ""))
    return out


def make_chanpin():
    data = pd.read_csv("./datasets/tv_data/chanpinxinxi.csv", usecols=['正题名', '内容描述', '连续剧分类', '分类名称'])
    data['产品名称'] = data['正题名'].apply(replaces_digits)
    del data['正题名']
    data = data.drop_duplicates()

    movies = pd.DataFrame()
    movies['MovieID'] = data['产品名称']
    movies['title'] = data['分类名称']
    movies['Genres'] = data['内容描述']


def titles_1(usrid):
    titles = ['收视偏好', '基本特征']
    return random.choice(titles)


def titles_2(usrid):
    titles_2 = ['电视剧', '电影', '娱乐', '语言']
    return random.choice(titles_2)


def titles_3(usrid):
    titles_3 = ['动作', '军旅片', '古装剧', '动画', '粤语', '语言', '综艺']
    return random.choice(titles_3)


def classifiy_user():
    from sklearn.cluster import KMeans, MiniBatchKMeans
    clusters_number = 3
    output = './datasets/tv_data/shoushionehot.csv'
    data = pd.read_csv(output)
    df = data.fillna(0)
    user = df['userid']
    del df['userid']
    X = df.values
    k_means = KMeans(n_clusters=clusters_number, init='k-means++', n_init=10,
                     max_iter=1000, tol=1e-4, precompute_distances='auto',
                     verbose=0, random_state=None, copy_x=True,
                     n_jobs=1, algorithm='auto')
    k_means = MiniBatchKMeans(n_clusters=clusters_number, init='k-means++', max_iter=100,
                              batch_size=100, verbose=0, compute_labels=True,
                              random_state=None, tol=0.0, max_no_improvement=10,
                              init_size=None, n_init=3, reassignment_ratio=0.01)

    k_means.fit(X)
    labels = k_means.labels_
    rows = {'user': user, "label": labels}

    df = pd.DataFrame(rows)
    df['一级标签'] = df['user'].apply(titles_1)
    df['二级标签'] = df['user'].apply(titles_2)
    df['三级标签'] = df['user'].apply(titles_3)

    df.to_csv("./datasets/tv_data/kmeans_labels.csv", index=None)


def get_smallest_n(dicts):
    res = sorted(dicts.items(), key=lambda x: x[1], reverse=False)
    limlit = 5
    out = []
    k = 1
    for one in res:
        k += 1
        out.append(one[0])
        if k > limlit:
            break
    return out


def build_first_question():
    # 输入一个userid，根据与他最相似的一个用户，然后找这个用户的历史观看记录，
    # 选出在产品信息中相同的产品，然后利用相似度，寻找类似的产品，给出评分。
    output = './datasets/tv_data/shoushionehot.csv'
    onehot = pd.read_csv(output)
    df = onehot.fillna(0)
    df = df.groupby(by=['userid']).sum()
    df['userid'] = df.index
    df.set_index('userid')
    mydis_ed = {}
    for index1, row1 in tqdm(df.iterrows()):
        dis_ed = {}
        for index2, row2 in df.iterrows():
            if index2 == index1:
                continue
            ed = np.sqrt(np.sum((df.loc[index1, :] - df.loc[index2, :]) ** 2))
            dis_ed[index2] = ed
        mydis_ed[index1] = get_smallest_n(dis_ed)
    mydis_ed = pd.DataFrame(mydis_ed)
    mydis_ed.to_csv("./datasets/distance.csv", index=None)


if __name__ == '__main__':
    method = "classifiy_user"

    if method == 'make_shoushi':
        make_shoushi()

    if method == 'make_huikan':
        make_huikan()

    if method == 'make_dianbo':
        make_dianbo()

    if method == 'make_danpiandianbo':
        make_danpiandianbo()

    if method == 'group_user':
        group_user()

    if method == 'get_onehot':
        get_onehot()

    if method == 'make_chanpin':
        make_chanpin()

    if method == 'classifiy_user':
        classifiy_user()

    if method == 'build_first_question':
        build_first_question()

    if method == 'get_smallest_n':
        d = {'a': 1, 'b': 4, 'c': 2, 'd': 3}
        get_smallest_n(d)
