from collections import Counter
import time
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


def fun1():
    data = pd.read_csv("Order_predicts/datasets/train/orderHistory_train.csv", encoding='utf-8')
    data_pos = data[data['orderType'].isin(['1'])]['userid']
    # data_pos.to_csv("Order_predicts/datasets/results/posids.csv", index=None)

    data_neg = data[data['orderType'].isin(['0'])]
    print((data_neg.shape, data_pos.shape))
    pos_city = data_pos['city']
    neg_city = data_neg['city']

    for x, y in Counter(pos_city.values).most_common(20):
        print(x, y)
    print("= " * 50)
    for x, y in Counter(neg_city.values).most_common(20):
        print(x, y)


def split_data_v1():
    pos_root = 'Order_predicts/datasets/results/pos/'
    neg_root = 'Order_predicts/datasets/results/neg/'
    if not os.path.exists(pos_root):
        os.makedirs(pos_root)
    if not os.path.exists(neg_root):
        os.makedirs(neg_root)
    profile = pd.read_csv("Order_predicts/datasets/train/userProfile_train.csv")
    action = pd.read_csv("Order_predicts/datasets/train/action_train.csv")
    orderHistory = pd.read_csv("Order_predicts/datasets/train/orderHistory_train.csv",
                               usecols=['userid', 'orderTime', 'orderType'])
    data_pos = orderHistory[orderHistory['orderType'].isin(['1'])]['userid'].values
    data_neg = orderHistory[orderHistory['orderType'].isin(['0'])]['userid'].values
    for d in [data_pos, data_neg]:
        for ids in tqdm(d):
            g = profile[profile['userid'].isin([ids])]['gender'].values.tolist()[0]
            age = profile[profile['userid'].isin([ids])]['age'].values.tolist()[0]
            province = profile[profile['userid'].isin([ids])]['province'].values.tolist()[0]

            pos_action_type = action[action['userid'].isin([ids])]['actionType'].values.tolist()
            pos_action_time = action[action['userid'].isin([ids])]['actionTime'].values.tolist()
            length = len(pos_action_time)

            df = pd.DataFrame()
            for i in range(length):
                df.loc[i, 'id'] = "{}".format(ids)
                df.loc[i, 'type'] = pos_action_type[i]
                df.loc[i, 'time'] = pos_action_time[i]
                df.loc[i, 'gender'] = g if isinstance(g, str) else "未知"
                df.loc[i, 'age'] = age if isinstance(age, str) else "未知"
                df.loc[i, 'province'] = province if isinstance(province, str) else "未知"
                df.to_csv("Order_predicts/datasets/results/pos/{}.csv".format(ids), index=None)


def split_data(step='train'):
    pos_root = 'Order_predicts/datasets/results/{}/pos/'.format(step)
    neg_root = 'Order_predicts/datasets/results/{}/neg/'.format(step)
    if not os.path.exists(pos_root):
        os.makedirs(pos_root)
    if not os.path.exists(neg_root):
        os.makedirs(neg_root)
    action = pd.read_csv("Order_predicts/datasets/{}/action_{}.csv".format(step, step))
    orderHistory = pd.read_csv("Order_predicts/datasets/{}/orderHistory_{}.csv".format(step, step),
                               usecols=['userid', 'orderTime', 'orderType'])
    data_pos = orderHistory[orderHistory['orderType'].isin(['1'])]['userid'].values
    data_neg = orderHistory[orderHistory['orderType'].isin(['0'])]['userid'].values

    for posid in tqdm(data_pos):
        pos_features = {}
        pos_action_type = action[action['userid'].isin([posid])]['actionType']
        pos_action_time = action[action['userid'].isin([posid])]['actionTime']
        pos_features['userid'] = posid
        pos_features['type'] = pos_action_type
        pos_features['actionTime'] = pos_action_time
        df_pos_action = pd.DataFrame(pos_features)
        df_pos_action.to_csv("Order_predicts/datasets/results/{}/pos/{}.csv".format(step, posid), index=None)

    for negid in tqdm(data_neg):
        neg_actions = {}
        pos_action_type = action[action['userid'].isin([negid])]['actionType']
        pos_action_time = action[action['userid'].isin([negid])]['actionTime']
        neg_actions['id'] = negid
        neg_actions['type'] = pos_action_type
        neg_actions['time'] = pos_action_time
        neg_actions = pd.DataFrame(neg_actions)
        neg_actions.to_csv("Order_predicts/datasets/results/{}/neg/{}.csv".format(step, negid), index=None)


def fun_yc():
    da1 = pd.read_csv("Order_predicts/datasets/other/pos_features.csv", dtype=np.float32)
    da2 = pd.read_csv("Order_predicts/datasets/other/neg_features.csv", dtype=np.float32)
    y1 = da1['10_have_order']
    y2 = da2['10_have_order']
    y = pd.concat([y1, y2])
    del da1['10_have_order']
    del da2['10_have_order']
    del da1['0_id']
    del da2['0_id']
    x = pd.concat([da1, da2])
    x['label'] = y
    x.to_csv("Order_predicts/datasets/other/train.csv", index=None)


def fun3():
    types = []
    for file in os.listdir("Order_predicts/datasets/results/train/pos"):
        data_dir = os.path.join("Order_predicts/datasets/results/train/pos", file)
        data = pd.read_csv(data_dir, usecols=['type']).values.tolist()
        for x in data:
            types.append(x[0])

    c = Counter(types)
    c_sum = sum(c.values())
    for x, y in c.most_common(10):
        print(x, y, y / c_sum)


def order_future():
    data = pd.read_csv("Order_predicts/datasets/train/orderFuture_train.csv")
    uid_1 = data[data['orderType'].isin(['1'])]['userid']
    uid_0 = data[data['orderType'].isin(['0'])]['userid']


def get_freq_of_day_and_month(df):
    res = []
    for d, j in df:
        res.append(len(j))
    return sum(res) / len(res)


def fun2():
    from pyduyp.utils.utils import time2day, time2mouth, get_type_freq
    from collections import OrderedDict
    data = pd.read_csv('./datasets/results/train/action_pos/100000001023.csv')
    types = data['actionType'].values
    print(types)


def fun4():
    # city,country,continent
    userprofile = pd.read_csv("./datasets/train/userProfile_train.csv", usecols=['age']).values
    userprofile2list = userprofile.tolist()
    res = [j for i in userprofile2list for j in i]
    from collections import Counter
    fre = Counter(res)
    sumfre = sum(fre.values())
    out = {}
    for x, y in fre.most_common(1000):
        if isinstance(x, str):
            out[x] = "{:0.6f}".format(y / sumfre)

    print(out)


def fun5():
    data = pd.read_csv("./datasets/test/orderFuture_test.csv")
    length = len(data)
    rand = np.random.random_sample(size=1000000).tolist()

    res = []
    for x in rand:
        if 0 < x < 1:
            res.append(x)
        if len(res) > length:
            break

    data_new = data.copy()
    data_new['orderType'] = pd.Series(res)
    data_new = data_new.round(6)
    data_new.to_csv("./datasets/results.csv", index=None, columns=['userid', 'orderType'])


fun5()
