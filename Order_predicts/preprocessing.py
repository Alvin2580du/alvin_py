from collections import Counter
import time
import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def time_conv(x):
    timeArray = time.localtime(x)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


def fun1():
    data = pd.read_csv("Order_predicts/datasets/trainingset/orderHistory_train.csv", encoding='utf-8')
    data_pos = data[data['orderType'].isin(['1'])]
    data_neg = data[data['orderType'].isin(['0'])]
    print((data_neg.shape, data_pos.shape))
    pos_city = data_pos['city']
    neg_city = data_neg['city']

    for x, y in Counter(pos_city.values).most_common(20):
        print(x, y)
    print("= " * 50)
    for x, y in Counter(neg_city.values).most_common(20):
        print(x, y)


def split_data():
    action = pd.read_csv("Order_predicts/datasets/trainingset/action_train.csv")
    orderHistory = pd.read_csv("Order_predicts/datasets/trainingset/orderHistory_train.csv",
                               usecols=['userid', 'orderTime', 'orderType'])
    data_pos = orderHistory[orderHistory['orderType'].isin(['1'])]['userid'].values
    data_neg = orderHistory[orderHistory['orderType'].isin(['0'])]['userid'].values

    for posid in data_pos:
        pos_features = {}
        pos_action_type = action[action['userid'].isin([posid])]['actionType']
        pos_action_time = action[action['userid'].isin([posid])]['actionTime']
        pos_features['id'] = posid
        pos_features['type'] = pos_action_type
        pos_features['time'] = pos_action_time
        df_pos_action = pd.DataFrame(pos_features)
        df_pos_action.to_csv("Order_predicts/datasets/results/pos/{}.csv".format(posid), index=None)

    for negid in data_neg:
        neg_actions = {}
        pos_action_type = action[action['userid'].isin([negid])]['actionType']
        pos_action_time = action[action['userid'].isin([negid])]['actionTime']
        neg_actions['id'] = negid
        neg_actions['type'] = pos_action_type
        neg_actions['time'] = pos_action_time
        neg_actions = pd.DataFrame(neg_actions)
        neg_actions.to_csv("Order_predicts/datasets/results/neg/{}.csv".format(negid), index=None)


def compute_time_feature(time_list):
    timelist2sort = sorted(time_list)
    tmax = np.max(timelist2sort)
    tmin = np.min(timelist2sort)
    new = timelist2sort - tmin
    mean = np.mean(new)
    std = np.std(new)

    cha = np.inf
    cha_list = []
    length = len(timelist2sort)
    for i in range(length):
        if i + 1 < length:
            t = np.abs(int(timelist2sort[i]) - int(timelist2sort[i + 1]))
            cha_list.append(t)
            if t < cha:
                cha = t
    if len(cha_list) > 4:
        x1, x2, x3, x4 = cha_list[-1], cha_list[-2], cha_list[-3], cha_list[-4]
        lastthreemean = np.mean([x1, x2, x3])
        lastthreestd = np.std([x1, x2, x3])
    else:
        x1, x2, x3, x4 = 0, 0, 0, 0
        lastthreemean, lastthreestd = 0, 0
    return mean, std, cha, x1, x2, x3, x4, lastthreemean, lastthreestd


def compute_type_feature(test_da):
    c = Counter(test_da)
    values_sum = sum(c.values())
    rates = {}
    for x, y in c.items():
        rate = y / values_sum
        rates[x] = rate
    return rates


def get_features():
    pos_root = 'Order_predicts/datasets/results/pos/'
    neg_root = 'Order_predicts/datasets/results/neg/'
    for root in [pos_root, neg_root]:
        base_name = root.split("/")[-2]
        res = []
        for file in tqdm(os.listdir(root)):
            rows = {}
            aid = file.split(".")[0]
            file_name = os.path.join(root, file)
            data = pd.read_csv(file_name)
            atime = data['time'].values
            atype = data['type'].values
            mean, std, cha, x1, x2, x3, x4, lastthreemean, lastthreestd = compute_time_feature(atime)
            rates = compute_type_feature(atype)
            rows['0_id'] = aid
            rows['1_atmean'] = mean
            rows['2_atstd'] = std
            rows['3_atcha'] = cha
            rows['4_tlast'] = x1
            rows['5_t2'] = x2
            rows['6_t3'] = x3
            rows['7_t4'] = x4
            rows['8_lastmean'] = lastthreemean
            rows['9_laststd'] = lastthreestd
            rows['10_have_order'] = 1 if base_name == 'pos' else 0
            rows['11_rate1'] = rates[1] if 1 in rates else 0
            rows['12_rate2'] = rates[2] if 2 in rates else 0
            rows['13_rate3'] = rates[3] if 3 in rates else 0
            rows['14_rate4'] = rates[4] if 4 in rates else 0
            rows['15_rate5'] = rates[5] if 5 in rates else 0
            rows['16_rate6'] = rates[6] if 6 in rates else 0
            rows['17_rate7'] = rates[7] if 7 in rates else 0
            rows['18_rate8'] = rates[8] if 8 in rates else 0
            rows['19_rate9'] = rates[9] if 9 in rates else 0
            res.append(rows)
        df = pd.DataFrame(res)
        df.to_csv("Order_predicts/datasets/results/{}_features.csv".format(base_name), index=None)


