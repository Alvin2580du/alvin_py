from collections import Counter
import time
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def time_conv(x):
    timeArray = time.localtime(x)
    otherStyleTime = time.strftime("%Y-%m-%d", timeArray)
    return otherStyleTime


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
        pos_features['id'] = posid
        pos_features['type'] = pos_action_type
        pos_features['time'] = pos_action_time
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


def split_data_v2(step='train'):
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
    pos_ids = []
    for posid in tqdm(data_pos):
        pos_features = {}
        pos_action_type = action[action['userid'].isin([posid])]['actionType']
        pos_action_time = action[action['userid'].isin([posid])]['actionTime']
        pos_features['id'] = posid
        pos_features['type'] = pos_action_type
        pos_features['time'] = pos_action_time
        df_pos_action = pd.DataFrame(pos_features)
        df_pos_action.to_csv("Order_predicts/datasets/results/{}/pos/{}.csv".format(step, posid), index=None)
        if posid not in pos_ids:
            pos_ids.append(posid)
    df = pd.DataFrame(pos_ids)
    df.to_csv('Order_predicts/datasets/results/posids.csv', index=None)


def get_neg_action_by_id(step='train'):
    ids = pd.read_csv('Order_predicts/datasets/results/posids.csv').values.tolist()
    ids = [j for i in ids for j in i]
    action = pd.read_csv("Order_predicts/datasets/{}/action_{}.csv".format(step, step))
    action_neg = action[~action.userid.isin(ids)]
    # action_neg.to_csv("Order_predicts/datasets/train/action_neg.csv", index=None)
    action_neg_grouped = action_neg.groupby(action_neg['userid'])

    for i, j in tqdm(action_neg_grouped):
        j.to_csv("Order_predicts/datasets/results/{}/neg/{}.csv".format(step, i), index=None)


def get_history_by_id(step='train'):
    ids = pd.read_csv('Order_predicts/datasets/results/posids.csv').values.tolist()
    ids = [j for i in ids for j in i]
    History = pd.read_csv("Order_predicts/datasets/{}/orderHistory_{}.csv".format(step, step))
    History_neg = History[~History.userid.isin(ids)]

    History_neg_grouped = History_neg.groupby(History_neg['userid'])
    for i, j in tqdm(History_neg_grouped):
        j.to_csv("Order_predicts/datasets/results/{}/History_neg/{}.csv".format(step, i), index=None)


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


def get_features(step='train'):
    pos_root = 'Order_predicts/datasets/results/{}/pos/'.format(step)
    neg_root = 'Order_predicts/datasets/results/{}/neg/'.format(step)
    if not os.path.exists(pos_root):
        os.makedirs(pos_root)
    if not os.path.exists(neg_root):
        os.makedirs(neg_root)

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
            if step == 'train':
                rows['label'] = 1 if base_name == 'pos' else 0
            else:
                rows['label'] = 0
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
            # rows['10_have_order'] = 1 if base_name == 'pos' else 0
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
        save_name = "Order_predicts/datasets/results/{}/{}_features.csv".format(step, base_name)
        if step == 'test':
            del df['label']
        df.to_csv(save_name, index=None)


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


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def plot_data():
    data = pd.read_csv("Order_predicts/datasets/results/train/pos/100000001023.csv", usecols=['type']).values.tolist()
    plt.figure()
    plt.plot(data)
    plt.show()


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
        """
        5 45859 0.29460436969607423
        1 34878 0.22406095218516925
        6 27804 0.1786166269441036
        3 16709 0.10734085813584474
        4 10935 0.07024790733828848
        2 8538 0.05484925769129466
        8 4630 0.02974374128726801
        7 3664 0.02353802766232181
        9 2646 0.01699825905963524
        
        ====================
        5 188205 0.3508067221878425
        1 150944 0.28135368281353684
        6 88711 0.16535381701870672
        3 31920 0.059497625314077374
        8 17331 0.032304302766863253
        2 16795 0.03130521983552411
        4 16167 0.030134652520447648
        7 15251 0.02842726452584568
        9 11168 0.020816713017155895


        """


def get_dumm():
    roots = ["Order_predicts/datasets/results/train/pos/", "Order_predicts/datasets/results/train/neg/"]
    for root in roots:
        step = root.split("/")[-2]
        for file in os.listdir(root):
            file_name = os.path.join(root, file)

            data = pd.read_csv(file_name)
            df1 = data.copy()
            df1['time'] = data['time'].apply(time_conv)
            df_grouped = df1.groupby(by='time')
            res = []
            for i, j in df_grouped:

                j2df = pd.get_dummies(j, columns=['type'])
                rows = {}
                rows['0_t1'] = j2df['type_1'].sum() if 'type_1' in j2df.columns else 0
                rows['1_t2'] = j2df['type_2'].sum() if 'type_2' in j2df.columns else 0
                rows['2_t3'] = j2df['type_3'].sum() if 'type_3' in j2df.columns else 0
                rows['3_t4'] = j2df['type_4'].sum() if 'type_4' in j2df.columns else 0
                rows['4_t5'] = j2df['type_5'].sum() if 'type_5' in j2df.columns else 0
                rows['5_t6'] = j2df['type_6'].sum() if 'type_6' in j2df.columns else 0
                rows['6_t7'] = j2df['type_7'].sum() if 'type_7' in j2df.columns else 0
                rows['7_t8'] = j2df['type_8'].sum() if 'type_8' in j2df.columns else 0
                rows['8_t9'] = j2df['type_9'].sum() if 'type_9' in j2df.columns else 0
                rows['9_time'] = i
                rows['10_id'] = j2df['id']
                res.append(rows)

            df = pd.DataFrame(res)
            df.to_csv("/home/duyp/ddddd/{}/{}.csv".format(step, file), index=None)


get_neg_action_by_id(step='train')

