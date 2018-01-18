import pandas as pd
import os
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from scipy.stats import mode
from pyduyp.utils.utils import time2day, time2mouth, time2week
from pyduyp.utils.utils import compute_interval_of_day
from pyduyp.utils.utils import get_freq_of_day_and_month, get_week_freq, get_type_freq
from pyduyp.logger.log import log

log.info("Start runing ...")


def get_pos_action_by_id(step='train'):
    # 根据历史订单,获得精品订单的用户id,并在行为表中分离出其行为数据
    pos_root = 'Order_predicts/datasets/results/{}/action_pos/'.format(step)
    if not os.path.exists(pos_root):
        os.makedirs(pos_root)

    action = pd.read_csv("Order_predicts/datasets/{}/action_{}.csv".format(step, step))
    orderHistory = pd.read_csv("Order_predicts/datasets/{}/orderHistory_{}.csv".format(step, step),
                               usecols=['userid', 'orderTime', 'orderType'])
    data_pos = orderHistory[orderHistory['orderType'].isin(['1'])]['userid'].values
    pos_ids = []
    for posid in tqdm(data_pos):
        pos_features = {}
        pos_action_type = action[action['userid'].isin([posid])]['actionType']
        pos_action_time = action[action['userid'].isin([posid])]['actionTime']
        pos_features['userid'] = posid
        pos_features['actionType'] = pos_action_type
        pos_features['actionTime'] = pos_action_time
        df_pos_action = pd.DataFrame(pos_features)
        df_pos_action.to_csv("Order_predicts/datasets/results/{}/action_pos/{}.csv".format(step, posid), index=None)
        if posid not in pos_ids:
            pos_ids.append(posid)
    df = pd.DataFrame(pos_ids)
    df.to_csv('Order_predicts/datasets/results/posids.csv', index=None)


def get_neg_action_by_id(step='train'):
    # 根据上一步得到的精品订单的用户id,得到不是精品订单用户的行为数据
    neg_root = 'Order_predicts/datasets/results/{}/action_neg/'.format(step)
    if not os.path.exists(neg_root):
        os.makedirs(neg_root)
    pos_root = 'Order_predicts/datasets/results/{}/action_pos/'.format(step)
    neg_root = 'Order_predicts/datasets/results/{}/action_neg/'.format(step)
    if not os.path.exists(pos_root):
        os.makedirs(pos_root)
    if not os.path.exists(neg_root):
        os.makedirs(neg_root)

    ids = pd.read_csv('Order_predicts/datasets/results/posids.csv').values.tolist()
    ids = [j for i in ids for j in i]
    action = pd.read_csv("Order_predicts/datasets/{}/action_{}.csv".format(step, step))
    action_neg = action[~action.userid.isin(ids)]
    # action_neg.to_csv("Order_predicts/datasets/train/action_neg.csv", index=None)
    action_neg_grouped = action_neg.groupby(action_neg['userid'])

    for i, j in tqdm(action_neg_grouped):
        if len(j) == 0:
            continue
        j.to_csv("Order_predicts/datasets/results/{}/action_neg/{}.csv".format(step, i), index=None)


def get_pos_history_by_id(step='train'):
    # 　根据id获取历史订单数据
    ids = pd.read_csv('Order_predicts/datasets/results/posids.csv').values.tolist()
    ids = [j for i in ids for j in i]
    History = pd.read_csv("Order_predicts/datasets/{}/orderHistory_{}.csv".format(step, step))
    History_neg = History[History.userid.isin(ids)]
    History_neg_grouped = History_neg.groupby(History_neg['userid'])
    for i, j in tqdm(History_neg_grouped):
        if len(j) == 0:
            continue
        j.to_csv("Order_predicts/datasets/results/{}/history_pos/{}.csv".format(step, i), index=None)


def get_neg_history_by_id(step='train'):
    # 　根据id获取历史订单数据
    ids = pd.read_csv('Order_predicts/datasets/results/posids.csv').values.tolist()
    ids = [j for i in ids for j in i]
    History = pd.read_csv("Order_predicts/datasets/{}/orderHistory_{}.csv".format(step, step))
    History_neg = History[~History.userid.isin(ids)]
    History_neg_grouped = History_neg.groupby(History_neg['userid'])
    for i, j in tqdm(History_neg_grouped):
        if len(j) == 0:
            continue
        j.to_csv("Order_predicts/datasets/results/{}/history_neg/{}.csv".format(step, i), index=None)


def get_history(step='train'):
    pos_root = 'Order_predicts/datasets/results/{}/history_pos/'.format(step)
    if not os.path.exists(pos_root):
        os.makedirs(pos_root)
    neg_root = 'Order_predicts/datasets/results/{}/history_neg/'.format(step)
    if not os.path.exists(neg_root):
        os.makedirs(neg_root)
    get_pos_history_by_id(step)
    get_neg_history_by_id(step)


def get_action_features(step='train'):
    pos_root = 'Order_predicts/datasets/results/{}/action_pos/'.format(step)
    neg_root = 'Order_predicts/datasets/results/{}/action_neg/'.format(step)
    if not os.path.exists(pos_root):
        os.makedirs(pos_root)
    if not os.path.exists(neg_root):
        os.makedirs(neg_root)

    for root in [pos_root, neg_root]:
        base_name = root.split("/")[-2]
        actions = []
        for file in tqdm(os.listdir(root)):
            data = pd.read_csv(os.path.join(root, file))
            data_types = data['actionType'].values.tolist()
            data_copy = data.copy()
            data_copy['time2days'] = data['actionTime'].apply(time2day)
            data_copy['time2mouth'] = data['actionTime'].apply(time2mouth)
            data_copy['time2week'] = data['actionTime'].apply(time2week)
            data_copy_grouped_day = data_copy.groupby(by='time2days')
            data_copy_grouped_month = data_copy.groupby(by='time2mouth')

            time_counts = compute_interval_of_day(data_copy)

            type_freq, types_sum = get_type_freq(data)  # 每个操作的总数，　总次数
            rows = OrderedDict()

            if step == 'train':
                rows['0_label'] = 1 if base_name == 'action_pos' else 0
            else:
                rows['0_label'] = 0
            aid = file.split(".")[0]
            rows['1_id'] = aid
            rows['2_t1'] = type_freq['2_t1']  # 类型1－9点击总数
            rows['3_t2'] = type_freq['3_t2']
            rows['4_t3'] = type_freq['4_t3']
            rows['5_t4'] = type_freq['5_t4']
            rows['6_t5'] = type_freq['6_t5']
            rows['7_t6'] = type_freq['7_t6']
            rows['8_t7'] = type_freq['8_t7']
            rows['9_t8'] = type_freq['9_t8']
            rows['10_t9'] = type_freq['10_t9']
            rows['11_rate1'] = type_freq['2_t1'] / types_sum  # 打开app的比例
            rows['12_rate9'] = type_freq['10_t9'] / types_sum  # 下单的比例
            rows['13_atmean'] = np.mean(time_counts)  # 时间均值
            rows['14_atstd'] = np.std((time_counts))  # 时间标准差
            rows['15_atmedian'] = np.median(time_counts)  # 时间中位数
            rows['16_tmode'] = mode(time_counts)  # 时间众数
            rows['17_atptp'] = np.max(time_counts) - np.min(time_counts) if len(time_counts) > 0 else 0  # 时间极差
            rows['18_atvar'] = np.var(time_counts)  # 时间方差
            rows['19_xishu'] = np.mean(time_counts) / np.std(time_counts) if len(time_counts) > 0 else 0  # 时间变异系数
            rows['20_lastmean'] = np.mean(time_counts[-1:-4])  # 最后三天间隔的均值
            rows['21_laststd'] = np.std(time_counts[-1:-4])  # 最后三天间隔的标准差
            rows['22_dayrate'] = get_freq_of_day_and_month(data_copy_grouped_day)  # 日均
            rows['23_monthrate'] = get_freq_of_day_and_month(data_copy_grouped_month)  # 月均
            rows['24_weekrate'] = get_week_freq(data_copy['time2week'].values)  # 周均
            rows['25_atmean'] = np.mean(data_types)  # 类型均值
            rows['26_atstd'] = np.std((data_types))  # 类型标准差
            rows['27_atmedian'] = np.median(data_types)  # 类型中位数
            rows['28_tmode'] = mode(data_types)  # 类型众数
            rows['29_atptp'] = np.max(data_types) - np.min(data_types) if len(data_types) > 0 else 0  # 类型极差
            rows['30_atvar'] = np.var(data_types)  # 类型方差
            rows['31_xishu'] = np.mean(data_types) / np.std(data_types) if len(data_types) > 0 else 0  # 类型变异系数
            rows['32_rate2'] = type_freq['3_t2'] / types_sum  # 2的比例
            rows['33_rate3'] = type_freq['4_t3'] / types_sum  # 3的比例
            rows['34_rate4'] = type_freq['5_t4'] / types_sum  # 4的比例
            rows['35_rate5'] = type_freq['6_t5'] / types_sum  # 5的比例
            rows['36_rate6'] = type_freq['7_t6'] / types_sum  # 6的比例
            rows['37_rate7'] = type_freq['8_t7'] / types_sum  # 7的比例
            rows['38_rate8'] = type_freq['9_t8'] / types_sum  # 8的比例

            actions.append(rows)

        df = pd.DataFrame(actions)
        df = df.round(7)
        df = df.round({'0_label': 0, '1_id': 0})
        save_name = "Order_predicts/datasets/results/{}/{}_features.csv".format(step, base_name)
        if step == 'test':
            del df['0_label']
        df.to_csv(save_name, index=None)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        exit(1)

    method = sys.argv[1]
    if method == 'first':
        get_pos_action_by_id()
    if method == 'second':
        get_neg_action_by_id()
    if method == 'third':
        get_history()
    if method == 'final':
        get_action_features()
