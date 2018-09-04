import pandas as pd
from itertools import combinations, groupby
from collections import Counter

orders_full = pd.read_csv('./datasets/order_products__prior.csv')
orders = orders_full
orders = orders.set_index('order_id')['product_id'].rename('item_id')
print(orders.head())


# Returns frequency counts for items and item pairs
def freq(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else:
        return pd.Series(Counter(iterable)).rename("freq")


# Returns number of unique orders
def order_count(order_item):
    return len(set(order_item.index))


# Returns generator that yields item pairs, one at a time
def get_item_pairs(order_item):
    order_item = order_item.reset_index().as_matrix()
    for order_id, order_object in groupby(order_item, lambda x: x[0]):
        item_list = [item[1] for item in order_object]

        for item_pair in combinations(item_list, 2):
            yield item_pair


# Returns frequency and support associated with item
def merge_item_stats(item_pairs, item_stats):
    return (item_pairs.merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A',
                             right_index=True).merge(
        item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B',
        right_index=True))


# 计算与项目相关联的名称
def merge_item_name(rules, item_name):
    columns = ['itemA', 'itemB', 'freqAB', 'supportAB', 'freqA', 'supportA', 'freqB', 'supportB',
               'confidenceAtoB', 'confidenceBtoA', 'lift',
               'conviction', 'leverage'
               ]
    rules = (rules
             .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
             .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
    return rules[columns]


def association_rules(order_item, min_support):
    print("Starting order_item: {:22d}".format(len(order_item)))

    # 计算频繁项和支持度
    item_stats = freq(order_item).to_frame("freq")
    item_stats['support'] = item_stats['freq'] / order_count(order_item) * 100

    # 根据最低支持度过滤频繁项
    qualifying_items = item_stats[item_stats['support'] >= min_support].index
    order_item = order_item[order_item.isin(qualifying_items)]

    print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
    print("Remaining order_item: {:21d}".format(len(order_item)))

    # 过滤掉频率低于2的项
    order_size = freq(order_item.index)
    qualifying_orders = order_size[order_size >= 2].index
    order_item = order_item[order_item.index.isin(qualifying_orders)]

    print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
    print("Remaining order_item: {:21d}".format(len(order_item)))

    # 重新计算频繁项与支撑集
    item_stats = freq(order_item).to_frame("freq")
    item_stats['support'] = item_stats['freq'] / order_count(order_item) * 100

    # 构造项的生成器
    item_pair_gen = get_item_pairs(order_item)

    # 计算成对项的支撑度
    item_pairs = freq(item_pair_gen).to_frame("freqAB")
    item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100

    print("Item pairs: {:31d}".format(len(item_pairs)))

    # 过滤掉成对项小于最低支撑度的项
    item_pairs = item_pairs[item_pairs['supportAB'] >= min_support]

    print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))

    # 创建关联规则表，计算相关指标
    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    item_pairs = merge_item_stats(item_pairs, item_stats)

    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
    item_pairs['lift'] = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
    item_pairs['conviction'] = (1-item_pairs['supportA'])/(1-item_pairs['confidenceAtoB'])
    # leverage
    item_pairs['leverage'] = item_pairs['confidenceAtoB'] - item_pairs['supportA']*item_pairs['supportB']
    # 按照left排序
    return item_pairs.sort_values('lift', ascending=False)


rules = association_rules(orders, 0.01)
rules.to_csv("rules.csv", index=None)
# 用项的名字代替ID，并保存结果
item_name = pd.read_csv('./datasets/products.csv')
item_name = item_name.rename(columns={'product_id': 'item_id', 'product_name': 'item_name'})
rules_final = merge_item_name(rules, item_name).sort_values('lift', ascending=False)

res = rules_final.rename(columns={'itemA': 'antecedants',
                                  'itemB': 'consequents',
                                  'supportAB': 'support',
                                  'supportA': 'antecedent support',
                                  'supportB': 'consequent support',
                                  'confidenceAtoB': 'confidence',
                                  })

res.to_csv("results_pre.csv", index=None)
del res['freqAB']
del res['freqA']
del res['freqB']

res.to_csv("results.csv", index=None)

