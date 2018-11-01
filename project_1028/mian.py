import pandas as pd

from collections import OrderedDict

zddf = pd.DataFrame({'对方账号': ['1102', '1102', '4102', '4192', '2012', '1111'],
                     '交易卡号': ['36369', '36369', '88569', '11023', '77898', '36369'],
                     '交易金额': ['1660', '1660', '14021', '0', '98569', '1660'],
                     '档案编号': ['111', '111', '111', '111', '111', '111'],
                     '交易类型名称': ['存现', '取现', '存现', '取现', '存现', '存现'],
                     '交易方式': ['工资', '网转', '网转', '网转', '工资', '网转'],
                     '及时余额': ['1102', '41', '1023', '4016', '131', '1125'],
                     '交易时间': ['2010-6', '2010-6', '2018-1', '2014-5', '2014-2', '2014-2']})

"""
存现工资次数（count if 对方账号&交易卡号&交易类型名称=存现&交易方式=工资）
存现网转次数（count if 对方账号&交易卡号&交易类型名称=存现&交易方式=网转）

取现工资次数（count if 对方账号&交易卡号&交易类型名称=取现&交易方式=工资）
取现网转次数（count if 对方账号&交易卡号&交易类型名称=取现&交易方式=网转）

存现工资总金额（sum if 对方账号&交易卡号&交易类型名称=存现&交易方式=工资）
存现网转总金额（sum if 对方账号&交易卡号&交易类型名称=存现&交易方式=网转）

取现工资总金额（sum if 对方账号&交易卡号&交易类型名称=取现&交易方式=工资）
取现网转总金额（sum if 对方账号&交易卡号&交易类型名称=取现&交易方式=网转）
"""


def combine(x1, x2):
    return "{}_{}".format(x1, x2)


def get_values(inputs):
    if len(inputs) == 0:
        return ""
    else:
        return inputs[0]


def add(x1, x2):
    return x1 + x2
print(zddf)
print(combine(zddf['对方账号'], zddf['交易卡号']))

exit(1)

save = []
zddf['对方账号_交易卡号'] = zddf.apply(lambda row: combine(row['对方账号'], row['交易卡号']), axis=1)
for x, y in zddf.groupby(by='对方账号_交易卡号'):
    rows = OrderedDict()
    rows['对方账号'] = y['对方账号'].values.tolist()[0]
    rows['交易卡号'] = y['交易卡号'].values.tolist()[0]
    rows['及时余额'] = y['及时余额'].values.tolist()[0]
    rows['交易金额'] = y['交易金额'].values.tolist()[0]
    rows['存现工资次数'] = y[y['交易类型名称'].isin(['存现']) & y['交易方式'].isin(['工资'])].shape[0]
    rows['存现网转次数'] = y[y['交易类型名称'].isin(['存现']) & y['交易方式'].isin(['网转'])].shape[0]

    rows['取现工资次数'] = y[y['交易类型名称'].isin(['取现']) & y['交易方式'].isin(['工资'])].shape[0]
    rows['取现网转次数'] = y[y['交易类型名称'].isin(['取现']) & y['交易方式'].isin(['网转'])].shape[0]

    rows['存现次数'] = y[y['交易类型名称'].isin(['存现'])].shape[0]
    rows['取现次数'] = y[y['交易类型名称'].isin(['取现'])].shape[0]

    rows['存现工资总金额'] = get_values(y[y['交易类型名称'].isin(['存现']) & y['交易方式'].isin(['工资'])]['交易金额'].values.tolist())
    rows['存现网转总金额'] = get_values(y[y['交易类型名称'].isin(['存现']) & y['交易方式'].isin(['网转'])]['交易金额'].values.tolist())

    rows['取现工资总金额'] = get_values(y[y['交易类型名称'].isin(['取现']) & y['交易方式'].isin(['工资'])]['交易金额'].values.tolist())
    rows['取现网转总金额'] = get_values(y[y['交易类型名称'].isin(['取现']) & y['交易方式'].isin(['网转'])]['交易金额'].values.tolist())

    save.append(rows)
    print(rows)
df = pd.DataFrame(save)
df['总交易次数'] = df.apply(lambda row: add(row['存现次数'], row['取现次数']), axis=1)
df.to_excel("result_.xlsx", sheet_name="Sheet1", index=None)
