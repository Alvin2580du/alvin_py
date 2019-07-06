import os
from collections import OrderedDict
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

gdp_rows2018 = {'广东省': 97277.77, '江苏省': 92595.4, '山东省': 76469.7, '浙江省': 56197.0,
                '河南省': 48055.86, '四川省': 40678.13, '湖北省': 39366.55, '湖南省': 36425.78,
                '河北省': 36010.3, '福建省': 35804.04, '上海市': 32679.87, '北京市': 30320.0,
                '安徽省': 30006.8, '辽宁省': 25315.4, '陕西省': 24438.32, '江西省': 21984.78,
                '重庆市': 20363.19, '广西壮族自治区': 20352.51, '天津市': 18809.64, '云南省': 17811.12,
                '内蒙古自治区': 17289.2, '黑龙江省': 15902.68, '山西省': 16818.11, '吉林省': 14944.53,
                '贵州省': 14806.45, '新疆维吾尔自治区': 10881.96, '甘肃省': 8246.1, '海南省': 4832.05,
                '宁夏回族自治区': 3705.18, '青海省': 2865.23, '西藏': 1310.92}

gdp_rows2017 = {'广东省': 89705.23, '江苏省': 85869.76, '山东省': 72634.15, '浙江省': 51768.26,
                '河南省': 44552.83, '四川省': 36980.22, '湖北省': 35478.09, '湖南省': 33902.96,
                '河北省': 34016.32, '福建省': 32182.09, '上海市': 30632.99, '北京市': 28014.94,
                '安徽省': 27018.0, '辽宁省': 23409.24, '陕西省': 21898.81, '江西省': 20006.31,
                '重庆市': 19424.73, '广西壮族自治区': 18523.26, '天津市': 18549.19, '云南省': 16376.34,
                '内蒙古自治区': 16096.21, '黑龙江省': 15902.67, '山西省': 15528.42, '吉林省': 14944.53,
                '贵州省': 13540.83, '新疆维吾尔自治区': 10881.96, '甘肃省': 7459.9, '海南省': 4462.54,
                '宁夏回族自治区': 3443.56, '青海省': 2624.83, '西藏': 1310.92}


def is_changsheng(inputs):
    if "长春长生生物科技有限责任公司" in inputs:
        return '1'
    return '0'


def is_kuanguan(inputs):
    if '狂犬病' in inputs:
        return '1'
    return '0'


def get_price(inputs):
    try:
        res = re.match('\d+', inputs).group()
        return int(res)
    except:
        return 0


def task_one():
    data_path = './datasets'
    dataAll = []
    for file in os.listdir(data_path):
        file_name = os.path.join(data_path, file)
        if 'jpg' in file_name:
            continue
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                line_sp = line.split(',')
                if 'name' in line:
                    continue
                try:
                    if len(line_sp[4]) > 8:
                        continue
                    rows = OrderedDict()
                    rows['name'] = line_sp[0]
                    rows['src'] = line_sp[1]
                    rows['create_company'] = line_sp[2]
                    rows['report_company'] = line_sp[3]
                    rows['prov'] = line_sp[4]
                    rows['year'] = line_sp[5]
                    rows['price'] = line_sp[6]
                    dataAll.append(rows)
                except Exception as e:
                    continue

    df = pd.DataFrame(dataAll)
    df['changsheng'] = df['report_company'].apply(is_changsheng)
    df['price_int'] = df['price'].apply(get_price)
    df['kuanguan'] = df['name'].apply(is_kuanguan)
    df.to_excel("dataAll.xlsx", index=None)
    kuanguan = df[df['kuanguan'].isin(['1'])]
    kuanguan.to_excel("kuanguan.xlsx", index=None)

    shichang = {}
    for prov, y in df.groupby(by='prov'):
        rates = {}
        for changshen, y1 in y.groupby(by='changsheng'):
            rates[changshen] = y1['price_int'].sum()

        if '1' not in rates.keys():
            shichang[prov] = 0
        elif '0' not in rates.keys():
            shichang[prov] = 1
        else:
            shichang[prov] = rates['1'] / rates['0']

    plt.figure(figsize=(20, 10))
    plt.bar(list(shichang.keys()), list(shichang.values()), width=0.5)
    plt.xticks(rotation=36)
    plt.title("长春长生在疫苗行业的市场份额条形图")
    plt.savefig("长春长生在疫苗行业的市场份额条形图.png")
    plt.close()


def task_two():
    data = pd.read_excel("kuanguan.xlsx")
    save = []

    for x, y in data.groupby(by='prov'):
        for x1, y1 in y.groupby(by='create_company'):
            prices = y['price_int'].sum()
            rows = {}
            if '长春长生生物科技有限责任公司' in x1:
                rows['create_company'] = 1
            else:
                rows['create_company'] = 0
            rows['prov'] = x
            rows['prices'] = prices
            save.append(rows)

    df = pd.DataFrame(save).drop_duplicates()
    x_train, x_test, y_train, y_test = train_test_split(df['prices'].values.reshape(-1, 1),
                                                        df['create_company'].values.reshape(-1, 1),
                                                        test_size=0.3)
    print(x_train.shape, y_train.shape)

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    print(lr.predict(x_test))
    print(y_test)


def task_three():
    data = pd.read_excel("dataAll.xlsx")
    name_price = []
    xticklabels = []
    gdps = []
    for prov, y in data.groupby(by='prov'):
        if y['year'].values[0] == '2018':
            gdp = gdp_rows2018[prov]
            gdps.append(gdp)
        else:
            gdp = gdp_rows2017[prov]
            gdps.append(gdp)

        price_name = []
        for name, y1 in y.groupby(by='name'):
            price = y1['price_int'].sum()
            price_name.append(price)

        name_price.append(price_name)
        xticklabels.append(prov)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.boxplot(name_price)
    ax.set_xticklabels(xticklabels)  # 设置x轴刻度标签
    plt.xticks(rotation=25)
    plt.title("各省各类疫苗定价箱线图")
    plt.savefig("各省各类疫苗定价箱线图.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.bar(xticklabels, gdps, width=0.5, facecolor='yellowgreen')
    plt.xticks(rotation=36)
    plt.title("各省GDP")
    plt.savefig("各省GDP.png")
    plt.close()


def task_four():
    save = []
    data = pd.read_excel("dataAll.xlsx")
    for prov, y in data.groupby(by='prov'):
        gdps = []
        prices = []
        if y['year'].values[0] == '2018':
            gdp = gdp_rows2018[prov]
        else:
            gdp = gdp_rows2017[prov]
        for name, y1 in y.groupby(by='name'):
            price = y1['price_int'].sum()
            gdps.append(gdp)
            prices.append(price)

        corr = np.corrcoef(gdps, prices)[0][1]
        rows = {'prov': prov, 'corr': corr}
        save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("各省 各类疫苗定价与GDP相关系数.xlsx", index=None)


# 第一题
task_one()
# 第二题
task_two()
# 第三题
task_three()
# 第四题
task_four()
