import pandas as pd
import urllib.request
from urllib import error
import urllib.parse
from bs4 import BeautifulSoup
from collections import OrderedDict
from tqdm import trange
import matplotlib.pyplot as plt
import re

import numpy as np

# ,Central_Pressure_mb,Max_Winds_kt
"""
a) Show appropriate summary statistics and visualizations for:
Months, Highest_Category, Central_Pressure_mb, Max_Winds_kt
"""


def get_counts(sequence):
    # 对一个列表统计频率，出现就+1
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts


def build_1a():
    data = pd.read_csv("Hurricane.csv", usecols=['Month', 'Highest_Category', 'Central_Pressure_mb', 'Max_Winds_kt'])
    # 读取数据，并做描述性统计分析
    summary_statistics = data.describe()
    summary_statistics_df = pd.DataFrame(summary_statistics)
    # 保存到文件
    summary_statistics_df.to_csv("summary_statistics.csv")
    # hist是画直方图， .values.tolist() 是取到数据的方法
    plt.figure()
    plt.hist(data['Month'].values.tolist(), bins=12, facecolor="blue")
    plt.savefig("Month.png")
    plt.close()

    plt.figure()
    plt.hist(data['Highest_Category'].values.tolist(), bins=8, facecolor="blue")
    plt.savefig("Highest_Category.png")
    plt.close()

    plt.figure()
    plt.hist(data['Central_Pressure_mb'].values.tolist(), bins=8, facecolor="blue")
    plt.savefig("Central_Pressure_mb.png")
    plt.close()

    plt.figure()
    plt.hist(data['Max_Winds_kt'].values.tolist(), bins=8, facecolor="blue")
    plt.savefig("Max_Winds_kt.png")
    plt.close()


# Highest_Category and Max_Winds_kt
"""

b) Show the relationship between 
Highest_Category and Max_Winds_kt
Central_Pressure_mb and Max_Winds_kt
"""


def build_1b():
    data = pd.read_csv("Hurricane.csv", usecols=['Month', 'Highest_Category', 'Central_Pressure_mb', 'Max_Winds_kt'])
    # 去掉缺失值
    data_naonit = data.dropna()
    x1 = data_naonit['Highest_Category'].values.tolist()
    x2 = data_naonit['Max_Winds_kt'].values.tolist()
    # scatter 画散点图，观察二者的关系
    plt.figure()
    plt.scatter(x1, x2)
    plt.title("relationship  between Highest_Category and Max_Winds_kt")
    plt.xlabel("Max_Winds_kt")
    plt.ylabel("Highest_Category")
    # 设置坐标轴， 以及保存图片
    plt.savefig("relationship  between Highest_Category and Max_Winds_kt.png")
    plt.show()
    # 关掉图片
    plt.close()
    # 选择Central_Pressure_mb 这一列
    x3 = data_naonit['Central_Pressure_mb'].values.tolist()
    plt.figure()
    plt.scatter(x1, x3)
    plt.title("relationship  between Central_Pressure_mb and Max_Winds_kt")
    plt.xlabel("Max_Winds_kt")
    plt.ylabel("Central_Pressure_mb")
    plt.savefig("relationship  between Central_Pressure_mb and Max_Winds_kt.png")
    plt.show()
    plt.close()


"""
c) Explain how you accounted for any missing data,
   and how that may have affected your results

"""
# 去掉缺失值
# 影响数据之间的关系，对结论造成影响。


# Part 2
def urlhelper(url):
    # 打开url 的方法
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36")
        req.add_header("Accept", "*/*")  # 添加请求头
        req.add_header("Accept-Language", "zh-CN,zh;q=0.8")
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')  # 转换编码方式
        return html
    except error.URLError as e:
        # 如果打不开，报异常
        print(e)


def get_counrty(inputs):
    # 用正式表达式 提取国家这个字段的数据
    res = re.findall('.*/>(.*)</td>', str(inputs))[0]
    return res


def build_2a():
    save = []
    # url 一共有5页，遍历每一页
    for page in trange(1, 5):
        try:
            url = 'https://www.top500.org/list/2018/06/?page={}'.format(page)
            print("Download :{}".format(url))
            html = urlhelper(url)
            # 解析网页数据
            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('table', attrs={"class": "table table-condensed table-striped"})
            for one in range(len(resp)):
                try:
                    post_info = resp[one]
                    title = post_info.find_all('tr')
                    for t in title:
                        try:
                            # 获取td 标签的数据，并保存到一个字典中，最后保存到list里面，
                            info = t.find_all('td')
                            rows = OrderedDict()
                            rows['Rank'] = info[0].text.replace('\n', '').strip()
                            rows['Site'] = info[1].text.replace(get_counrty(info[1]), "")
                            rows['Country'] = get_counrty(info[1])
                            rows['System'] = info[2].text.replace('\n', '').strip()
                            rows['Cores'] = info[3].text.replace('\n', '').replace(' ', '').replace(',', '')
                            rows['Rmax'] = info[4].text.replace('\n', '').replace(' ', '').replace(',', '')
                            rows['Rpeak'] = info[5].text.replace('\n', '').replace(' ', '').replace(',', '')
                            rows['Power'] = info[6].text.replace('\n', '').replace(' ', '').replace(',', '')
                            save.append(rows)
                        except Exception as e:
                            print(e)
                            continue
                except Exception as e:
                    continue
        except Exception as e:
            continue
    # 把抓取的数据保存到文件
    df = pd.DataFrame(save)
    df.to_csv("table.csv", index=None)


def clean_data(inputs):
    # 清晰数据，去掉换行符，空格以及逗号
    return inputs.replace('\n', '').replace(' ', '').replace(',', '')


def build_2bc():
    data2 = pd.read_csv("table.csv", usecols=['Cores', 'Rmax', 'Rpeak', 'Power'])
    # 读取数据， 只读取需要的4列数据
    data2_summary_statistics = data2.describe()
    data2_summary_statistics_df = pd.DataFrame(data2_summary_statistics)
    data2_summary_statistics_df.to_csv("data2_summary_statistics.csv")

    x1 = data2['Cores'].values.tolist()
    x2 = data2['Rpeak'].values.tolist()

    plt.figure()
    plt.scatter(x1, x2)
    plt.title("relationship  between Cores and Rpeak")
    plt.xlabel("Cores")
    plt.ylabel("Rpeak")
    plt.savefig("relationship  between Cores and Rpeak.png")
    plt.show()
    plt.close()

    x3 = data2['Power'].values.tolist()
    plt.figure()
    plt.scatter(x1, x3)
    plt.title("relationship  between Cores and Power")
    plt.xlabel("Cores")
    plt.ylabel("Power")
    plt.savefig("relationship  between Cores and Power.png")
    plt.show()
    plt.close()


def build_2d():
    # 读取国家数据，并做条形图展示
    data3 = pd.read_csv("table.csv", usecols=['Country']).values.tolist()
    sequence = [i for j in data3 for i in j]
    res = get_counts(sequence)
    objects = list(res.keys())
    performance = list(res.values())
    y_pos = np.arange(len(objects))
    plt.figure(figsize=(30, 20))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel("Usage")
    plt.title("Programming Language Usage")
    plt.xticks(rotation=45)
    plt.savefig("Country.png")
    plt.show()


if __name__ == "__main__":
    method = 'build_2d'
    if method == 'build_1a':
        build_1a()

    if method == 'build_1b':
        build_1b()

    if method == 'build_2a':
        build_2a()

    if method == 'build_2bc':
        build_2bc()

    if method == 'build_2d':
        build_2d()
