import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data = pd.read_excel("集美大学各省录取分数.xlsx")


def is_benyipi(inputs):
    if '本一批' in inputs:
        return '1'
    return '0'


# 1、集美大学2015-2018年间不同省份在本一批的平均分数，柱状图展示排名前十的省份。
for kemu in ['理工', '文史']:
    data['本一批'] = data['批次'].apply(is_benyipi)  # 筛选本一批
    data_plot = data[data['本一批'].isin(['1']) & data['科类'].isin(['{}'.format(kemu)])]  # 本一批和理工或者文史数据筛选
    rows = {}
    for x, y in data_plot.groupby(by='省份'):  # 按省份分组
        means = y['平均分'].mean()  # 平均分的平均数
        rows[x] = means

    dic1SortList = sorted(rows.items(), key=lambda x: x[1], reverse=True)[:10]  # 对分数排序， 取前10
    names = []
    values = []

    for one in dic1SortList:
        names.append(one[0])
        values.append(one[1])

    plt.figure(figsize=(20, 10))
    plt.bar(names, values, width=0.5)  # 画条形图
    plt.xticks(rotation=36)  # 坐标轴旋转
    plt.title("集美大学不同省份在本一批{}的平均分数".format(kemu))   # 标题
    plt.savefig("集美大学不同省份在本一批{}平均分数.png".format(kemu))  # 保存
    plt.close()

# 2、分析福建省这3年各批次成绩情况，使用折线图展示结果，并预测2019年录取成绩。
for kemu in ['理工', '文史']:
        fujian = data[data['省份'].isin(["福建"]) & data['科类'].isin(["{}".format(kemu)])].sort_values(by="年份")  # 福建省 的理工和文史
        for x, y in fujian.groupby(by='批次'):  # 按照批次分组计算
            names = [str(i) for i in y['年份'].values]  # 获取年份
            values = y['省控线'].values  # 省控线
            if len(values) != 3:  # 不是3年的数据不要
                continue
            plt.figure(figsize=(10, 6))
            plt.plot(names, values, label='{}'.format(x))  # 画折现图
            plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
            plt.xticks(rotation=36)
            plt.title("福建省这3年{}{}类成绩情况".format(x, kemu))
            plt.grid()  # 显示网格
            plt.savefig("福建省这3年{}{}类成绩情况.png".format(x, kemu), bbox_inches='tight')
            plt.close()


# 3、分析其他省份数据
for chengji in ['最高分', '最低分', '平均分']:
    fujian = data[data['省份'].isin(["贵州"]) & data['科类'].isin(["理工"])].sort_values(by="年份")  # 贵州省的理工类成绩
    plt.figure(figsize=(10, 6))
    for x, y in fujian.groupby(by='批次'):  # 按照批次分组
        names = [str(i) for i in y['年份'].values]
        values = y['{}'.format(chengji)].values
        plt.plot(names, values, label='{}'.format(x))  # 各批次的每年的成绩， 预测2019年'最高分', '最低分', '平均分'的成绩
    plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)  # 显示图例
    plt.xticks(rotation=36)
    plt.title("贵州省这3年各批次理工类{}情况".format(chengji))
    plt.grid()
    plt.savefig("贵州省这3年各批次理工类{}情况.png".format(chengji), bbox_inches='tight')  # 保存图片
    plt.close()
