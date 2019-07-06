import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl

# 设置字体
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

# 读取爬虫得到的数据
data = pd.read_excel("data.xlsx")

# 城市名称
names = data['city'].values.tolist()
# 房源总价
totalPrice = [int(float(i.replace("万", ""))) for i in data['totalPrice'].values.tolist()]

plt.figure(figsize=(20, 10))
# 作条形图
plt.bar(names, totalPrice, width=0.5)
plt.xticks(rotation=36)  # 坐标轴旋转36度
plt.yticks(range(0, max(totalPrice), 1000))
plt.title("不同城市房价对比")
plt.savefig("不同城市房价对比.png")  # 保存图片
plt.close()  # 关闭图形
print('1')

names1 = data['houseInfo'].values.tolist()
plt.figure(figsize=(20, 10))
plt.bar(names1, totalPrice, width=0.5)
plt.xticks(rotation=36)
plt.yticks(range(0, max(totalPrice), 1000))
plt.title("全国不同户型房价对比")
plt.savefig("全国不同户型房价对比.png")
print('2')
plt.close()

xz_data = data[data['city'].isin(['西安'])]
names2 = xz_data['positionInfo'].values.tolist()
xz_totalPrice = [int(float(i.replace("万", ""))) for i in xz_data['totalPrice'].values.tolist()]
plt.figure(figsize=(20, 10))
plt.bar(names2, xz_totalPrice, width=0.5)
plt.xticks(rotation=36)
plt.yticks(range(0, max(xz_totalPrice), 200))
plt.title("西安不同区域房价对比")
plt.savefig("西安不同区域房价对比.png")
print('3')
plt.close()
