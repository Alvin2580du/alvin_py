from math import log  # 求对数
import xlrd, xlwt
from numpy import mat
from   pprint import pprint
import networkx as nx  # 复杂网络需要用的包
import matplotlib.pyplot as plt  # 画图用

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文字体
chengfadu1 = 0
yaowuchuxiancishu1 = 10


# 计算熵
def unionShannonEnt(x1, x2, n=chengfadu1):
    numEntries = int(len(x1))
    # print(numEntries)
    X00 = float(0)
    X11 = float(0)
    X10 = float(0)
    X01 = float(0)
    # （以下五行）为所有可能分类创建字典
    # print(x1.sum(axis=0),x2.sum(axis=0))
    for i in range(numEntries):
        if x1[i, 0] == 0 and x2[i, 0] == 0:
            X00 += 1
        elif x1[i, 0] == 1 and x2[i, 0] == 1:
            X11 += 1
        elif x1[i, 0] == 0 and x2[i, 0]:
            X10 += 1
        else:
            X01 += 1
    # print(X00,X01,X10,X11)
    numEntries = float(numEntries)
    if (X00 + X01) / numEntries > 0:
        a1 = -(X00 + X01) / numEntries * (log((X00 + X01) / numEntries, 10))
    else:
        a1 = 0
    if (X10 + X11) / numEntries > 0:
        b1 = -(X10 + X11) / numEntries * (log((X10 + X11) / numEntries, 10))
    else:
        b1 = 0
    if (X10 + X00) / numEntries > 0:
        c1 = -(X10 + X00) / numEntries * log((X10 + X00) / numEntries, 10)
    else:
        c1 = 0
    if (X01 + X11) / numEntries > 0:
        d1 = -(X01 + X11) / numEntries * log((X01 + X11) / numEntries, 10)
    else:
        d1 = 0
    H_x1 = a1 + b1
    H_x2 = c1 + d1
    if X00 / numEntries > 0:
        a = -X00 / numEntries * log(X00 / numEntries, 10)
    else:
        a = 0
    if X11 / numEntries > 0:
        b = -X11 / numEntries * log(X11 / numEntries, 10)
    else:
        b = 0
    if X01 / numEntries > 0:
        c = -X01 / numEntries * log(X01 / numEntries, 10)
    else:
        c = 0
    if X10 / numEntries > 0:
        d = -X10 / numEntries * log(X10 / numEntries, 10)
    else:
        d = 0
    Hx1_x2 = a + b + c + d
    if X11 > n:
        unionEnt = (H_x1 + H_x2 - Hx1_x2) / (H_x1 * H_x2) ** 0.5
    else:
        if H_x1 * H_x2 > 0:
            unionEnt = (H_x1 + H_x2 - 2 * Hx1_x2) / (H_x1 * H_x2) ** 0.5
        else:
            unionEnt = 0.1
    return unionEnt


# 熵就是分类为0或者1的数目
def Readexcel(table, sheet):
    data = xlrd.open_workbook(table)
    try:
        # table=data.sheets()[2]  #通过索引顺序获取
        # # table = data.sheet_by_index()  # 通过索引顺序获取
        table = data.sheet_by_name(sheet)  # 通过名称获取d
    except:
        print("no sheet  in %s named  sheet1" % data)
    print(table.nrows, table.ncols)  # 读取数据的行和列
    dataset = []
    labels = []
    for n in range(table.nrows):
        col = []
        for c in range(table.ncols):
            col.append(table.cell(n, c).value)
        if n > 0:
            dataset.append(col[:])
        if n == 0:
            labels.append(col[:])
    return dataset, labels


# 读取数据
xMat, labels = Readexcel(r"C:\Users\Administrator\Desktop\方剂.xlsx", u'data')
xMat = mat(xMat)
# print(xMat)
print(labels)
z = unionShannonEnt(xMat[:, 1], xMat[:, 2], n=5)
print(z)
result_2 = []
set_result2 = {}
res = float(0)
for m in range(xMat.shape[1]):
    for n in range(xMat.shape[1]):
        if m < n and xMat[:, m].sum(axis=0) > yaowuchuxiancishu1 and xMat[:, n].sum(axis=0) > yaowuchuxiancishu1:  # 相关度
            # print(m,n)
            res = float(unionShannonEnt(xMat[:, m], xMat[:, n], n=0))
            result_2.append([labels[0][m], labels[0][n], res])
            set_result2[frozenset((labels[0][m], labels[0][n]))] = res
print(result_2[0])  #
result_2.sort(key=lambda x: x[2], reverse=True)  # 数据降序排列.
print(result_2)  # 药物关联度
# 生成三个类别、四个类别的组合
result_3 = []
for p in range(len(result_2)):
    for q in range(len(result_2)):
        if p < q and result_2[p][2] > 0.13 and result_2[q][2] > 0.13:
            result_3.append(list(set(result_2[p][0:2]) | set(result_2[q][0:2])))
print(result_3)
# 生成绘制网络图的数据
G = nx.Graph()  # 创建无向图
# 添加节点
for z1 in result_3:
    for z2 in z1:
        for z3 in z1:
            if frozenset((z2, z3)) in set_result2.keys():
                if set_result2[frozenset((z2, z3))] > 0.01:
                    G.add_edge(z2, z3)
# 创建画布，绘制和展示图形
plt.figure()  # 创建一幅图
nx.draw(G, node_color='white', with_labels=True, node_size=300, font_size=11, font_color='black')
# plt.figure()  # 创建一幅图
# nx.draw(G, node_color='black',with_labels=False, node_size=30)
plt.show()
# 导出数据

book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('关联', cell_overwrite_ok=True)
sheet2 = book.add_sheet('聚类', cell_overwrite_ok=True)
n = 0
for x in result_2:
    # print(x)
    sheet.write(n, 1, str(x))
    n += 1
b = 0
for y in result_3:
    sheet2.write(b, 1, str(y))
    b += 1
book.save(r'd:\聚类.csv')  # 保存结果
