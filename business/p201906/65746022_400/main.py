from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn import neighbors
import numpy as np
from sklearn import preprocessing
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

"""
2、基于knn的分类算法
要求 1 简述knn分类算法的基本思想和算法流程
     2 利用knn算法对鸢尾花数据进行分类，选取120个样本作为训练集，30个样本作为测试集（每类10个样本）
     3 将训练数据和测试数据以图像形式显示，从图像上要能够分清训练/测试数据、分类对错情况、以及花种类的不同
附加 1 尝试使用正则化方案对数据进行预处理，并评估对分类效果的影响
     2 归纳分析使用不同的k值是对分类效果的影响
     3 改变训练测试数据的比例，并分析对分类效果的影响

"""


# 1. 原理：
######### https://www.cnblogs.com/listenfwind/p/10311496.html


class Iris:
    iris = load_iris()
    x = iris.data
    y = iris.target
    X_normalized = preprocessing.normalize(x, norm='l2')

    def train(self):
        k_range = range(1, 30)
        k_error = []
        k_error1 = []

        # 循环，取k=1到k=30，查看误差效果
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            # cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
            scores = cross_val_score(knn, self.x, self.y, cv=6, scoring='accuracy')
            k_error.append(1 - scores.mean())

            scores1 = cross_val_score(knn, self.X_normalized, self.y, cv=6, scoring='accuracy')
            k_error1.append(1 - scores1.mean())

        # 画图，x轴为k值，y值为误差值
        plt.plot(k_range, k_error)
        plt.plot(k_range, k_error1)
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Error')
        plt.title("不同k值的训练误差对比")
        plt.legend(labels=['非正则化', '正则化'])
        plt.savefig("不同k值的训练误差对比.png")
        plt.close()

    def train_cv(self):
        k_range = range(3, 10)
        k_error = []
        for i in k_range:
            knn = KNeighborsClassifier(n_neighbors=10)
            # cv参数决定数据集划分比例，这里是按照cv划分训练集和测试集
            scores = cross_val_score(knn, self.x, self.y, cv=i, scoring='accuracy')
            k_error.append(1 - scores.mean())

        # 画图，x轴为训练集和测试集比例，y值为误差值
        plt.plot(k_range, k_error)
        plt.xlabel('Value of cv for KNN')
        plt.ylabel('Error')
        plt.title("不同分割比例的训练误差对比")
        plt.savefig("不同分割比例的训练误差对比.png")
        plt.close()

    def plot_acc(self):
        n_neighbors = 11
        # 导入一些要玩的数据
        x = self.iris.data[:, :2]  # 我们只采用前两个feature,方便画图在二维平面显示
        h = .02  # 网格中的步长
        # 创建彩色的图
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # 创建了一个knn分类器的实例，并拟合数据。
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform")
        clf.fit(x, self.y)

        # 绘制决策边界。为此，我们将为每个分配一个颜色
        # 来绘制网格中的点 [x_min, x_max]x[y_min, y_max].
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # 将结果放入一个彩色图中
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # 绘制训练点
        plt.scatter(x[:, 0], x[:, 1], c=self.y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i)" % n_neighbors)
        plt.savefig("训练结果可视化.png")


iris = Iris()
iris.train()
iris.train_cv()
iris.plot_acc()
