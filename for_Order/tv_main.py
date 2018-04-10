
""" 需求

1. 根据附件1中的用户观看信息数据，试分析用户的收视偏好，并给出附件2的产品的营销推荐方案

2. 为了更好的维用户服务，扩大营销范围，利用附件1到附件3的数据，试对相似偏好的用户进行分类（用户标签），
    对产品进行分类打包（产品标签），并给出营销推荐方案


附件1： 用户收视信息，用户回看信息，用户点播信息，用户单片点播信息。
附件2： 电视产品信息数据
附件3：用户基本信息
"""

import pandas as pd

shoushi = pd.read_csv("./datasets/tv_data/用户收视信息.csv")
print(shoushi.head())
huikan = pd.read_csv("./datasets/tv_data/用户回看信息.csv")
print(huikan.head())
dianbo = pd.read_csv("./datasets/tv_data/用户点播信息.csv")
print(dianbo.head())
danpiandianbo = pd.read_csv("./datasets/tv_data/用户单片点播信息.csv")
print(danpiandianbo.head())
chanpin = pd.read_csv("./datasets/tv_data/电视产品信息数据.csv")
print(chanpin.head())

