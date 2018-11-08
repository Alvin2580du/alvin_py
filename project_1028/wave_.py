# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:15:44 2018

@author: TK
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 12:25:42 2018

@author: TK
"""

"竞赛题目实战"

"包导入"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"随机森林需求"
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler  # 预测的核心
from sklearn.preprocessing import Imputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit


"Python三大工具，pandas，numpy，scipy"

trainName = ["时间", "小区名", "小区房屋出租数量", "楼层", "总楼层", "房屋面积", "房屋朝向", "居住状态",
             "卧室数量", "厅的数量", "卫的数量", "出租方式", "区", "位置", "地铁线路", "地铁站点", "距离", "装修情况", "totalSale"]
testName = ["Id", "时间", "小区名", "小区房屋出租数量", "楼层", "总楼层", "房屋面积", "房屋朝向", "居住状态",
            "卧室数量", "厅的数量", "卫的数量", "出租方式", "区", "位置", "地铁线路", "地铁站点", "距离", "装修情况", "totalSale"]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

"文件读取"

trainOfData = pd.read_csv('train.csv', encoding="utf-8", names=trainName)
testOfData = pd.read_csv('test.csv', encoding="utf-8", names=testName)

"3.数值分析"

"统计缺失数据"
missDatapersent = trainOfData.isnull().sum().sort_values(ascending=False)
totalMiss = trainOfData.isnull().count()
eachMiss = missDatapersent / totalMiss.sort_values(ascending=False)


# 定义百分比显示函数
def persentOfData(X):
    X = "%.2f%%" % (X * 100)
    return X


# 丢失数据的百分比显示
missDisPer = eachMiss.map(persentOfData)  # map需要函数名

# missSort总集 将总数据数，缺失百分比显示，缺失小数
missCollection = pd.concat([totalMiss, missDisPer, eachMiss], axis=1,
                           keys=["TotalAmount", "Persent", "PersentTemp"]).sort_values(by='PersentTemp',
                                                                                       ascending=False)
"输出缺失数据的情况"
# print(missSort[missSort["Persent"]!="0.00%"])
"缺失值处理"
deleteIndex = missCollection[missCollection["PersentTemp"] > 0.14].index
trainReal = trainOfData.drop(deleteIndex, axis=1)
# 这里testReal一定不能另做判断，因为最后一行的数据为totalSale,而testOfData中的所有的数据类型都是空，即NAN所以如果此处的testReal另做判断的情况下，将会导致
testReal = testOfData.drop(deleteIndex, axis=1)

"缺失处理之后数据段对比 去除了6个14%以上NAN的无用的数据项"

"字段分类 即将字段分为中文形式和数值类型的数据"
"""
这里十分精髓的利用了[]内的序号判断
columns为list,pandas.columns为list,其中该list为String类型的
col for col in trainReal.columns

"""
typeOfClass = [col for col in trainReal.columns if trainReal[col].dtypes == "O"]
typeOfData = [col for col in trainReal.columns if trainReal[col].dtypes != "O"]

"缺失数值填充 使用的是imputer"
paddingContent = Imputer(strategy="median")

"此处的test的时间与train完全不同，所以可以忽略不计"
"""
testReal=testReal.drop(["时间"],axis=1)
trainReal=trainReal.drop(["时间"],axis=1)
"""
testReal = testReal.drop(["Id"], axis=1)

"数值类型填充"
trainReal[typeOfData] = paddingContent.fit_transform(trainReal[typeOfData])
testReal[typeOfData[:-1]] = paddingContent.fit_transform(testReal[typeOfData[:-1]])
"文字类型填充"
trainReal[typeOfClass] = trainReal[typeOfClass].fillna("None")
testReal[typeOfClass] = testReal[typeOfClass].fillna("None")

"计算trainReal的相关性"

typeOfDataCorr = trainReal[typeOfData].corr("spearman")
typeOfDataReal = typeOfDataCorr[typeOfDataCorr["totalSale"] > 0.1]["totalSale"]
corrIndex = typeOfDataReal.sort_values(ascending=False).index
corrIndex = corrIndex.drop(["totalSale"])

"计算多重共线性"
X = np.matrix(trainReal[corrIndex])
mulOfList = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

"标准化数据"
"训练集数据标准化+降维操作"
dataScaler = StandardScaler()
trainRealData = dataScaler.fit_transform(trainReal[corrIndex])
dataPCA = PCA(n_components=5)
trainRealDataN = dataPCA.fit_transform(trainRealData)

"测试集数据标准化+降维操作"
dataScaler = StandardScaler()
testRealData = dataScaler.fit_transform(testReal[corrIndex])
dataPCA = PCA(n_components=5)
testRealDataN = dataPCA.fit_transform(testRealData)

"合成新结构框体trainRealDataN"
trainRealDataN = pd.DataFrame(trainRealDataN)
temp = np.matrix(trainRealDataN)
mulOfList = [variance_inflation_factor(temp, i) for i in range(temp.shape[1])]

"这一步可以略过，因为文字项就房屋朝向一个"
"文字项消除多重共线性+去重"
word = "+".join(typeOfClass)
wordFormula = "totalSale~ %s" % word
wordResult = anova_lm(ols(wordFormula, trainReal).fit())
deleteVar = list(wordResult[wordResult["PR(>F)"] > 0.05].index)

"定义文字编码函数"


def wordEncode(data):
    wordMap = {}
    for each in data.columns[:-1]:
        pid = pd.pivot_table(data, values="totalSale", index=each, aggfunc="mean")
        pid = pid.sort_values(by="totalSale")
        pid["rank"] = np.arange(1, pid.shape[0] + 1)
        wordMap[each] = pid["rank"].to_dict()
    return wordMap


"***************字符编码************ 完成对测试集和训练集的操作"
typeOfClass.append("totalSale")
wordMap = wordEncode(trainReal[typeOfClass])
for eachOne in typeOfClass[:-1]:
    trainReal[eachOne] = trainReal[eachOne].replace(wordMap[eachOne])
    testReal[eachOne] = testReal[eachOne].replace(wordMap[eachOne])

className = ["房屋朝向"]
"未降维类型DataFrame"
trainRealClassN = pd.DataFrame(trainReal[className])
trainRealClassN.columns = ["未降维类型A"]

"数据集合并操作"
trainRealDataN = pd.DataFrame(trainRealDataN)
trainRealDataN.columns = ["降维A", "降维B", "降维C", "降维D", "降维E"]
target = trainReal["totalSale"]
target = pd.DataFrame(target)
trainTotal = pd.concat([trainRealDataN], axis=1, ignore_index=True)

"测试集合并操作"
testRealDataN = pd.DataFrame(testRealDataN)
testRealDataN.columns = ["降维A", "降维B", "降维C", "降维D", "降维E"]
testRealClassN = pd.DataFrame(testReal[className])
testRealClassN.columns = ["未降维类型A"]
testTotal = pd.concat([testRealDataN], axis=1, ignore_index=True)

"分割训练集"
test_rate = 0.4
random_state = 0
trainData, testData, trainTarget, testTarget = train_test_split(trainTotal, target, test_size=test_rate, random_state=random_state)
print(trainData.shape)
cv = ShuffleSplit(n_splits=10, test_size=test_rate, random_state=random_state)

# rf_param_1 = {'max_features': range(1, trainData.shape[1], 10)}
rf_param_1 = {'n_estimators': range(1, 500, 50), 'max_depth': range(3, 29, 2), 'max_features': range(3, 15, 2)}
rf_grid_1 = GridSearchCV(RandomForestRegressor(), param_grid=rf_param_1, cv=cv)
rf_grid_1.fit(trainData, trainTarget.values.ravel())
score = r2_score(testTarget, rf_grid_1.predict(testData))
print("- "*20)
print(score)
print(rf_grid_1)
print("- "*20)
# predict = m.predict(testTotal)
# test = pd.read_csv('testnew.csv')['id']
# sub = pd.DataFrame()
# sub['id'] = test
# sub['price'] = pd.Series(predict)
# sub.to_csv('Predictions.csv', index=False)
