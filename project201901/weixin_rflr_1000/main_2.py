import pandas as pd
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import time
import os
from sklearn.utils import shuffle
from collections import Counter


# 每次读取chunksize行，分批处理数据
def ogr_preprocess(inputs):
    df2list = inputs.values.tolist()
    out = []
    for i in df2list:
        if i == 'OGR':
            out.append(1)
        else:
            out.append(0)
    return out


def undersample():
    table = pd.read_table("./datasets/TrainSet.txt", chunksize=10000, sep=',')

    for df in table:
        _y = ogr_preprocess(df['Test_Result1'])
        x1 = df['ID1']
        x2 = df['Analyzer']
        x3 = df['Test_Start_Time']
        x4 = df['Process_Type']
        x5 = df['ID2_1']
        x6 = df['ID2_2']
        x7 = df['ID4_4']
        x9 = df['Time_of_Result']
        x11 = df['Test_Impact']
        x12 = df['ID3_1']
        x13 = df['ID3_2']
        x14 = df['ID2_3']
        x15 = df['Process1_Time']
        x16 = df['ID4_1']
        x17 = df['ID4_2']
        x18 = df['Process2_Time']
        x19 = df['ID5']
        x20 = df['ID4_3']
        _X = pd.concat([x1, x2, x3, x4, x5, x6, x7, x9, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20], axis=1)
        ### 使用RandomUnderSampler方法进行欠抽样处理 ###
        # 建立RandomUnderSampler模型对象
        model_RandomUnderSampler = RandomUnderSampler()
        # 输入数据并作欠抽样处理
        x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled = model_RandomUnderSampler.fit_sample(_X, _y)
        x_RandomUnderSampler_resampled = pd.DataFrame(x_RandomUnderSampler_resampled)
        # 将数据转换为数据框并命名列名
        y_RandomUnderSampler_resampled = pd.DataFrame(y_RandomUnderSampler_resampled, columns=['label'])
        # 按列合并数据框
        RandomUnderSampler_resampled = pd.concat([x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled],
                                                 axis=1)

        RandomUnderSampler_resampled.to_csv("RandomUnderSampler_resampled.csv", index=None, mode='a', header=None)


def time_pro(inputs):
    # 时间处理
    # 根据字符串格式，通过分割，把年月日 时分秒提取出来，转换为日期格式
    if 'AM' in inputs:
        tmp = inputs[:19]

        ydm, sfm = tmp.split(" ")[0], tmp.split(" ")[1]
        y, d, m = ydm.split("/")[2], ydm.split("/")[1], ydm.split("/")[0]
        s, f, mi = sfm.split(":")[0], sfm.split(":")[1], sfm.split(":")[2]
        t = (int(y), int(m), int(d), int(d), int(f), int(mi), 0, 0, 0)
        dates = time.mktime(t)
    else:
        tmp = inputs[:19]
        ydm, sfm = tmp.split(" ")[0], tmp.split(" ")[1]
        y, d, m = ydm.split("/")[2], ydm.split("/")[1], ydm.split("/")[0]
        s, f, mi = sfm.split(":")[0], sfm.split(":")[1], sfm.split(":")[2]
        t = (int(y), int(m), int(d), int(s) + 12, int(f), int(mi), 0, 0, 0)
        dates = time.mktime(t)
    return dates


def getTimespan(e, s):
    # 计算两个时间差，返回秒
    return time_pro(e) - time_pro(s)


def con2cate(inputs):
    # 转变数据类型，
    inputs2set = list(set(inputs))
    i = 0
    rows = {}
    for x in inputs2set:
        # 给每一个数据编号
        i += 1
        rows[x] = i

    out = []
    for i in inputs:
        # 转换为编号
        out.append(rows[i])
    return pd.Series(out)  # list转变为pandas类型


def con2cateSp(inputs):
    # 转变数据类型，
    inputs2set = list(set(inputs))
    i = 0
    rows = {}
    for x in inputs2set:
        # 给每一个数据编号
        i += 1
        rows[x] = i

    out = []
    for i in inputs:
        # 转换为编号
        if i > 50000:
            out.append(1)
        elif i > 40000:
            out.append(2)
        elif i > 30000:
            out.append(3)
        elif i > 20000:
            out.append(4)
        elif i > 10000:
            out.append(5)
        else:
            out.append(0)

    return pd.Series(out)  # list转变为pandas类型


def encodingData():
    data = pd.read_csv("RandomUnderSampler_resampled.csv")
    cols = ['ID1', 'Analyzer', 'Test_Start_Time', 'Process_Type', 'ID2_1', 'ID2_2', 'ID4_4', 'Time_of_Result',
            'Test_Impact', 'ID3_1', 'ID3_2', 'ID2_3', 'Process1_Time', 'ID4_1', 'ID4_2', 'Process2_Time',
            'ID5', 'ID4_3', 'Test_Result1']
    data.columns = cols

    data['Time_span'] = data.apply(lambda row: getTimespan(row['Time_of_Result'], row['Test_Start_Time']), axis=1)
    data['Time_span2'] = data.apply(lambda row: getTimespan(row['Process2_Time'], row['Process1_Time']), axis=1)
    data['Analyzer_2'] = con2cate(data['Analyzer'])
    data['Process_Type_2'] = con2cate(data['Process_Type'])
    data['ID2_2_2'] = con2cate(data['ID2_2'])
    data['ID4_4_2'] = con2cate(data['ID4_4'])
    data['Test_Impact_2'] = con2cate(data['Test_Impact'])
    data['ID3_1_2'] = con2cate(data['ID3_1'])
    data['ID3_2_2'] = con2cate(data['ID3_2'])
    data['ID4_1_2'] = con2cate(data['ID4_1'])
    data['ID4_2_2'] = con2cateSp(data['ID4_2'])
    data['ID5_2'] = con2cate(data['ID5'])
    data['ID4_3_2'] = con2cate(data['ID4_3'])

    for x in cols:
        if x == 'Test_Result1':
            continue
        del data['{}'.format(x)]
    data = shuffle(data)  # 打乱数据，使不同类型数据均匀分布
    print("训练数据大小：{}".format(data.shape))
    data.to_csv("TrainSet.csv", index=None)
    pos, neg = 0, 0
    for x in data['Test_Result1'].values.tolist():
        if x == 0:
            pos += 1
        else:
            neg += 1
    print("采样后各正负样本数量：")
    print(pos, neg)


def build_model():
    data = pd.read_csv("TrainSet.csv")
    print(data.shape)
    _y = data['Test_Result1']
    del data['Test_Result1']
    _X = data
    test_rate = 0.3
    x_train, x_test, y_train, y_test = train_test_split(_X, _y, test_size=test_rate, random_state=0)  # 分割训练集和测试集
    # 随机森林
    print("- * -" * 20)
    print("模型一:随机森林")
    rf_classifier = RandomForestClassifier(criterion='entropy', n_estimators=50,
                                           min_samples_leaf=1, max_depth=10, random_state=0)

    rf_classifier.fit(x_train.values, y_train)  # 拟合模型
    joblib.dump(rf_classifier, 'rf.model')

    y_test_ = rf_classifier.predict(x_test)  # 预测数据
    acc = accuracy_score(y_test, y_test_)  # 准确率
    print("随机森林：acc:{:0.5f}".format(acc))
    #
    names = x_train.columns.tolist()
    # 对特征进行排序，选择重要的特征
    results = sorted(zip(map(lambda x: round(x, 4), rf_classifier.feature_importances_), names), reverse=True)
    num = 9
    features = [i[1] for i in results[:num]]
    print("选择的特征是", features)
    new_train = x_train[features]  # 选择重要的前num个特征
    #
    print("- * -" * 20)
    print("模型二 特征选择后的随机森林")
    rf_classifier2 = RandomForestClassifier(criterion='entropy', n_estimators=50,
                                            min_samples_leaf=1, max_depth=10, random_state=0)

    rf_classifier2.fit(new_train, y_train)  # 拟合模型
    new_test = x_test[features]  # 选择重要的前num个特征
    y_test_ = rf_classifier2.predict(new_test)  # 预测数据
    acc = accuracy_score(y_test, y_test_)  # 准确率
    print("特征选择后的 随机森林：acc:{:0.5f}".format(acc))

    # 逻辑回归
    print("- * -" * 20)
    print("模型三 特征选择后的逻辑回归")
    logis_classifier_fs = LogisticRegression(solver='lbfgs', max_iter=10000)
    logis_classifier_fs.fit(new_train, y_train)
    y_test_ = logis_classifier_fs.predict(new_test)
    acc = accuracy_score(y_test, y_test_)
    print("特征选择后的 逻辑回归：acc:{:0.5f}".format(acc))

    # 直接做逻辑回归
    print("- * -" * 20)
    print("模型四 直接做逻辑回归")
    logis_classifier = LogisticRegression(solver='lbfgs', max_iter=10000)
    logis_classifier.fit(x_train.values, y_train)
    joblib.dump(logis_classifier, 'lr.model')
    y_test_ = logis_classifier.predict(x_test)
    acc = accuracy_score(y_test, y_test_)
    print("逻辑回归：acc:{:0.5f}".format(acc))


if __name__ == '__main__':

    method = 'encodingData'

    if method == 'undersample':
        # 下采样方法
        undersample()

    if method == 'encodingData':
        # 数据编码方法
        encodingData()

    if method == 'build_model':
        # 建立模型方法
        build_model()
