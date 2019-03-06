import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

names = 'ID1,Analyzer,Test_Start_Time,Process_Type,ID2_1,ID2_2,ID4_4,Test_Result2,Time_of_Result,Test_Result1,Test_Impact,ID3_1,ID3_2,ID2_3,Process1_Time,ID4_1,ID4_2,Process2_Time,ID5,ID4_3,Product'


def combine_data():
    file_name1 = '2016_Product1.txt'
    file_name2 = '2017_Product1.txt'
    file_name3 = '2018_Product1_DateFormatChange.txt'

    file_name4 = '2017_Product2.txt'
    file_name5 = '2018_Product2.txt'

    fw = open("./datasets/TrainSet.txt", 'w', encoding='utf-8')  # 打开一个文件，准备写数据
    fw.writelines(names + "\n")  # 写入标题行
    for file in [file_name1, file_name2, file_name3, file_name4, file_name5]:  # 遍历全部文件
        name = file.split("_")[1].split(".")[0].replace("Product", "")  # 获取product编号
        with open('./datasets/{}'.format(file), 'r', encoding='utf-8') as fr:  # 读文件
            limit = 10
            num = 0
            while True:
                num += 1
                line = fr.readline()  # 每次读取一行
                if line:
                    if num == 1:  # 跳过标题行
                        continue
                    fw.writelines("{},Product{}".format(line.replace("\n", ""), name) + '\n')  # 写入文件TrainSet.txt
                    if num % 1000000 == 0:
                        print('{}'.format(num))
                    # if num > limit:
                    #     break
                else:  # 文件底部停止
                    break

            print("file:{}, lines:{}".format(file, num))

    fw.close()


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


def get_features(num, test_rate=0.3):
    table = pd.read_table("./datasets/TrainSet.txt", chunksize=1000, sep=',')
    # 每次读取chunksize行，分批处理数据
    for df in table:
        _y = con2cate(df['Test_Result1'])   # 取出因变量，并转换为分类型数据， 下同
        x1 = con2cate(df['ID1'])
        x2 = con2cate(df['Analyzer'])
        x3 = con2cate(df['Test_Start_Time'])
        x4 = con2cate(df['Process_Type'])
        x5 = con2cate(df['ID2_1'])
        x6 = con2cate(df['ID2_2'])
        x7 = con2cate(df['ID4_4'])
        x9 = con2cate(df['Time_of_Result'])
        x11 = con2cate(df['Test_Impact'])
        x12 = con2cate(df['ID3_1'])
        x13 = con2cate(df['ID3_2'])
        x14 = con2cate(df['ID2_3'])
        x15 = con2cate(df['Process1_Time'])
        x16 = con2cate(df['ID4_1'])
        x17 = con2cate(df['ID4_2'])
        x18 = con2cate(df['Process2_Time'])
        x19 = con2cate(df['ID5'])
        x20 = con2cate(df['ID4_3'])
        _X = pd.concat([x1, x2, x3, x4, x5, x6, x7, x9, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20], axis=1)
        # 合并数据
        x_train, x_test, y_train, y_test = train_test_split(_X, _y, test_size=test_rate, random_state=0)  # 分割训练集和测试集
        # 随机森林
        rf_classifier = RandomForestClassifier(criterion='entropy', n_estimators=50,
                                               min_samples_leaf=1, max_depth=10, random_state=0)

        rf_classifier.fit(x_train.values, y_train)  # 拟合模型
        joblib.dump(rf_classifier, 'rf.model')

        y_test_ = rf_classifier.predict(x_test)  # 预测数据
        acc = accuracy_score(y_test, y_test_)  # 准确率
        print("随机森林：acc:{:0.5f}".format(acc))

        names = x_train.columns.tolist()
        # 对特征进行排序，选择重要的特征
        results = sorted(zip(map(lambda x: round(x, 4), rf_classifier.feature_importances_), names), reverse=True)
        features = [i[1] for i in results[:num]]
        print(features)
        new_train = x_train[features]  # 选择重要的前num个特征

        # 逻辑回归
        logis_classifier = LogisticRegression()
        logis_classifier.fit(new_train, y_train)
        joblib.dump(logis_classifier, 'lr.model')
        y_test_ = rf_classifier.predict(x_test)
        acc = accuracy_score(y_test, y_test_)
        print("逻辑回归：acc:{:0.5f}".format(acc))


if __name__ == '__main__':

    method = 'get_features'   # 修改这个，执行下面的函数

    if method == 'combine_data':
        # 合并数据
        combine_data()

    if method == 'get_features':

        get_features(num=10)
