from numpy import *
import os
import pandas as pd


def split_datasets(filename="./datasets/knn/digit-training.txt"):
    dir_name = filename.split("/")[-1].split(".")[0].split("-")[1]
    save_path = './datasets/knn/{}'.format(dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = pd.read_csv(filename, header=None)

    def funs(inputs):
        if len(inputs) > 10:
            return "data"
        else:
            return inputs.strip()

    datacopy = data.copy()
    datacopy['labels'] = data[0].apply(funs)
    label = datacopy[~datacopy['labels'].isin(['data'])]
    label.columns = ['0', '1']
    train = datacopy[datacopy['labels'].isin(['data'])][0]
    k = 0
    index = 0
    limit = 32
    save = []
    for y in train:
        save.append(y)
        k += 1
        if k >= limit:
            df = pd.DataFrame(save)
            print(df.shape)
            df.to_csv("./datasets/knn/{}/{}_{}.txt".format(dir_name, index, label['1'].values[index]), index=None,
                      header=None)
            save = []
            k = 0
            index += 1


# KNN分类核心方法
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0]代表行数

    # # step 1: 计算欧式距离
    # tile(A, reps): 将A重复reps次来构造一个矩阵
    # the following copy numSamples rows for dataSet
    diff = tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5
    # # step 2: 对距离排序
    # argsort()返回排序后的索引
    sortedDistIndices = argsort(distance)

    classCount = {}  # 定义一个空的字典
    for i in range(k):
        # # step 3: 选择k个最小距离
        voteLabel = labels[sortedDistIndices[i]]

        # # step 4: 计算类别的出现次数
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # # step 5: 返回出现次数最多的类别作为分类结果
    maxCount = 0
    maxIndex = None
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex, maxCount


# 将图片转换为向量
def img2vector(filename):
    rows = 32
    cols = 32
    imgVector = zeros((1, rows * cols))
    fileIn = open(filename)
    for row in range(rows):
        lineStr = fileIn.readline()
        for col in range(cols):
            imgVector[0, row * 32 + col] = int(lineStr[col])
    return imgVector


# 加载数据集
def loadDataSet():
    # # step 1: 读取训练数据集
    print("---Getting training set...")
    dataSetDir = './datasets/knn/'
    trainingFileList = os.listdir(dataSetDir+"training")  # 加载测试数据
    numSamples = len(trainingFileList)

    train_x = zeros((numSamples, 1024))
    train_y = []
    for i in range(numSamples):
        filename = trainingFileList[i]
        # get train_x
        train_x[i, :] = img2vector(dataSetDir + 'training/%s' % filename)
        # get label from file name such as "1_18.txt"
        label = int(filename.split('.')[0].split("_")[1])  # return 1
        train_y.append(label)

    # # step 2:读取测试数据集
    print("---Getting testing set...")
    testingFileList = os.listdir(dataSetDir + 'testing')  # load the testing set
    numSamples = len(testingFileList)
    test_x = zeros((numSamples, 1024))
    test_y = []
    for i in range(numSamples):
        filename = testingFileList[i]
        # get train_x
        test_x[i, :] = img2vector(dataSetDir + 'testing/%s' % filename)
        # get label from file name such as "1_18.txt"
        label = int(filename.split('.')[0].split("_")[1])  # return 1
        test_y.append(label)
    return train_x, train_y, test_x, test_y


# 手写识别主流程
def HandWritingClass():
    # # step 1: 加载数据
    print("step 1: load data...")
    train_x, train_y, test_x, test_y = loadDataSet()

    numTestSamples = test_x.shape[0]
    matchCount = 0
    k = 10
    maxCount = None
    for i in range(numTestSamples):
        predict, maxCount = kNNClassify(test_x[i], train_x, train_y, k)
        if predict == test_y[i]:
            matchCount += 1
            print("maxCount:{}".format(maxCount))
    accuracy = float(matchCount) / numTestSamples

    # # step 4: 输出结果
    print("step 2: show the result...")
    print('The classify accuracy is: %.2f%%' % (accuracy * 100))


if __name__ == '__main__':

    method = 'HandWritingClass'

    if method == 'split_datasets':
        dataname = ['./datasets/knn/digit-training.txt', './datasets/knn/digit-testing.txt',
                    './datasets/knn/digit-predict.txt']
        for n in dataname:
            split_datasets(n)

    if method == 'loadDataSet':
        train_x, train_y, test_x, test_y = loadDataSet()
        print(train_x.shape, len(train_y), test_x.shape, len(test_y))


    if method == 'HandWritingClass':
        HandWritingClass()