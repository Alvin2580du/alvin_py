import os
import pandas as pd
import math
from functools import reduce
import numpy as np


def vector_add(v, w):
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_or(v, w):
    """boolean 'or' two vectors componentwise"""
    return [v_i or w_i for v_i, w_i in zip(v, w)]


def vector_and(v, w):
    """boolean 'and' two vectors componentwise"""
    return [v_i and w_i for v_i, w_i in zip(v, w)]


def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return sum(v_i * v_i for v_i in v)


def distance(v, w):
    s = vector_subtract(v, w)
    return math.sqrt(sum_of_squares(s))


def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))


def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def vector_sum(vectors):
    return reduce(vector_add, vectors)


def scalar_multiply(c, v):
    return [round(c * v_i, 2) for v_i in v]


def vector_mean(vectors):
    """compute the vector whose i-th element is the mean of the
    i-th elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


def applyfuns(inputs):
    if len(inputs) > 10:
        return "data"
    else:
        return inputs.strip()


def split_datasets(filename="./datasets/knn/digit-training.txt"):
    # 将原始数据分拆开，一个样本保存到一个文件中
    dir_name = filename.split("/")[-1].split(".")[0].split("-")[1]
    save_path = './datasets/knn/{}'.format(dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data = pd.read_csv(filename, header=None)

    datacopy = data.copy()
    datacopy['labels'] = data[0].apply(applyfuns)
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
            df.to_csv("./datasets/knn/{}/{}_{}.txt".format(dir_name, index, label['1'].values[index]), index=None,
                      header=None)
            save = []
            k = 0
            index += 1


def tile_array(inputsarr, times):
    save = []
    for i in range(times):
        save.append(inputsarr[0])
    out = np.array(save).reshape((times, len(inputsarr[0])))
    return out


# KNN分类核心方法
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0]代表行数
    # # step 1: 计算欧式距离
    diff = tile_array(newInput, numSamples) - dataSet  # Subtract element-wise
    for i in range(len(diff)):
        for j in range(len(diff)):
            distaces = distance(diff[i], diff[j])
            print(distaces)
    # sample_distance = np.sqrt(np.sum(np.square(diff), axis=1))
    # # # step 2: 对距离排序
    # # argsort()返回排序后的索引
    # sortedDistIndices = np.argsort(sample_distance)

    classCount = {}  # 定义一个空的字典
    for i in range(k):
        # # step 3: 选择k个最小距离
        voteLabel = labels[sortedDistIndices[i]]
        # # step 4: 计算类别的出现次数
        # when the key voteLabel is not in dictionary classCount, get() will return 0
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
    imgVector = np.zeros((1, rows * cols))
    fileIn = open(filename)
    for row in range(rows):
        lineStr = fileIn.readline()
        for col in range(cols):
            imgVector[0, row * 32 + col] = int(lineStr[col])
    return imgVector


def handwritingClass(k=3):
    hwLabels = []
    print("---Getting training set...")
    dataSetDir = './datasets/knn/'
    trainingFileList = os.listdir(dataSetDir + "training")  # 加载测试数据

    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('.')[0].split("_")[1])  # return 1
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(dataSetDir + 'training/%s' % fileNameStr)

    print("---Getting testing set...")
    testFileList = os.listdir(dataSetDir + 'testing')  # load the testing set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('.')[0].split("_")[1])  # return 1
        vectorUnderTest = img2vector(dataSetDir + 'testing/%s' % fileNameStr)
        classifierResult, mxcount = kNNClassify(vectorUnderTest, trainingMat, hwLabels, k)
        if classifierResult == classNumStr:
            errorCount += 1.0
    print("\n{} acc: : {}".format(k, errorCount / float(mTest)))


def buildPredict(k=3):
    print("---Getting predict set...")
    dataSetDir = './datasets/knn/'
    trainingFileList = os.listdir(dataSetDir + "training")  # 加载测试数据

    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    hwLabels = []

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('.')[0].split("_")[1])  # return 1
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(dataSetDir + 'training/%s' % fileNameStr)

    dataSetDir = './datasets/knn/'
    for filename in os.listdir(dataSetDir + "predict"):
        predict_i = img2vector(dataSetDir + 'predict/%s' % filename)

        diff = tile_array(predict_i, m) - trainingMat
        distance = np.sqrt(np.sum(np.square(diff), axis=1))
        sortedDistIndices = np.argsort(distance)
        classCount = {}  # 定义一个空的字典
        voteLabel = hwLabels[sortedDistIndices[k]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        # # step 5: 返回出现次数最多的类别作为分类结果
        maxCount = 0
        maxIndex = None
        for key, value in classCount.items():
            if value > maxCount:
                maxCount = value
                maxIndex = key
        print("file: {}, Predict: {}".format(filename, maxIndex))


if __name__ == '__main__':

    method = 'handwritingClass'

    if method == 'split_datasets':
        dataname = ['./datasets/knn/digit-training.txt', './datasets/knn/digit-testing.txt',
                    './datasets/knn/digit-predict.txt']
        for n in dataname:
            split_datasets(n)

    if method == 'handwritingClass':
        k_list = [3, 5, 7, 9]
        for k in k_list:
            handwritingClass(k)

    if method == 'buildPredict':
        buildPredict()

    if method == 'distance':
        res = distance(v=[1, 2, 3], w=[2, 3, 4])
        print(res)
