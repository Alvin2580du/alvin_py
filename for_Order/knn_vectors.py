import os
import math
from functools import reduce
import numpy as np
from collections import Counter
import pandas as pd
from datetime import datetime

dataSetDir = './datasets/knn/'

logs = open("./datasets/knn/logs.txt", 'a+', encoding='utf-8')

acc_logs = open("./datasets/knn/acclogs.txt", 'a+', encoding='utf-8')

predict_logs = open("./datasets/knn/predictlogs.txt", 'a+', encoding='utf-8')


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


def img2vectorV1(filename):
    # get data
    rows = 32
    cols = 32
    imgVector = []
    fileIn = open(filename)
    for row in range(rows):
        lineStr = fileIn.readline()
        for col in range(cols):
            imgVector.append(int(lineStr[col]))
    return imgVector


def get_dict_min(lis, k):
    # find most Nearest Neighbors
    gifts = lis[:k]
    save = []
    for g in gifts:
        res = g[1]
        save.append(res)
    return Counter(save).most_common(1)[0][0]


def knnclassifiy(k=3):

    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    hwLabels = []
    trainingFileList = os.listdir(dataSetDir + "training")  # load training data
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('.')[0].split("_")[1])

        if classNumStr == 0:
            k0 += 1
        elif classNumStr == 1:
            k1 += 1
        elif classNumStr == 2:
            k2 += 1
        elif classNumStr == 3:
            k3 += 1
        elif classNumStr == 4:
            k4 += 1
        elif classNumStr == 5:
            k5 += 1
        elif classNumStr == 6:
            k6 += 1
        elif classNumStr == 7:
            k7 += 1
        elif classNumStr == 8:
            k8 += 1
        else:  # 9
            k9 += 1
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vectorV1(dataSetDir + 'training/%s' % fileNameStr)  # read data to python list

    testFileList = os.listdir(dataSetDir + 'testing')
    tk0, tk1, tk2, tk3, tk4, tk5, tk6, tk7, tk8, tk9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    tkp0, tkp1, tkp2, tkp3, tkp4, tkp5, tkp6, tkp7, tkp8, tkp9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    C = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        TestclassNumStr = int(fileStr.split('.')[0].split("_")[1])
        if TestclassNumStr == 0:
            tkp0 += 1
        elif TestclassNumStr == 1:
            tkp1 += 1
        elif TestclassNumStr == 2:
            tkp2 += 1
        elif TestclassNumStr == 3:
            tkp3 += 1
        elif TestclassNumStr == 4:
            tkp4 += 1
        elif TestclassNumStr == 5:
            tkp5 += 1
        elif TestclassNumStr == 6:
            tkp6 += 1
        elif TestclassNumStr == 7:
            tkp7 += 1
        elif TestclassNumStr == 8:
            tkp8 += 1
        else:  # 9
            tkp9 += 1
        data_file_name = dataSetDir + 'testing/%s' % fileNameStr
        vectorUnderTest = img2vectorV1(data_file_name)
        distaces_list = {}
        for j in range(m):
            distaces = distance(vectorUnderTest, trainingMat[j])  # compute distance
            distaces_list[distaces] = hwLabels[j]
        sorted_distance_list = sorted(distaces_list.items(), key=lambda e: e[0], reverse=False)  # sorted distance
        gifts = get_dict_min(sorted_distance_list, k)  # get kth min distance
        if TestclassNumStr == gifts:
             C += 1

        if gifts == 0:
            tk0 += 1
        elif gifts == 1:
            tk1 += 1
        elif gifts == 2:
            tk2 += 1
        elif gifts == 3:
            tk3 += 1
        elif gifts == 4:
            tk4 += 1
        elif gifts == 5:
            tk5 += 1
        elif gifts == 6:
            tk6 += 1
        elif gifts == 7:
            tk7 += 1
        elif gifts == 8:
            tk8 += 1
        else:  # 9
            tk9 += 1
    print("- " * 20)
    print('              Training info                 ')
    print("              {}  =  {}               ".format("0", k0))
    print("              {}  =  {}               ".format("1", k1))
    print("              {}  =  {}               ".format("2", k2))
    print("              {}  =  {}               ".format("3", k3))
    print("              {}  =  {}               ".format("4", k4))
    print("              {}  =  {}               ".format("5", k5))
    print("              {}  =  {}               ".format("6", k6))
    print("              {}  =  {}               ".format("7", k7))
    print("              {}  =  {}               ".format("8", k8))
    print("              {}  =  {}               ".format("9", k9))
    print("- " * 20)
    print("     Total Sample = {} ".format(m))
    print()
    print("- " * 20)
    print('              Testing info                 ')
    print("- " * 20)
    print("            {}  =  {},   {},   {:0.2f}%         ".format("0", tkp0, abs(tkp0 - tk0), 1-abs(tkp0 - tk0)/tkp0))
    print("            {}  =  {},   {},   {:0.2f}%         ".format("1", tkp1, abs(tkp1 - tk1), 1-abs(tkp1 - tk1)/tkp1))
    print("            {}  =  {},   {},   {:0.2f}%         ".format("2", tkp2, abs(tkp2 - tk2), 1-abs(tkp2 - tk2)/tkp2))
    print("            {}  =  {},   {},   {:0.2f}%         ".format("3", tkp3, abs(tkp3 - tk3), 1-abs(tkp3 - tk3)/tkp3))
    print("            {}  =  {},   {},   {:0.2f}%         ".format("4", tkp4, abs(tkp4 - tk4), 1-abs(tkp4 - tk4)/tkp4))
    print("            {}  =  {},   {},   {:0.2f}%         ".format("5", tkp5, abs(tkp5 - tk5), 1-abs(tkp5 - tk5)/tkp5))
    print("            {}  =  {},   {},   {:0.2f}%         ".format("6", tkp6, abs(tkp6 - tk6), 1-abs(tkp6 - tk6)/tkp6))
    print("            {}  =  {},   {},   {:0.2f}%         ".format("7", tkp7, abs(tkp7 - tk7), 1-abs(tkp7 - tk7)/tkp7))
    print("            {}  =  {},   {},   {:0.2f}%         ".format("8", tkp8, abs(tkp8 - tk8), 1-abs(tkp8 - tk8)/tkp8))
    print("            {}  =  {},   {},   {:0.2f}%         ".format("9", tkp9, abs(tkp9 - tk9), 1-abs(tkp9 - tk9)/tkp9))
    print("- " * 20)
    print(" Accuracy = {:0.2f}%".format(C / float(mTest)))
    print("Correct/Total = {}/{}".format(int(C), mTest))
    print(" End of Training @ {} ".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def build_knnclassifier():
    # build knn classifier
    ks = [3, 5, 7, 9]
    for k in ks:
        print(" Beginning of Training @ {} ".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        knnclassifiy(k)
        print()


def buildPredict(k=7):
    hwLabels = []
    trainingFileList = os.listdir(dataSetDir + "training")  # 加载测试数据

    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('.')[0].split("_")[1])  # return 1
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vectorV1(dataSetDir + 'training/%s' % fileNameStr)

    predictFileList = os.listdir(dataSetDir + 'predict')  # load the testing set
    mTest = len(predictFileList)
    for i in range(mTest):
        fileNameStr = predictFileList[i]
        data_file_name = dataSetDir + 'predict/%s' % fileNameStr
        vectorUnderTest = img2vectorV1(data_file_name)
        distaces_list = {}
        for j in range(m):
            distaces = distance(vectorUnderTest, trainingMat[j])
            distaces_list[distaces] = hwLabels[j]
        sorted_distance_list = sorted(distaces_list.items(), key=lambda e: e[0], reverse=False)
        gifts = get_dict_min(sorted_distance_list, k)
        print(gifts)


if __name__ == '__main__':

    method = 'buildPredict'

    if method == 'split_datasets':
        dataname = ['./datasets/knn/digit-training.txt', './datasets/knn/digit-testing.txt',
                    './datasets/knn/digit-predict.txt']
        for n in dataname:
            split_datasets(n)

    if method == 'build_knnclassifier':
        build_knnclassifier()

    if method == 'buildPredict':
        buildPredict(k=7)
