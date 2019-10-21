import numpy as np


def loaddata():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    with open('./data/wdbc_data.csv', 'r') as fr:
        lines = fr.readlines()
        num = -1
        for line in lines:
            line_spl = line.split(',')
            y_ = 1.0
            if line_spl[1] == 'B':
                y_ = -1.0
            tmp_ = [float(i) for i in line_spl[2:]]
            num += 1
            if num < 300:  # 前300 作为训练集
                train_y.append(y_)
                train_x.append(tmp_)
            else:
                test_y.append(y_)
                test_x.append(tmp_)
    return train_x, train_y, test_x, test_y


def stump_build(train_x, train_y, D):
    # 构建单层决策树
    data2matrix = np.mat(train_x)
    labelMat = np.mat(train_y).T
    m, n = data2matrix.shape
    numSteps = 10.0
    stump_rows = {}
    best_label = np.mat(np.zeros((m, 1)))
    minError = float('inf')
    # 遍历所有特征
    for i in range(n):
        # 找到(每列)特征中的最小值和最大值
        rangeMin = data2matrix[:, i].min()
        rangeMax = data2matrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['less', 'greater']:
                thresh = (rangeMin + float(j) * stepSize)  # 计算阈值
                predictedVals = np.ones((np.shape(data2matrix)[0], 1))
                if inequal == 'less':
                    # 如果小于阈值则赋值为-1
                    predictedVals[data2matrix[:, i] <= thresh] = -1.0
                else:
                    # 如果大于阈值则赋值为1
                    predictedVals[data2matrix[:, i] > thresh] = 1.0
                errArr = np.mat(np.ones((m, 1)))  # 初始化误差矩阵
                errArr[predictedVals == labelMat] = 0  # 分类正确的，赋值为0
                weightedError = D.T * errArr  # 计算误差

                if weightedError < minError:  # 找到误差最小的分类方式
                    minError = weightedError
                    best_label = predictedVals
                    stump_rows['dim'] = i
                    stump_rows['thresh'] = thresh
                    stump_rows['ineq'] = inequal
    return stump_rows, minError, best_label


def adaboost_model(train_x, train_y):
    weakClassArr = []
    m = np.shape(train_x)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 样本权重，每个样本权重相等，即1/n
    adalabels = np.mat(np.zeros((m, 1)))  # 初始化为全零列
    train_error = []
    for i in range(numIt):
        # 构建单层决策树
        stump_rows, error, label_est = stump_build(train_x, train_y, D)
        # 计算弱学习算法权重alpha，使error不等于0，因为分母不能为0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        stump_rows['alpha'] = alpha  # 存储弱学习算法权重
        weakClassArr.append(stump_rows)  # 存储单层决策树
        expon = np.multiply(-1 * alpha * np.mat(train_y).T, label_est)  # 计算e的指数项
        D = np.multiply(D, np.exp(expon))  # 计算递推公式的分子
        D = D / D.sum()  # 根据样本权重公式，更新样本权重
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        adalabels += alpha * label_est  # 以下为错误率累计计算
        aggErrors = np.multiply(np.sign(adalabels) != np.mat(train_y).T, np.ones((m, 1)))  # 计算误差
        errorRate = aggErrors.sum() / m
        train_error.append(errorRate)
        if errorRate == 0.0:  # 误差为0退出循环
            break
    return weakClassArr, adalabels, train_error


def adaboost_predicts(test_x, classifier_data, test_y):
    data2matrix = np.mat(test_x)
    m = np.shape(data2matrix)[0]
    adalabels = np.mat(np.zeros((m, 1)))
    test_error = []

    for i in range(len(classifier_data)):  # 遍历所有分类器进行分类
        label_est = np.ones((np.shape(data2matrix)[0], 1))
        if classifier_data[i]['ineq'] == 'less':
            # 如果小于阈值则赋值为-1
            label_est[data2matrix[:, classifier_data[i]['dim']] <= classifier_data[i]['thresh']] = -1.0
        else:
            # 如果大于阈值则赋值为1
            label_est[data2matrix[:, classifier_data[i]['dim']] > classifier_data[i]['thresh']] = 1.0
        error = classifier_data[i]['alpha'] * label_est
        test_error.append(sum(error) / len(error))

        adalabels += error

    return np.sign(adalabels), test_error


if __name__ == "__main__":

    numIt = 294

    for method in ["my_adaboost", 'build_in_method']:

        if method == 'my_adaboost':
            import matplotlib.pyplot as plt

            train_x, train_y, test_x, test_y = loaddata()
            weakClassArr, adalabels, train_error = adaboost_model(train_x, train_y)
            aggClassESt, test_error = adaboost_predicts(test_x, weakClassArr, test_y)
            value = range(numIt)
            plt.figure(figsize=(10, 8))
            plt.plot(value, train_error, 'go-', label='train error')
            plt.legend()
            plt.savefig('train error.png')
            plt.close()

            plt.figure(figsize=(10, 8))
            plt.plot(value, test_error, 'r--', label='test error')
            plt.legend()
            plt.savefig('test error.png')
            plt.close()

            fw = open('results.txt', 'w', encoding='utf-8')
            # 保存预测结果
            fw.writelines('predict,y_true,test_x' + '\n')

            for x, y, y_ in zip(test_x, test_y, aggClassESt):
                rows = "{},{},{}".format(y, str(y_).replace(']', '').replace('[', ''), x)
                fw.writelines(rows + '\n')

            acc = 0  # 计算预测准确率
            for i, j in zip(test_y, aggClassESt):
                if i == j[0][0]:
                    acc += 1
            print(" my_adaboost acc: {:0.5f}".format(acc / len(test_y)))

        if method == 'build_in_method':
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import accuracy_score

            train_x, train_y, test_x, test_y = loaddata()
            bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                                     algorithm="SAMME", n_estimators=200, learning_rate=0.8)
            bdt.fit(train_x, train_y)
            y_pred = bdt.predict(test_x)
            print("build_in_method acc: {:0.5f}".format(accuracy_score(y_true=test_y, y_pred=y_pred)))
