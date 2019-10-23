import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd


def fl_score(logits, labels):
    '''
    Function:
        计算各分类的fl_score值
    Arguments:
    logits-- 预测值
    labels -- 真实值
    '''
    class_list = set(labels)
    for y in class_list:
        TP = np.sum(((labels == logits) & (logits == y)))
        n_pred_pos = np.sum(logits == y)
        n_real_pos = np.sum(labels == y)
        precision = recall = flscore = 0
        precision = TP / n_pred_pos
        recall = TP / n_real_pos
        fl_score = (2 * precision * recall) / (precision + recall)
        print(f"{y} 类的fl_score 为:{fl_score}")


def calc_accuracy_class(y_pred, y):
    '''
    Function:
    计算分类正确率
    Arguments:
    y_pred---预测值
    y -- 真实值
    Return:
    正确率
    '''
    y = y.reshape(-1, 1)
    m = y.shape[0]
    correct_num = np.sum((np.squeeze(y_pred) == np.squeeze(y)))
    return correct_num / m


def nomalize(X, axis):
    mean = np.mean(X, axis)
    std = np.std(X, axis)
    return (X - mean) / std, mean, std


def init_parameters(n):
    theta = np.random.randn(n, 1)
    return theta


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(y_hat, y):
    '''
    y_hat --当前阶段的预测值
    y -- 真实值
    '''
    m = y.shape[0]
    cost = -np.sum(y * np.log(y_hat) + (1 - y) * (np.log(1 - y_hat))) / m
    return cost


def gradient_descent(X, y_hat, y, theta, lr):
    '''
    y_hat --当前阶段的预测值
    y -- 真实值
    lr -- 学习速率
    '''
    d_theta = np.dot(X.T, y_hat - y)
    theta = theta - lr * d_theta
    return theta


class BasicLogicUnit:
    def __init__(self, X, y, category):
        '''
        X -- 训练样本,shape:(m,nx)
        y -- 0 or 1 shape:(m,1)
        category -- 真正的类别，即y为1时，所代表的类别
        '''
        self.X = X
        self.y = y
        self.category = category
        self.theta = init_parameters(self.X.shape[1])

    def fit(self, lr, steps):
        '''
        训练
        '''
        m, n_x = self.X.shape
        costs = []
        for step in range(steps):
            z = np.dot(self.X, self.theta)
            y_ = sigmoid(z)
            loss = compute_cost(y_, self.y)
            costs.append(loss)

            self.theta = gradient_descent(self.X, y_, self.y, self.theta, lr)

            if step % 50 == 0:
                print(f"\nAfter {step} step(s),cost is :{loss}")

        return costs

    def predict(self, X):
        '''
        预测
        '''
        z = np.dot(X, self.theta)
        return sigmoid(z)


def AUC(label, pre):

    # 计算正样本和负样本的索引，以便索引出之后的概率值
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]
    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5
    return auc / (len(pos) * len(neg))


class LogicRegressionModel:
    def __init__(self):
        self.logic_unit_list = []

    def fit(self, tain_X, train_y, learning_rate=0.005, steps=50):
        classes = set(np.squeeze(train_y))
        n_classes = len(classes)
        m, n_x = tain_X.shape
        # 根据分类的类别，一个个使用逻辑单元进行分类训练
        for c in classes:
            unit_train_y = np.where(train_y == c, 1, 0)
            logic_unit = BasicLogicUnit(x_train, unit_train_y, c)
            self.logic_unit_list.append(logic_unit)

            costs = logic_unit.fit(learning_rate, steps)
            # 绘制损失曲线
            plt.xlim(0, steps)
            plt.plot(costs)
            plt.xlabel("steps")
            plt.ylabel("costs")
            y_pred = logic_unit.predict(x_train)
            y_pred = np.where(y_pred > 0.5, 1, 0)
            acc = calc_accuracy_class(y_pred, unit_train_y)
            print(f"{c}类的准确率为:{acc}")
            print(y_pred.tolist())

    def predict(self, X):
        m = X.shape[0]
        # 为了可视化，我们将其以DataFrame的形式输出
        zeros = np.zeros((m, 1), dtype=int)
        results_pd = pd.DataFrame(zeros, columns=["result"])
        for logic_unit in self.logic_unit_list:
            prob_y = logic_unit.predict(X)
            results_pd[logic_unit.category] = prob_y
        max_indexs = np.argmax(np.array(results_pd), axis=1)
        y_ = np.array(results_pd.columns)[max_indexs]
        y_ = y_.T
        results_pd["result"] = y_
        print(results_pd.head())
        return y_


data = pd.read_csv("train.csv").head(1000)
print(data.shape)
data = data.fillna(-1)
data.drop(labels=['Date', 'Location', 'WindDir9am', 'WindSpeed3pm', 'WindGustDir', ], axis=1, inplace=True)
data.drop(labels=['WindDir3pm'], axis=1, inplace=True)
print(data.shape)
data.RainTomorrow = data.RainTomorrow.map({'No': 0, 'Yes': 1})
data.RainToday = data.RainToday.map({'No': 0, 'Yes': 1, 'nan': -1})
Y_all = np.array(data['RainTomorrow'].values)
del data['RainTomorrow']
x_train = data.iloc[2:int(data.shape[0] * 0.7), :]
x_test = data.iloc[int(data.shape[0] * 0.7):, :]
y_train = Y_all[:int(data.shape[0] * 0.7)]
y_test = Y_all[int(data.shape[0] * 0.7):]

model = LogicRegressionModel()
model.fit(x_train, y_train, learning_rate=0.05, steps=40)
y_ = model.predict(x_test)
auc = AUC(label=y_test, pre=y_)
print(auc)


