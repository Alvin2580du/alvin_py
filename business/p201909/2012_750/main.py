import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.utils import shuffle
from sklearn.decomposition import PCA


def plot(train_sizes, train_acc, test_acc, save_name, label):
    # 绘制效果
    plt.plot(train_sizes, train_acc, color='blue', marker='o', markersize=5, label='training {}'.format(label))
    plt.plot(train_sizes, test_acc, color='green', linestyle='--', marker='s', markersize=5,
             label='test {}'.format(label))
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel(label)
    plt.legend(loc='lower right')
    plt.savefig(save_name)
    plt.close()


def train(features):
    data2 = pd.read_excel("归一化数据超注浆.xlsx", header=None)
    data3 = pd.read_excel("归一化数据欠注浆.xlsx", header=None)
    data4 = pd.read_excel("归一化数据正常注浆.xlsx", header=None)
    print('各类数据集的大小')
    print(data2.shape, data3.shape, data4.shape)
    label = [1] * data2.shape[0] + [-1] * data3.shape[0] + [0] * data4.shape[0]

    data = pd.concat((data2.iloc[:, :features], data3.iloc[:, :features], data4.iloc[:, :features]), axis=0)
    data = shuffle(data)
    ss = StandardScaler()
    std_data = ss.fit_transform(data)
    # selector = SelectKBest(score_func=f_classif, k=100)
    # newData = selector.fit_transform(std_data, label)
    # print(newData.shape)
    # support = selector.get_support(True)
    # df_support = pd.DataFrame(support)
    # df_support.to_csv("support_{}.csv".format(features), index=None)
    # print("选择的特征数量", df_support.shape[0])
    pca = PCA(n_components=20)
    newData = pca.fit_transform(std_data, label)
    le = LabelEncoder()
    y = le.fit_transform(label)  # 类标整数化
    # 划分训练集合测试集
    train_acc = []
    test_acc = []

    train_loss = []
    test_loss = []

    x_label = []

    X_train, X_test, y_train, y_test = train_test_split(newData, y, test_size=0.2, random_state=1)
    print(X_train.shape)

    acc_ = 0
    for train_size in range(50, X_train.shape[0], 300):
        rate_ = train_size / X_train.shape[0]
        X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, test_size=rate_, random_state=1)
        x_label.append(train_size)
        mlp = MLPClassifier(solver='adam', activation='logistic', max_iter=1000, learning_rate='constant',
                            alpha=1e-9, hidden_layer_sizes=(50000, 50), random_state=1, verbose=False)
        mlp.fit(X_train_small, y_train_small)
        tr_acc = mlp.score(X_train_small, y_train_small)  # 训练集准确率
        train_acc.append(tr_acc)

        tr_loss = mlp.loss_  # 训练集损失
        train_loss.append(tr_loss)

        test_pred = mlp.predict(X_test)
        te_acc = accuracy_score(y_test, test_pred)    # 测试集准确率
        test_acc.append(te_acc)
        if te_acc > acc_:
            acc_ = te_acc
            joblib.dump(mlp, 'mlp_{}.model'.format(features))

        te_loss = zero_one_loss(y_test, test_pred)  # 测试集损失
        test_loss.append(te_loss)

        print(
            "tr_loss:{:0.2f}， tr_acc：{:0.2f}, te_acc:{:0.2f}, te_loss:{:0.2f}".format(tr_loss, tr_acc, te_acc, te_loss))

    df1 = pd.DataFrame({"train_acc": train_acc, "test_acc": test_acc, 'x_label': x_label})
    df2 = pd.DataFrame({"train_loss": train_loss, "test_loss": test_acc, 'x_label': x_label})

    df1.to_csv('df1_{}.csv'.format(features), index=None)
    df2.to_csv('df2_{}.csv'.format(features), index=None)


def build_plot(features=1024):
    df1 = pd.read_csv("df1_{}.csv".format(features))
    plot(df1['x_label'], df1['train_acc'], df1['test_acc'], save_name='acc_{}.png'.format(features), label='accuracy')

    df1 = pd.read_csv("df2_{}.csv".format(features))
    plot(df1['x_label'], df1['train_loss'], df1['test_loss'], save_name='loss_{}.png'.format(features), label='loss')


if __name__ == '__main__':

    method = 'train'

    if method == 'train':
        features = 1024
        train(features=features)

    if method == 'build_plot':
        features = 1024
        build_plot(features=features)

    if method == 'pred':
        # 预测新数据
        file_name = "./iriss/1024测试.xlsx"
        features = 1024
        df_test = pd.read_excel(file_name, header=None)

        support = pd.read_csv("support_{}.csv".format(features)).values
        for inx in df_test.columns.tolist():
            if inx not in support:
                del df_test[inx]
        mlp = joblib.load('mlp_{}.model'.format(features))
        new_pred = mlp.predict(df_test)
        print('new_pred', new_pred)

        df_test = pd.read_excel(file_name, header=None)
        df_test['预测'] = new_pred
        df_test.to_excel("new_pred_{}.xlsx".format(features), index=None)



