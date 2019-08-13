import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("12346.csv", sep=' ')
Y_data = dataframe['c']
X = dataframe['c'][:int(len(Y_data) * 0.9)].values
print(len(X), X)

y_train, y_test = Y_data[:int(len(Y_data) * 0.9)].values, Y_data[int(len(Y_data) * 0.9):].values
print(len(y_train), y_train)
print(len(y_test), y_test)


def create_baseline():
    model = Sequential()
    model.add(Dense(500, input_dim=1, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train():
    encoder = LabelBinarizer()
    encoded_Y = encoder.fit_transform(y_train)
    neural_network = create_baseline()
    history = neural_network.fit(X, encoded_Y, epochs=20, batch_size=1000, verbose=2)
    neural_network.save('neural_network.h5')
    for i in y_test:
        predicted = neural_network.predict_proba(numpy.array([i]))[0]
        rows = {}
        for key, v in enumerate(predicted):
            rows[key] = "{:0.5f}".format(v)
        top_n = 3
        rows_sorted = sorted(rows.items(), key=lambda x: x[1], reverse=True)[:top_n]
        res = ["预测：{} 概率：{}".format(i[0], i[1]) for i in rows_sorted]
        print("y: {}, 预测:{}".format(i, res))

    history_dict = history.history
    # 画图
    acc = history_dict['acc']
    loss = history_dict['loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.title('Accuracy and Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("Accuracy and loss.png")


if __name__ == "__main__":

    method = 'train'

    if method == 'train':
        # 训练模型
        train()

    if method == 'predicts':
        # 预测
        neural_network = load_model("neural_network.h5")
        for i in range(1):
            predicted = neural_network.predict_proba(numpy.array(numpy.random.randint(5, 6, 1)))[0]
            rows = {}
            for key, v in enumerate(predicted):
                rows[key] = "{:0.5f}".format(v)
            top_n = 3
            rows_sorted = sorted(rows.items(), key=lambda x: x[1], reverse=True)[:top_n]
            res = ["预测：{} 概率：{}".format(i[0], i[1]) for i in rows_sorted]
            print("|".join(res))

