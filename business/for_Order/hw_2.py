import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error


def get_data():
    data3 = pd.read_csv("./datasets/train-1000-100.csv")
    data3.iloc[:50].to_csv("./datasets/train-50(1000)-100.csv", index=None)
    data3.iloc[:100].to_csv("./datasets/train-100(1000)-100.csv", index=None)
    data3.iloc[:150].to_csv("./datasets/train-150(1000)-100.csv", index=None)


def train_model(train_df, test_df):
    reg_target = train_df['y']
    del train_df['y']
    reg_data = train_df
    rcv = RidgeCV()
    rcv.fit(reg_data, reg_target)
    data1t_y = test_df['y']
    del test_df['y']
    data1t_x = test_df
    y_pred = rcv.predict(X=data1t_x)
    mse = mean_squared_error(data1t_y, y_pred)
    return rcv.alpha_, mse


def get_lambda_1():
    data1 = pd.read_csv("./datasets/train-1000-100.csv")
    data3t = pd.read_csv("./datasets/test-1000-100.csv")
    lam1, mse1 = train_model(data1, data3t)
    return lam1, mse1


def get_lambda_2():
    data2 = pd.read_csv("./datasets/train-50(1000)-100.csv")
    data2t = pd.read_csv("./datasets/test-100-100.csv")
    lam2, mse2 = train_model(data2, data2t)
    return lam2, mse2


def get_lambda_3():
    data3 = pd.read_csv("./datasets/train-100(1000)-100.csv")
    data2t = pd.read_csv("./datasets/test-100-100.csv")
    lam3, mse3 = train_model(data3, data2t)
    return lam3, mse3


def get_lambda_4():
    data4 = pd.read_csv("./datasets/train-150(1000)-100.csv")
    data2t = pd.read_csv("./datasets/test-100-100.csv")
    lam4, mse4 = train_model(data4, data2t)
    return lam4, mse4


def get_lambda_5():
    data5 = pd.read_csv("./datasets/train-100-100.csv")
    data2t = pd.read_csv("./datasets/test-100-100.csv")
    lam5, mse5 = train_model(data5, data2t)
    return lam5, mse5


def get_lambda_6():
    data6 = pd.read_csv("./datasets/train-100-10.csv")
    data1t = pd.read_csv("./datasets/test-100-10.csv")
    lam6, mse6 = train_model(data6, data1t)
    return lam6, mse6


if __name__ == "__main__":
    lam1, mse1 = get_lambda_1()
    lam2, mse2 = get_lambda_2()
    lam3, mse3 = get_lambda_3()
    lam4, mse4 = get_lambda_4()
    lam5, mse5 = get_lambda_5()
    lam6, mse6 = get_lambda_6()

    lam = [lam1, lam2, lam3, lam4, lam5, lam6]
    mse = [mse1, mse2, mse3, mse4, mse5, mse6]

    print(lam)
    print(mse)
