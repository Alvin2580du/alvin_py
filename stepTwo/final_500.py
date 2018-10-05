import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
data = pd.read_csv('HSCITop_month.csv', index_col=['date'], date_parser=dateparse)

sidentity_matrix = np.asmatrix(np.ones((data.shape[1], 1)))


def get_simple_returns():
    returns_tmp = {}
    columns = data.columns.tolist()
    for col in columns:
        if col == 'date':
            continue
        df = data[col]
        rate = (df - df.shift(1)) / df.shift(1)
        returns_tmp[col] = rate.values.tolist()
    returns = pd.DataFrame(returns_tmp).dropna()
    returns.to_csv("returns.csv", index=None)
    print("[*] simple returns have saved!")
    simple_returns = np.asmatrix(np.array((data.values[-1] / data.values[0]) ** (1 / len(returns)) - 1))
    return simple_returns


def get_abc(input, simple_returns):
    omg = np.asmatrix(np.cov(input.T))
    A_value = (simple_returns * np.linalg.inv(omg) * simple_returns.T)
    B_value = (simple_returns * np.linalg.inv(omg) * sidentity_matrix)
    C_value = (sidentity_matrix.T * np.linalg.inv(omg) * sidentity_matrix)
    return A_value, B_value, C_value


def get_lgw(mup, input, simple_returns, a, b, c):
    lmd = (mup * c - b) / (a * c - b ** 2)
    gam = (a - b * mup) / (a * c - b ** 2)
    omg = np.asmatrix(np.cov(input.T))
    weight1 = np.dot(np.linalg.inv(omg), simple_returns.T) * lmd
    weight2 = np.dot(np.linalg.inv(omg), sidentity_matrix) * gam
    weight = weight1 + weight2
    standard_deviation = np.sqrt((c * mup ** 2 - 2 * b * mup + a) / (a * c - b ** 2))
    return lmd.getA(), gam.getA(), weight, standard_deviation.getA()


def plot(input, simple_returns, five=False, six=False):
    A_value, B_value, C_value = get_abc(input=data, simple_returns=simple_returns)
    mups = np.array(np.arange(0.005, 0.1005, 0.005))
    lmds, gams, weights, standard_deviations = [], [], [], []
    for mup in mups:
        lmd, gam, weight, standard_deviation = get_lgw(mup, input=input, simple_returns=simple_returns, a=A_value, b=B_value, c=C_value)
        lmds.append(lmd[0][0])
        gams.append(gam[0][0])
        weights.append(weight[0][0])
        standard_deviations.append(standard_deviation[0][0])

    plt.plot(standard_deviations, mups, color='red')
    plt.xlabel('risk')
    plt.ylabel('simple Return')
    plt.title('Efficient Frontier')
    plt.savefig("Plot the efficient frontier_4.png")
    rf = 0.005
    a = np.asmatrix(simple_returns.T - rf * sidentity_matrix)
    omgs = np.asmatrix(np.cov(input.T))
    weight_new = np.linalg.inv(np.asmatrix(omgs)) * a * ((mups - rf) / (a.T * np.linalg.inv(np.asmatrix(omgs)) * a))
    standard_deviation_new = np.array(list(map(lambda x: (x * omgs * x.T).getA(), weight_new.T)))
    standard_deviation_new_s = np.sqrt(standard_deviation_new).reshape(1, 1, 20)[0][0]
    plt.plot(standard_deviation_new_s, mups, color='blue')
    if five:
        plt.savefig("the efficient frontier blue_5.png")
        plt.show()
        plt.close()
    if six:
        plt.savefig("the efficient frontier blue_6.png")
        plt.show()
        plt.close()
    return lmds, gams, weights, standard_deviations, omgs


if __name__ == "__main__":
    for method in ['build_pre', 'build_six']:
        if method == 'build_pre':
            simple_returns = get_simple_returns()
            print("simple_returns:{}".format(simple_returns.shape))
            lmds, gams, weights, standard_deviations, omgs = plot(input=data, simple_returns=simple_returns, five=True)
            print("= * ="*20)
            print("lambda:{}".format(lmds))
            print("= "*20)
            print("gamma:{}".format(gams))
            print("= "*20)
            print("weights:{}".format(weights))
            print("= "*20)
            print("standard_deviations: {}".format(standard_deviations))
            print("= "*20)
            print("omgas:{}".format(omgs))

        if method == 'build_six':
            target = pd.read_csv('Target_price.csv')
            simple_returns_new = np.array((target['Target Price'] / target['Last Price']) ** (1 / 12) - 1).reshape((1, 11))
            lmds, gams, weights, standard_deviations, omgs = plot(input=data, simple_returns=simple_returns_new, six=True)
            print("= * ="*20)
            print("lambda:{}".format(lmds))
            print("= "*20)
            print("gamma:{}".format(gams))
            print("= "*20)
            print("weights:{}".format(weights))
            print("= "*20)
            print("standard_deviations: {}".format(standard_deviations))
            print("= "*20)
            print("omgas:{}".format(omgs))