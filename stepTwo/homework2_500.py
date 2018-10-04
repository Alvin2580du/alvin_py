import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
1. Within Hang Seng vendor classification, download the monthly price series of the largest stocks 
(in terms of weight within the corresponding sector indices) from 1-Jan-2013 to 31-Dec-2017 (there should be 11 stocks)
2. Use Pandas to read the prices and calculate the corresponding simple returns
3. Assuming the expected return to be the same as their 5-year monthly geometric-average, 
follow the equations in Lecture 2, find A, B, C
4. Assuming a range of 𝜇𝑝 from 0.5% to 10% (step 0.5%), find the corresponding 𝜆 and 𝛾 for each 𝜇𝑝, 
the weights of the portfolio (for each stock) for each 𝜇𝑝, and the corresponding standard deviation of the efficient
 portfolios. Plot the efficient frontier with correct x-label and y-label. Color the curved efficient frontier as ‘red’
Assume Hong Kong risk-free rate at 0.5%
5. With the presence of risk-free rate, construct the efficient frontier using the same range of 𝜇𝑝 as Q4. 
Plot the efficient frontier on the same graph as Q4 (a straight efficient frontier). Color it with ‘Blue’
6. Now suppose instead of using the previous 5-year monthly geometric average to estimate the expected return for next month,
 you estimate it using the price-target of the average analyst estimate from Bloomberg.
  Redo Q4 and Q5 and comments on your results. Note: Price targets are usually 12-month from now, 
  you need to convert it back to monthly (geometrically)
"""


def parse_datasets():
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
    data = pd.read_csv('hscitop_split.csv', index_col=['date'], date_parser=dateparse)
    data_sub = data['2013':'2017']
    data_sub.to_csv("hscitop_2013_2017.csv")


def get_simple_returns():
    data_sub = pd.read_csv("hscitop_2013_2017.csv")
    columns = data_sub.columns.tolist()
    for col in columns:
        if col == 'date':
            continue
        df = data_sub[col]
        rate = (df - df.shift(1)) / df.shift(1)
        rate.to_csv("simple_returns_{}.csv".format(col), index=None)


mu_p = [0.005, 0.010, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
        0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]


def parse_date(input):
    res = input[:7]
    return res


def get_geometric_average():
    data = pd.read_csv("hscitop_2013_2017.csv")
    data['datetime'] = data['date'].apply(parse_date)
    del data['date']
    k = 0
    save = []
    for x, y in data.groupby(by='datetime'):
        y_mean = y.mean()
        rows = {}
        Equity_HK_1398 = y_mean['1398 HK Equity']
        rows['Equity_HK_1398'] = Equity_HK_1398
        Equity_HK_700 = y_mean['700 HK Equity']
        rows['Equity_HK_700'] = Equity_HK_700
        Equity_HK_16 = y_mean['16 HK Equity']
        rows['Equity_HK_16'] = Equity_HK_16
        Equity_HK_857 = y_mean['857 HK Equity']
        rows['Equity_HK_857'] = Equity_HK_857
        Equity_HK_1 = y_mean['1 HK Equity']
        rows['Equity_HK_1'] = Equity_HK_1
        Equity_HK_1928 = y_mean['1928 HK Equity']
        rows['Equity_HK_1928'] = Equity_HK_1928
        Equity_HK_3 = y_mean['3 HK Equity']
        rows['Equity_HK_3'] = Equity_HK_3
        Equity_HK_2018 = y_mean['2018 HK Equity']
        rows['Equity_HK_2018'] = Equity_HK_2018
        Equity_HK_914 = y_mean['914 HK Equity']
        rows['Equity_HK_914'] = Equity_HK_914
        Equity_HK_1099 = y_mean['1099 HK Equity']
        rows['Equity_HK_1099'] = Equity_HK_1099
        Equity_HK_2319 = y_mean['2319 HK Equity']
        rows['Equity_HK_2319'] = Equity_HK_2319
        save.append(rows)
        k += 1
    geometric_average = pd.DataFrame(save)
    geometric_average.to_csv("geometric average.csv", index=None)


def get_abc(R):
    R_ = R
    cov_omu = np.cov(R_, R_.T)
    A = R.T * np.linalg.inv(cov_omu) * R
    B = R.T * np.linalg.inv(cov_omu) * np.eye(11)
    C = np.eye(11).T * np.linalg.inv(cov_omu) * np.eye(11)
    return A, B, C


def build_three():
    data = pd.read_csv("geometric average.csv")

    for col in data.columns.tolist():
        company = data[col]
        A, B, C = get_abc(company)
        print(A, C, B)


def get_lambda(mup, a, b, c):
    lam = (mup * c - b) / (a * c - np.square(b))
    gam = (a - b * mup) / (a * c - np.square(b))
    return lam, gam


def plot_qianxian():
    plt.rcParams['axes.unicode_minus'] = False
    x = np.linspace(-30, 30, 10000)
    y = (x ** 2 - 5 * x + 10)  # 方程式
    z = (-5 * x + 10)  # 二次函数的0点切线方程
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, color="red", linewidth=2, label='fangcheng')  # 方程
    plt.plot(x, z, color="blue", label='qiexian')
    plt.xlabel("Time(s)")
    plt.ylabel("Volt")
    plt.title("PyPlot First Example")
    plt.ylim(-100, 500)
    plt.legend(loc='best')
    plt.show()


