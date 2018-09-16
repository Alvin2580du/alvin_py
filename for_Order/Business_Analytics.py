import pandas as pd


def Exercise_1():
    k = int(input("请输入一个整数："))
    assert isinstance(k, int)
    left = sum(range(1, k + 2))
    right = int(((k + 1) * ((k + 1) + 1)) / 2)
    print("left:{}, right:{}".format(left, right))
    assert left == right


def xiaoshu(x):
    return "{:0.2f}".format(x)


def Exercise_2():
    data = pd.read_csv("wiki_data.csv")
    data['people-per-km'] = data['Population'] / data['Totalarea(km)']
    data['people-per-miles'] = data['Population'] / (data['Totalarea(km)'] * 1000)
    data['people-per-km'] = data['people-per-km'].apply(xiaoshu)
    data['people-per-miles'] = data['people-per-miles'].apply(xiaoshu)
    df = pd.DataFrame(dtype=float)
    df['people-per-km'] = data['people-per-km']
    df['people-per-miles'] = data['people-per-miles']
    df['Country'] = data['Country']
    print(df.sort_values(by='people-per-miles'))


def Exercise_3():
    import numpy as np
    import statsmodels.api as sm
    x = np.arange(-10, 10)
    y = 2 * x + np.random.normal(size=len(x))
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    fit = model.fit()
    print("y={}".format(fit.params[0]), "+", "{}x".format(fit.params[1]))


Exercise_1()
print("===========================================")
Exercise_2()
print("===========================================")
Exercise_3()


