import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import OrderedDict

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m').strftime("%Y-%m")
data = pd.read_excel('Homework3.xlsx', index_col=0, date_parser=dateparse)


def get_top_10():
    Re_top10 = []
    for x, y in data.iterrows():
        rows = OrderedDict()
        rows['NoDur'] = y['NoDur'] - y['Risk-free rate']
        rows['Durbl'] = y['Durbl'] - y['Risk-free rate']
        rows['Durbl'] = y['Durbl'] - y['Risk-free rate']
        rows['Manuf'] = y['Manuf'] - y['Risk-free rate']
        rows['Enrgy'] = y['Enrgy'] - y['Risk-free rate']
        rows['HiTec'] = y['HiTec'] - y['Risk-free rate']
        rows['Telcm'] = y['Telcm'] - y['Risk-free rate']
        rows['Shops'] = y['NoDur'] - y['Risk-free rate']
        rows['Hlth '] = y['Hlth '] - y['Risk-free rate']
        rows['Utils'] = y['Utils'] - y['Risk-free rate']
        rows['Other'] = y['Other'] - y['Risk-free rate']
        Re_top10.append(rows)

    Re_top10_df = pd.DataFrame(Re_top10, index=data.index)
    return Re_top10_df


Re = get_top_10()


def mean():
    return Re.mean()


def std():
    return Re.std()


def cov():
    return Re.cov()


def rf():
    return data['Risk-free rate'].mean()


def get_mue(w):
    return np.matmul(w, Re.mean().values)


def get_var(w):
    return np.matmul(np.matmul(w, cov()), w)


def get_sp(w):
    return get_mue(w) / (get_var(w) ** .5)


def get_spNg(w):
    return -get_sp(w)


w0 = np.ones(10) / 10

results1 = np.matmul(w0, Re.mean())
print(results1)
w0_sp_ng = get_spNg(w0)
print(w0_sp_ng)
cons = ({'type': 'eq', 'fun': lambda W: W.sum() - 1})
MVP_result = minimize(get_var, w0, constraints=cons)
print(MVP_result)
results2 = MVP_result.fun
results3 = MVP_result.x

Tang_result = minimize(get_spNg, w0, constraints=cons)
results4 = Tang_result.x

w = np.matmul(np.linalg.inv(data.iloc[:, :10].subtract(data['Risk-free rate'], axis='index').cov()), mean().values)
w_Tang = w / w.sum()
print(w_Tang)
wt_sum = w_Tang.sum()
print(wt_sum)
get_sp(Tang_result.x)
get_sp(w_Tang)

#  part 2
get_mue(results3)
get_var(results3)
get_var(w_Tang)
get_var(w_Tang)

#
std_mu = pd.DataFrame([mean() + rf(), std()], index=['Mu', 'Std']).T
print(std_mu)

targetmue = np.linspace(0.6, 1.2, 60)

# 设定目标回报率，这个方程的意思，在0.6，到1.4，平均分成40个点，形成一个列矩阵。

ef_std_mu = []
# 创建一个list，用来记录每一个目标回报率对应的标准差。

for mue in targetmue:  # 对于每一个目标回报率
    cons = ({'type': 'eq', 'fun': lambda W: get_mue(W) - mue},
            {'type': 'eq', 'fun': lambda W: W.sum() - 1})
    result = minimize(get_var, w0, constraints={'type': 'eq', 'fun': lambda W: W.sum() - 1})
    w = result.x
    std = result.fun ** 0.5
    ef_std_mu.append([mue + rf(), std])

ef_std_mu = np.array(ef_std_mu)

plt.xlabel('standard deviation')
plt.ylabel('expected return')
plt.title('Efficient Frontier: Mu-Std ')

plt.scatter(ef_std_mu[:, 1], ef_std_mu[:, 0])
for i in std_mu.index:
    plt.text(std_mu.loc[i, 'Std'], std_mu.loc[i, 'Mu'], i)

plt.xlabel('standard deviation')
plt.ylabel('expected return')
plt.title('Efficient Frontier: Mu-Std ')

plt.scatter(ef_std_mu[:, 1] ** 2, ef_std_mu[:, 0])
for i in std_mu.index:
    plt.text(std_mu.loc[i, 'Std'] ** 2, std_mu.loc[i, 'Mu'], i)
plt.savefig("plt.png")
plt.show()

# part 3
std_mu['95%CI_lower'] = std_mu['Mu'] - 1.64 * (std_mu['Std'] / (Re.shape[0] ** .5))
std_mu['95%CI_higher'] = std_mu['Mu'] + 1.64 * (std_mu['Std'] / (Re.shape[0] ** .5))
Mue_L = std_mu['95%CI_lower']
Mue_H = std_mu['95%CI_higher']

# Part 4

Cov_Diag = np.diag(np.diag(cov()))
Cov_Zdiag = cov() - Cov_Diag

w0 = np.ones(10) / 10

results1 = np.matmul(w0, Re.mean())
print(results1)
w0_sp_ng = get_spNg(w0)
print(w0_sp_ng)
cons = ({'type': 'eq', 'fun': lambda W: W.sum() - 1})
MVP_result = minimize(get_var, w0, constraints=cons)
print(MVP_result)
results41 = MVP_result.fun
results42 = MVP_result.x

Tang_result = minimize(get_spNg, w0, constraints=cons)
results43 = Tang_result.x

w = np.matmul(np.linalg.inv(Cov_Zdiag), mean().values)
w_Tang = w / w.sum()
print(w_Tang)
print(w_Tang.sum())
get_sp(Tang_result.x)
get_sp(w_Tang)
