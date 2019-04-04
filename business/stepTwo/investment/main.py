import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# 数据导入

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m').strftime("%Y-%m")
data = pd.read_excel('Homework3.xlsx', index_col=0, date_parser=dateparse)


# ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth ', 'Utils', 'Other', 'Risk-free rate']

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


def get_msc(df):
    return df.mean(), df.std(), df.cov()


Re_top10_df = get_top_10()
# 然后计算十个资产的期望超额回报（平均数），标准差，和协方差。
# 注意，不是一个一个计算，而是作为一个整体一起计算。
Mue_top10, Std_top10, Cov_top10 = get_msc(Re_top10_df)
Rf_mean = data['Risk-free rate'].mean()
print(type(Rf_mean))
w0 = np.ones(10) / 10
print(np.matmul(w0, np.array(Rf_mean)))

exit(1)


# 接下来为了方便计算，我们定义几个方程方便我们之后计算
# 为什么这么计算，看一下我的note4。看得懂最好，看不懂就记着

# 给出一个权重，返回对应portfolio的超额收益
def W_Mue(w):
    return np.matmul(w, Mue_top10.values)  # 注意，Mue是一个series，不能和array进行运算，需要用一个.values来解决这个问题。


# 给出一个权重，返还对应portfolio的方差
def W_Var(w):
    return np.matmul(np.matmul(w, Cov_top10), w)


# 给出一个权重，返还对应portfolio的sharp ratio
def W_SP(w):
    return W_Mue(w) / (W_Var(w) ** .5)


# 给出一个权重，返还对应portfolio的sharp ratio的相反数
# 不要问为什么会有这个方程，下面就知道了
def W_SP_ng(w):
    return -W_SP(w)


from scipy.optimize import minimize

w0 = np.ones(10) / 10
w0 @ Re_top10_df.mean().values
W_SP_ng(w0)

cons = ({'type': 'eq', 'fun': lambda W: W.sum() - 1})

MVP_result = minimize(W_Var, w0, constraints=cons)

mvp_fun = MVP_result.fun

mvp_x = MVP_result.x

w_MVP = MVP_result.x

Tang_result = minimize(W_SP_ng, w0, constraints=cons)
tx = Tang_result.x

Covinv = np.linalg.inv(Cov_top10)
w = Covinv @ Mue_top10.values
w_Tang = w / w.sum()
print(w_Tang)
print(w_Tang.sum())

W_SP(Tang_result.x)

W_SP(w_Tang)
W_Mue(w_MVP)
W_Var(w_MVP)
W_Mue(w_Tang)
W_Var(w_Tang)

# 记录每一个资产的回报和标准差,大致看一下范围
std_mu = pd.DataFrame([Mue_top10 + Rf_mean, Std_top10], index=['Mu', 'Std_top10']).T

targetmue = np.linspace(0.6, 1.2, 60)

ef_std_mu = []

for Mue_top10 in targetmue:  # 对于每一个目标回报率
    cons = ({'type': 'eq', 'fun': lambda W: W_Mue(W) - Mue_top10},  # 注意，要重新修改cons的条件，要加入portfolio的回报等于目标回报
            {'type': 'eq', 'fun': lambda W: W.sum() - 1})  # 而且，这个修改，每一次换目标回报率的时候都要重新写
    result = minimize(W_Var, w0, constraints=cons)
    w = result.x
    std = result.fun ** 0.5  # 标准差即是目标值的开方
    ef_std_mu.append([Mue_top10 + Rf, std])  # 将结果分两列存在efficientp里面。
    # 之所以要list格式，只是因为append这个方程用起来比较轻松
    # 不知道这个方程干啥的，看我的note1
ef_std_mu = np.array(ef_std_mu)  # 将list转成array，因为array可以做图。
ef_std_mu

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

std_mu['95%CI_lower'] = std_mu['Mu'] - 1.64 * (std_mu['Std'] / (Re.shape[0] ** .5))
std_mu['95%CI_higher'] = std_mu['Mu'] + 1.64 * (std_mu['Std'] / (Re.shape[0] ** .5))
std_mu

# 重新定义Mue，然后把第一题的东西做一遍
Mue_L = std_mu['95%CI_lower']
Mue_H = std_mu['95%CI_higher']

# 重新定义Cov，然后把第一题的东西做一遍
Cov_Diag = np.diag(np.diag(Cov_top10))
Cov_Diag

Cov_Zdiag = Cov_top10 - Cov_Diag
Cov_Zdiag
