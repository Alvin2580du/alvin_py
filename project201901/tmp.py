import pandas as pd

dice = pd.read_excel("沣东新城四经普清查底册.xlsx")
print(dice.columns.tolist())

def get_renshu(inputs):
    res = dice[dice['统一社会信用代码']]