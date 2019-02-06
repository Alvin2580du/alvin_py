import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("./data/DataSet_HAR.csv")

fpr = train['tBodyAcc-mean()-X'].values
tpr = train['tBodyAcc-mean()-Y'].values

plt.scatter(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('tBodyAcc-mean()-X')
plt.ylabel('tBodyAcc-mean()-Y')
plt.legend(loc="lower right")
plt.savefig("train——scatter.png")

