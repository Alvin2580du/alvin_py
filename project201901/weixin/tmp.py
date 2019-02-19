import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("./data/DataSet_HAR.csv")
print(train.columns.tolist())

fpr = train['tBodyAcc-mean()-X'].values
tpr = train['tBodyAcc-mean()-Y'].values

plt.boxplot(train['tBodyAcc-mean()-Z'].values)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.xlabel('tBodyAcc-mean()-Z')
plt.ylabel('')
plt.title("tBodyAcc-mean()-Z boxplot")
plt.legend(loc="lower right")
plt.savefig("train——scatter.png")

