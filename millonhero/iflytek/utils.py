
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

enc = OneHotEncoder()

data = []
data.append({"a": '北京', 'b': '西安', 'c':'成都'})
data.append({"a": '北京', 'b': '上海', 'c':'成都'})
data.append({"a": '北京', 'b': '广州', 'c':'西安'})
data.append({"a": '北京', 'b': '上海', 'c':'广州'})
print(data)

df = pd.DataFrame(data)
print(df)
del df['a']
del df['c']

df_results = enc.fit_transform(df).toarray()
print(df_results)
print(df_results.shape)

# 高维稀疏特征，通过TruncatedSVD方法降维
tsvd = TruncatedSVD(n_components=1)
decomposition_feature = tsvd.fit_transform(df_results)
mm = MinMaxScaler()
decomposition_feature = mm.fit_transform(decomposition_feature)
print(decomposition_feature)

print('降维后数据方差解释率为：', tsvd.explained_variance_ratio_.sum())
