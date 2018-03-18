import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import FunctionTransformer

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X shape:{},y shape:{}".format(X_train.shape, y_train.shape))

X_scaled = preprocessing.scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)
m = X_scaled.mean(axis=0)
s = X_scaled.std(axis=0)
print("Mean:{}, \n Std:{}".format(m, s))
"""
X shape:(331, 10),y shape:(331,)
Mean:[ -4.46101699e-17   2.42840323e-16   2.01248887e-18  -2.12988405e-17
  -2.34790368e-17  -2.49884034e-17  -1.25780554e-17  -5.16538809e-17
  -1.34165924e-17   1.67707406e-17], 
 Std:[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
 """

# 另一种方法
scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
X_scaled = scaler.transform(X_train)
m = X_scaled.mean(axis=0)
s = X_scaled.std(axis=0)
print("Mean:{}, \n Std:{}".format(m, s))
"""
Mean:[ -4.46101699e-17   2.42840323e-16   2.01248887e-18  -2.12988405e-17
  -2.34790368e-17  -2.49884034e-17  -1.25780554e-17  -5.16538809e-17
  -1.34165924e-17   1.67707406e-17], 
 Std:[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
"""

# 最大最小归一化,归一化到[0,1]之间
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print()
ma = X_train_minmax.max(axis=0)
mi = X_train_minmax.min(axis=0)
print("max:{},\n min:{}".format(ma, mi))
"""
max:[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.],
 min:[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 """

# 最大最小归一化,归一化到[-1,1]之间
max_abs_scaler = preprocessing.MaxAbsScaler()
x_maxabs = max_abs_scaler.fit_transform(X_train)

ma = x_maxabs.max(axis=0)
mi = x_maxabs.min(axis=0)
print("max:{},\n min:{}".format(ma, mi))
"""
max:[ 1.          1.          1.          1.          1.          1.          1.
  1.          1.          0.98435481],
 min:[-0.96838121 -0.88085106 -0.52930243 -0.85122699 -0.70749565 -0.58158979
 -0.5646737  -0.41242062 -0.94384991 -1.        ]
 """

quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
percentile = np.percentile(a=X_train[:, 0], q=[0, 25, 50, 75, 100], axis=None, out=None,
                           overwrite_input=False, interpolation='linear', keepdims=False)
print()
print(percentile)
"""
[-0.10722563 -0.0382074   0.00538306  0.03807591  0.11072668]
"""

X_normalized = preprocessing.normalize(X, norm='l2', axis=1, copy=True, return_norm=False)  # l2, l1, max

print()
print("{},\n {}".format(X[0], X_normalized[0]))
"""
[ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
 -0.04340085 -0.00259226  0.01990842 -0.01764613],
 [ 0.32100597  0.42726811  0.52014127  0.18439893 -0.37283438 -0.29356288
 -0.36589885 -0.02185454  0.16784162 -0.14876892]
 """
# 另一种方法
normalizer = preprocessing.Normalizer(norm='l2', copy=True).fit(X)
X_normalized = normalizer.transform(X)
print()
print("{},\n {}".format(X[0], X_normalized[0]))
"""
[ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
 -0.04340085 -0.00259226  0.01990842 -0.01764613],
 [ 0.32100597  0.42726811  0.52014127  0.18439893 -0.37283438 -0.29356288
 -0.36589885 -0.02185454  0.16784162 -0.14876892]
 """
exit(1)

binarizer = preprocessing.Binarizer(threshold=0.0, copy=True)
binarized = binarizer.transform(X)
print()
print(binarized)

# 多项式转换
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
polyed = poly.fit_transform(X)
print(polyed)
print()
"""
[[  1.   1.   2.   3.   4.   1.   2.   3.   4.   4.   6.   8.   9.  12.
   16.]]
   """
print("{},{}, {}".format(polyed.shape, np.max(polyed), np.min(polyed)))
"""
(1, 15),16.0, 1.0
"""
# 自定义函数转换
transformer = FunctionTransformer(func=np.log1p, inverse_func=None, validate=True,
                                  accept_sparse=False, pass_y='deprecated',
                                  kw_args=None, inv_kw_args=None)

transformered = transformer.transform(X)
print()
print("{},\n {}".format(X[0], transformered[0]))
print(np.log1p(0.03807591))
"""

[ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
 -0.04340085 -0.00259226  0.01990842 -0.01764613],
 [ 0.03736891  0.04943769  0.05986782  0.02163659 -0.04523118 -0.03544146
 -0.04437083 -0.00259563  0.01971284 -0.01780367]
 0.0373689130909
 """