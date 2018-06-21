import numpy as np
import scipy.stats

KL = scipy.stats.entropy(x, y)
print(KL)

# 编程实现
KL = 0.0
for i in range(len(px)):
    KL += px[i] * np.log(px[i] / py[i])

print(KL)
import numpy as np
import cv

pp = cv.fromarray(p)
qq = cv.fromarray(q)
emd = cv.CalcEMD2(pp, qq, cv.CV_DIST_L2)
