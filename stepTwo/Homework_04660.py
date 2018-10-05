import numpy as np  # arrays and functions which operate on array
from numpy import linspace, arange
import matplotlib.pyplot as plt  # normal plotting
# import seaborn as sns # Fancy plotting
# import pandas as pd # Data input and manipulation
from random import random, randint, uniform, choice, sample, shuffle, seed
from collections import Counter


# Calculating permutations and combinations efficiently
def P(N, K):
    res = 1
    for i in range(K):
        res *= N
        N = N - 1
    return res


def C(N, K):
    if K < N / 2:
        K = N - K
        X = [1] * (K + 1)
    for row in range(1, N - K + 1):
        X[row] *= 2
        for col in range(row + 1, K + 1):
            X[col] = X[col] + X[col - 1]
    return X[K]


# Round to 4 decimal places for printing numeric answers.
def round4(x):
    return round(x + 0.00000000001, 4)


def round4_list(L):
    return [round4(x) for x in L]


# This function takes a list of outcomes and a list of probabilities and
# draws a chart of the probability distribution.
def draw_distribution(Rx, fx, title='Probability Distribution for X'):
    plt.bar(Rx, fx, width=1.0, edgecolor='black')
    plt.ylabel("Probability")
    plt.xlabel("Outcomes")
    if Rx[-1] - Rx[0] < 30:
        ticks = range(Rx[0], Rx[-1] + 1)
        plt.xticks(ticks, ticks)
        plt.title(title)
        plt.show()


"""
Problem One
S = {BBBBB,ABBBB,AABBB,AAABB,AAAAB,AAAAA}
Rx = {0, 1, 2, 3, 4, 5}
fx = {C(13,0)*C(39,5)/C(52,5), C(13,1)*C(39,4)/C(52,5),C(13,2)*C(39,3)/C(52,5),C(13,3)*C(39,2)/C(52,5),C(13,4)*C(39,1)/C(52,5),C(13,5)*C(39,0)/C(52,5)}
draw_distribution([0, 1, 2, 3, 4, 5],[C(13,0)*C(39,5)/C(52,5), C(13,1)*C(39,4)/C(52,5),C(13,2)*C(39,3)/C(52,5),C(13,3)*C(39,2)/C(52,5),C(13,4)*C(39,1)/C(52,5),C(13,5)*C(39,0)/C(52,5)] )

"""

"""
Problem Two
S = {BBBBBR, BBBBR, BBBR, BBR, BR, R}
Rx = {6, 5, 4, 3, 2, 1}
fx = {1/7*1/6*1/5*1/4*1/3, 1/7*1/6*1/5*1/4*2/3, 1/7*1/6*1/5*2/4, 1/7*1/6*2/5, 1/7*2/6, 2/7}
draw_distribution([6, 5, 4, 3, 2, 1], [1/7*1/6*1/5*1/4*1/3, 1/7*1/6*1/5*1/4*2/3, 1/7*1/6*1/5*2/4, 1/7*1/6*2/5, 1/7*2/6, 2/7]
"""

"""
Problem Three
S = {RW, RB, WB, RR, WW, BB}
Rx = {-1, -2, 3, -6, -4, 2}
fx = {5/15*5/14, 5/15*5/14, 5/15*5/14, 5/15*4/14, 5/15*4/14, 5/15*4/14}
draw_distribution([-1, -2, 3, -6, -4, 2], [5/15*5/14, 5/15*5/14, 5/15*5/14, 5/15*4/14, 5/15*4/14, 5/15*4/14])

"""

"""
Problem Four
You do not need to give the sample space S.
Rx = {0, 1}
fx = {1/2, 1/2}

draw_distribution([0,1], [1/2, 1/2])
"""

"""
Problem Five
Rx = {-1, 2, 5}
fx = { 1/4, 1/2, 1/4 }

"""

"""Problem Six
(a):

Rx = {0, 1, 2, 3, 4, 5}
fx = {C(13,0)*C(39,5)/C(52,5), C(13,1)*C(39,4)/C(52,5),C(13,2)*C(39,3)/C(52,5),C(13,3)*C(39,2)/C(52,5),C(13,4)*C(39,1)/C(52,5),C(13,5)*C(39,0)/C(52,5)}

Ex = C(13, 0) * C(39, 5) / C(52, 5) * 0 + C(13, 1) * C(39, 4) / C(52, 5) * 1 + C(13, 2) * C(39, 3) / C(52, 5) * 2 + C(
    13, 3) * C(39, 2) / C(52, 5) * 3 + C(13, 4) * C(39, 1) / C(52, 5) * 4 + C(13, 5) * C(39, 0) / C(52, 5) * 5=1.25


VarX = round4((0 - Ex) ** 2 * C(13, 0) * C(39, 5) / C(52, 5) + (1 - Ex) ** 2 * C(13, 1) * C(39, 4) / C(52, 5) + 
(2 - Ex) ** 2 * C(13, 2) * C(39, 3) / C(52, 5) + (3 - Ex) ** 2 * C(13, 3) * C(39, 2) / C(52, 5) +
 (4 - Ex) ** 2 * C(13, 4) * C(39, 1) / C(52, 5) + (5 - Ex) ** 2 * C(13, 5) * C(39, 0) / C(52, 5)) =0.864

import math
sigmax = round4(math.sqrt(VarX))


"""

"""
Problem Seven

(a):
no, because:
Rx = {2, 6, 10, 16, 21}
fx = {1/5*1/4, 2/5*2/4, 1/5*1/4, 1/5*2/4, 1/5*2/4}
Ex = 2*1/5*1/4 + 6*2/5*2/4+10*1/5*1/4+16*1/5*2/4+21*1/5*2/4=5.5 
5.5 is less than 10, so not fair for player.

"""

"""
Problem Eight
(a):
Rx = {-1, 74}
fx = {1/100*99/100, 1/100*1/100}

Ex1 = 1/100*1/100*74-1/100*99/100

Rx = {-1, 2}
fx = {1/4*3/4, 1/4*1/4}
Ex2 = 1/4*1/4*2 - 1/4*3/4

(b):
import math

var1 = (-1-Ex1)**2*1/100*99/100 + (74-Ex1)**2*1/100*1/100 
sd1 = math.sqrt(var1)

var2 = (-1-Ex2)**2*1/4*3/4 + (74-Ex2)**2*1/4*1/4
sd2 = math.sqrt(var2)


(c):
long time : Bolita
a couple of times：Keno
"""

"""
Problem Nine
(a):
standardized random varibles The salesperson in store A:
(10-13) / 5 = -0.6

standardized random varibles The salesperson in store B:
(6-7) / 4 = -0.25

so The salesperson in store B should Mr. Norton hire.

(b): A should sell 12 sets.
"""

"""
Problem Ten
(a): 
Rx = {-n, n}
fx = {1/2, 1/2}
Ex = 1/2*n - 1/2*n = 0
(b): 

"""
