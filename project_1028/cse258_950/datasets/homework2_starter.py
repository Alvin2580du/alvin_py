import numpy
from urllib.request import urlopen
import scipy.optimize
import random
from math import exp
from math import log


def parseData(fname):
    for l in urlopen(fname):
        yield eval(l)


print("Reading data...")
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print("done")


def feature(datum):
    feat = [1, datum['review/taste'], datum['review/appearance'], datum['review/aroma'], datum['review/palate'],
            datum['review/overall']]
    return feat


X = [feature(d) for d in data]
y = [d['beer/ABV'] >= 6.5 for d in data]


def inner(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


##################################################
# Logistic regression by gradient ascent         #
##################################################

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
    loglikelihood = 0
    for i in range(len(X)):
        logit = inner(X[i], theta)
        loglikelihood -= log(1 + exp(-logit))
        if not y[i]:
            loglikelihood -= logit
    for k in range(len(theta)):
        loglikelihood -= lam * theta[k] * theta[k]
    # for debugging
    # print("ll =" + str(loglikelihood))
    return -loglikelihood


# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
    dl = [0] * len(theta)
    for i in range(len(X)):
        logit = inner(X[i], theta)
        for k in range(len(theta)):
            dl[k] += X[i][k] * (1 - sigmoid(logit))
            if not y[i]:
                dl[k] -= X[i][k]
    for k in range(len(theta)):
        dl[k] -= lam * 2 * theta[k]
    return numpy.array([-x for x in dl])


X_train = X
y_train = y


##################################################
# Train                                          #
##################################################

def train(lam):
    theta, _, _ = scipy.optimize.fmin_l_bfgs_b(f, [0] * len(X[0]), fprime, pgtol=10, args=(X_train, y_train, lam))
    return theta


##################################################
# Predict                                        #
##################################################

def performance(theta):
    scores = [inner(theta, x) for x in X]
    predictions = [s > 0 for s in scores]
    correct = [(a == b) for (a, b) in zip(predictions, y_train)]
    acc = sum(correct) * 1.0 / len(correct)
    return acc


##################################################
# Validation pipeline                            #
##################################################

lam = 1.0

theta = train(lam)
acc = performance(theta)
print("lambda = " + str(lam) + ":\taccuracy=" + str(acc))
