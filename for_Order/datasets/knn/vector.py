import math
from functools import reduce, partial

"""
CSC1002
Use the following functions to implement kNN models
"""


def vector_add(v, w):
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_or(v, w):
    """boolean 'or' two vectors componentwise"""
    return [v_i or w_i for v_i, w_i in zip(v, w)]


def vector_and(v, w):
    """boolean 'and' two vectors componentwise"""
    return [v_i and w_i for v_i, w_i in zip(v, w)]


def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return sum(v_i * v_i for v_i in v)


def distance(v, w):
    s = vector_subtract(v, w)
    return math.sqrt(sum_of_squares(s))


def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))


def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def vector_sum(vectors):
    return reduce(vector_add, vectors)


def scalar_multiply(c, v):
    return [round(c * v_i, 2) for v_i in v]


def vector_mean(vectors):
    """compute the vector whose i-th element is the mean of the
    i-th elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))
