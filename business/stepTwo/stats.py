"""
This module is a collection of statistical functions for analyzing
the results of a non-nominal survey question.

To test your current solution, run the `test_my_solution.py` file.
Refer to the instructions on Canavs for more information.

Author name: Zilong Wang
I have neither given nor received unauthorized assistance on
this assignment. I did not fabricate the answers to my
survey question.
"""
__version__ = 1

# 0) Question and Results
SURVEY_QUESTION = "What's the salary of your dream job annually?"
SURVEY_RESULTS = [300000, 150000, 90000, 2000000, 180000, 58000, 80000, 175000, 78000, 350000]


# 1) count
def count(inputs_list):
    try:
        i = 0
        for x in inputs_list:
            i += 1
        return i
    except:
        return 0


# 2) summate
def summate(inputs_list):
    i = 0
    for x in inputs_list:
        i += x
    return i


# 3) mean
def mean(inputs_list):
    if count(inputs_list) > 0:
        return summate(inputs_list) / count(inputs_list)
    else:
        return 0


# 4) maximum
def maximum(inputs_list):
    if count(inputs_list) < 1:
        return None
    else:
        res = -10000
        for x in inputs_list:
            if x > res:
                res = x
        return res


# 5) minimum
def minimum(inputs_list):
    if count(inputs_list) < 1:
        return None
    i = 99999999
    for x in inputs_list:
        if x < i:
            i = x
    return i


# 6) median
def median(inputs_list):
    if count(inputs_list) < 1:
        return None
    else:
        inputs_list.sort()
        return inputs_list[count(inputs_list) // 2]


# 7) square
def square(inputs_list):
    if count(inputs_list) < 0:
        return None
    return [i ** 2 for i in inputs_list]


def standard_deviation(sequence):
    if len(sequence) < 1:
        return None
    else:
        avg1 = summate(sequence) / count(sequence)
        sdsq = summate([(i - avg1) ** 2 for i in sequence])
        stdev = (sdsq / (count(sequence) - 1)) ** .5
        return stdev


def main():
    print("We asked", count(SURVEY_RESULTS), "people the following question.")
    print('"' + SURVEY_QUESTION + '"')
    print("Here are the statistical results:")
    print("\tSum:", summate(SURVEY_RESULTS))
    print("\tMean:", mean(SURVEY_RESULTS))
    print("\tMedian:", median(SURVEY_RESULTS))
    print("\tMaximum:", maximum(SURVEY_RESULTS))
    print("\tMinimum:", minimum(SURVEY_RESULTS))
    print("\tSquare:", square(SURVEY_RESULTS))
    print("\tStandard Deviation:", standard_deviation(SURVEY_RESULTS))


# The following code can be used to try out your functions.
# Uncomment each line as you implement the functions to try them out.
# When you have implemented each function, copy the output from the
#   console into a comment below.
if __name__ == "__main__":
    main()
