import math


# 计算sina

def Prompt_and_calculate_print():
    # first question
    width = float(input("the width of the bloodstain: "))
    length = float(input("the length of the bloodstain: "))
    results = math.asin(width / length) * 180 / math.pi
    print("the angle of the bloodstain : {:0.7f}".format(results))
    # second question
    distance = float(input("Enter the bloodstain’s distance:"))
    height = distance * math.tan(results * math.pi / 180)
    print("the height the bloodstain fell from:{:0.7f}".format(height))


Prompt_and_calculate_print()
