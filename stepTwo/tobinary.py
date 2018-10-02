import math


def isOdd(num):
    if num % 2 == 0:
        return True
    else:
        return False

#
# def toBinary(n):
#     times = abs(math.log(n, 2)) + 1
#     b = []
#     for i in range(int(times)):
#         s = n // 2
#         b = b + [n % 2]
#         n = s
#     b.reverse()
#     res = []
#     for i in b:
#         res.append("{}".format(i))
#     return "".join(res)


def toBinary(n):
    times = math.floor(math.log(n, 2)) + 1
    b = ''
    for i in range(int(times)):
        s = n // 2
        b += "{}".format(n % 2)
        n = s
    return b[::-1]

#
# def toBinary_2(n):
#     times = abs(math.log(n, 2)) + 1
#     s = []
#     for i in range(int(times)):
#         if isOdd(n):
#             s.append('0')
#         else:
#             s.append('1')
#         n = n // 2
#     return "".join(s[::-1])


def toBinary_2(n):
    times = abs(math.log(n, 2)) + 1
    s = ''
    for i in range(int(times)):
        if isOdd(n):
            s += '0'
        else:
            s += '1'
        n = n // 2
    return s[::-1]


def main():
    n = int(input("Please input integer: "))
    results1 = toBinary(n)
    print(results1)
    results2 = toBinary_2(n)
    print(results2)

    for i in range(33):
        if i == 0:
            print("{},{}".format(i, '0'))
        else:
            res = toBinary(i)
            print("{},{}".format(i, res))


main()
