import random


def csort(alist):
    n = max(alist) - min(alist) + 1
    helperList = [0] * n

    for i in range(len(alist)):
        a = alist[i] - min(alist)
        b = alist.count(alist[i])
        helperList[a] = b

    for j in range(1, n):
        helperList[j] += helperList[j - 1]

    sortedList = [0] * len(alist)

    for k in range(len(alist)):
        c = alist[k] - min(alist)
        helperList[c] = helperList[c] - 1
        d = helperList[c]
        sortedList[d] = alist[k]

    return sortedList


def main():
    size = int(input("Enter the size of your list:"))
    alist = []
    for i in range(size):
        alist.append(random.randint(1, 100))

    print("alist:", alist)
    print("sorted list:", csort(alist))


main()
