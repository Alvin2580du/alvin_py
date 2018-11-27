""""
Python常用文件读写操作总结

author: Alvin

time: 2018-01-01

"""

import pandas as pd
import numpy as np


def save_arr2file():
    data = np.array(np.arange(0, 100), dtype=np.int).reshape((20, 5))
    df = pd.DataFrame(data)
    df.to_csv("test_file.txt", index=None, header=None)


def read_file(file_name, mode="read"):
    fr = open(file_name, 'r')
    if mode == "read":
        lines = fr.read()
        # read方法每次只读取一个字符
        print("当前方法的读取行数为： {}".format(len(lines)))
    elif mode == "readline":
        lines = fr.readline()
        # readline 每次读取一行
        print("当前方法的读取行数为： {}".format(len(lines)))

    else:
        lines = fr.readlines()
        # readlines 一次读取所有行
        print("当前方法的读取行数为： {}".format(len(lines)))

    for line in lines:
        print("执行的方法为： {}， 当前行输出为：{}".format(mode, line))
    fr.close()


def write_file(file_name, mode='write', write_way='w'):
    s1 = '今天是2018年元旦'
    s2 = '祝大家新年快乐！'
    s3 = '工作顺利，事事顺心！'

    data2save = [s1, s2, s3]
    if mode == 'write':
        fw = open(file_name, '{}'.format(write_way), encoding='utf-8')
        for i in data2save:
            fw.write(i + "\n")
    elif mode == 'writelines':
        fw = open(file_name, '{}'.format(write_way), encoding='utf-8')
        for i in data2save:
            fw.writelines(i + "\n")


def with_operator(file_name, mode='w'):
    """
    with 语句的使用
    with 语句可以自动关闭文件，不用显式的调用close方法。
    """
    if mode == 'r':
        with open(file_name, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                print("Current line: {}".format(line))
    elif mode == 'w':
        for x in range(5):
            with open(file_name, 'a+') as fw:
                fw.writelines(str(x) + "\n")
    else:
        raise NotImplementedError("No This method, please choose : w or r")


if __name__ == "__main__":

    method = "write"
    if method == 'save_data':
        save_arr2file()
    if method == "read":
        for i in ['read', 'readline', 'readlines']:
            read_file(file_name="test_file.txt", mode="{}".format(i))
            print("================Next===================")

    if method == 'write':
        for i in ['write', 'writelines']:
            write_file("test_w_file_{}.txt".format(i), mode="{}".format(i))
        for j in ['w', 'a+']:
            # "w" 是写入文件，后面写入文件的内容会覆盖之前写入的内容。
            # "a+" 表示追加文件，后面写入的内容不会覆盖之前的内容，这个根据自己的具体任务，要会灵活使用。
            write_file("test_w_file_{}".format(j), mode="writelines", write_way="{}".format(j))

    if method == 'with':
        with_operator(file_name="test_file.txt")

"""
对于二进制文件，我感觉用的比较少，对于Keras或者TensorFlow等
框架的模型文件，都有相应的load方法，所以常用的也就是'r', 'w', 'a+'.
熟练掌握这三种方法，对于日常操作文件也就够了。另外对于大文件操作，感觉
Pandas还是挺有用的，后面再介绍。

"""
