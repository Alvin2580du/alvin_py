import os
import pandas as pd


def make_data():
    save_train = []
    num = 0
    with open('./datasets/test11.txt', 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            if line:
                try:
                    res = line.split("\t")
                    scores = int(res[0].replace("\n", "").replace('"', "").strip())
                    if scores > 3:
                        sentiment = 1
                    else:
                        sentiment = 0
                    rows = {
                        'sentiment': sentiment,
                        'content': res[1].strip()
                    }
                    save_train.append(rows)
                    num += 1
                    if num % 10000 == 0:
                        print(num)
                except:
                    continue
            else:
                break
    print(num)
    df = pd.DataFrame(save_train)
    df.to_excel("./datasets/test11_2_class.xlsx", index=None)
    print(df.shape)


def get_train_data():
    data = pd.read_excel("./datasets/test11_2_class.xlsx")
    if not os.path.exists("./data2"):
        os.makedirs("./data2")
    for x, y in data.groupby(by='sentiment'):
        tmp = y['content']
        tmp.to_excel("./data2/{}.xlsx".format(x), index=None)
        print(tmp.shape)


if __name__ == "__main__":

    method = 'get_train_data'

    if method == 'make_data':
        make_data()

    if method == 'get_train_data':
        get_train_data()

    if method == 'test':
        fw = open('train.txt', 'a+', encoding='utf-8')
        with open('./data2/1.csv', 'r', encoding='utf-8') as fr1:
            lines =fr1.readlines()
            for line in lines:
                res = '{}\t{}'.format(line.replace("\n", ""), '1')
                fw.writelines(res+"\n")

        with open('./data2/0.csv', 'r', encoding='utf-8') as fr2:
            lines = fr2.readlines()
            for line in lines:
                res = '{}\t{}'.format(line.replace("\n", ""), '0')
                fw.writelines(res + "\n")

    if method == 'test23':

        fw = open('trainSmall.txt', 'a+', encoding='utf-8')
        with open('./datasets/pos.txt', 'r', encoding='utf-8') as fr1:
            lines = fr1.readlines()
            for line in lines:
                res = '{}\t{}'.format(line.replace("\n", ""), '1')
                fw.writelines(res + "\n")

        with open('./datasets/neg.txt', 'r', encoding='utf-8') as fr2:
            lines = fr2.readlines()
            for line in lines:
                res = '{}\t{}'.format(line.replace("\n", ""), '0')
                fw.writelines(res + "\n")

