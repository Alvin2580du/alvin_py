import numpy
import random
import gzip
import pandas as pd
from collections import OrderedDict


def parseData(f):
    for l in gzip.open(f):
        yield eval(l)


def get_datasets():
    data = list(parseData("./datasets/train.json.gz"))
    save = []
    num = 0
    for info in data:
        rows = OrderedDict()
        rows['userID'] = info['userID']
        rows['businessID'] = info['businessID']
        rows['rating'] = info['rating']
        rows['reviewTime'] = info['reviewTime']
        rows['reviewHash'] = info['reviewHash']
        rows['unixReviewTime'] = info['unixReviewTime']
        rows['categories'] = "|".join(info['categories'])
        rows['reviewText'] = info['reviewText']

        save.append(rows)
        num += 1
        if num == 100000:
            df = pd.DataFrame(save)
            numpy.random.shuffle(df)
            df.to_csv("./datasets/train.data.csv", index=None, encoding="utf-8")
            print(df.shape)
            save = []

    df = pd.DataFrame(save)
    numpy.random.shuffle(df)

    df.to_csv("./datasets/valid.data.csv", index=None, encoding="utf-8")
    print(df.shape)

get_datasets()
exit(1)


def get_valid_data():
    data = list(parseData("train.json.gz"))

    train_set = data[0:100000]
    valid_set = data[100000:200000]

    usersID = []
    businessesID = []
    visit = {}
    nonvisit = {}

    for info in data:
        usersID.append(info['userID'])
        businessesID.append(info['businessID'])
        if info['userID'] in visit.keys():
            visit[info['userID']].append(info['businessID'])
        else:
            visit[info['userID']] = [info['businessID']]

    numpy.random.shuffle(usersID)
    numpy.random.shuffle(businessesID)

    count = 0
    while count < 100000:
        user = random.choice(usersID)
        business = random.choice(businessesID)
        if business in visit[user]:
            pass
        else:
            if user in nonvisit.keys():
                if business in nonvisit[user]:
                    pass
                else:
                    nonvisit[user].append(business)
                    count += 1
            else:
                nonvisit[user] = [business]
                count += 1

    with open('pairs_Visit_valid.txt', 'w+') as f:
        for pos_datum in valid_set:
            f.writelines(pos_datum['userID'] + '-' + pos_datum['businessID'] + ',' + '1\n')

        for neg_datum in nonvisit.keys():
            if len(nonvisit[neg_datum]) > 1:
                for business in nonvisit[neg_datum]:
                    f.writelines(neg_datum + '-' + business + ',' + '0\n')
            else:
                f.writelines(neg_datum + '-' + nonvisit[neg_datum][0] + ',' + '0\n')
    f.close()


def shuffle_vaild_data():
    fread = open("pairs_Visit_valid.txt", "r")
    lines = fread.readlines()
    fread.close()
    random.shuffle(lines)
    fwrite = open("pairs_Visit_valid.txt", "w")
    fwrite.writelines('userID-businessID,prediction\n')
    fwrite.writelines(lines)
    fwrite.close()


def get_prediction():
    businessCount = {}
    totalPurchases = 0

    for l in data:
        user, business = l['userID'], l['businessID']
        if business not in businessCount.keys():
            businessCount[business] = 1
        else:
            businessCount[business] += 1
        totalPurchases += 1

    mostPopular = [(businessCount[x], x) for x in businessCount]
    mostPopular.sort()
    mostPopular.reverse()

    for i in range(100):
        threshold = i * 0.01
        return1 = set()
        count = 0
        for ic, i in mostPopular:
            count += ic
            return1.add(i)
            if count > int(totalPurchases * threshold):
                break

        right_count = 0
        wrong_count = 0
        for l in open("pairs_Visit_valid.txt"):
            if l.startswith("userID"):
                pass
            else:
                info = l.strip().split(',')
                pairs = info[0].split('-')
                if pairs[1] in return1:
                    if info[1] == '1':
                        right_count += 1
                    else:
                        wrong_count += 1
                else:
                    if info[1] == '0':
                        right_count += 1
                    else:
                        wrong_count += 1
        print(str(threshold) + '\t\t' + str(float(right_count) / (right_count + wrong_count)))
        predictions = open("predictions_Visit.txt", 'w')
        for l in open("pairs_Visit_valid.txt"):
            if l.startswith("userID"):
                predictions.write(l)
                continue
            u, i = l.strip().split('-')
            if i in return1:
                predictions.write(u + '-' + i + ",1\n")
            else:
                predictions.write(u + '-' + i + ",0\n")
        predictions.close()


get_prediction()
