import os
import re
import pandas as pd
from collections import OrderedDict, defaultdict, Counter

import csv


def get_doc(filename):
    res = []
    with open(filename, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res.append(line.replace("\n", ""))
    return " ".join(res)


def replace(inputs):
    res = []
    for x in inputs.split():
        if x.isalpha() or x.isdigit():
            res.append(x)
    return " ".join(res)


def get_freq(doc, word):
    res = []
    for x, y in Counter(doc.split()).most_common(1000):
        if x == word:
            rows = {x: y}
            res.append(rows)
    return res


def makedatasets():
    save = []
    for file in os.listdir("./cranfieldDocs"):
        try:
            file_name = os.path.join("cranfieldDocs", file)
            doc = get_doc(file_name)
            docno = re.search("\d+", doc[:30]).group()
            title = re.findall('.*<TITLE>(.*)</TITLE>', doc)[0]
            author = re.findall('.*<AUTHOR>(.*)</AUTHOR>', doc)[0]
            biblto = re.findall('.*<BIBLIO>(.*)</BIBLIO>', doc)[0]
            text = re.findall('.*<TEXT>(.*)</TEXT>', doc)[0]
            rows = OrderedDict()
            rows['docno'] = docno
            rows['title'] = title
            rows['author'] = author
            rows['biblto'] = biblto
            rows['text'] = text
            res = "{} {} {} {}".format(replace(title), replace(author), replace(biblto), replace(text))
            rows['content'] = res
            save.append(rows)

        except Exception as e:
            print("Exception:{}, {}".format(file, e))
            continue
    df = pd.DataFrame(save)
    df.to_csv("datasets.csv", index=None)
    print("{},{}".format(df.shape[0], df.shape[1]))


def build_inverted_index(datasets_file_name):
    file = datasets_file_name
    with open(file, newline='') as csvfile:
        result = {}
        spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')  # 读取csv
        for row in spamreader:
            num = row['docno']
            words = row['content'].split(' ')
            word_dict = {}
            for word in words:
                # 计算content里面单词的次数
                word_dict[word] = word_dict.get(word, 0) + 1
            for word in word_dict:
                # 将单个content的计算结果添加到inverted_index中
                if word in result:
                    result[word].append((num, word_dict[word]))
                else:
                    result[word] = [(num, word_dict[word])]
    return result


def search_(words, inverted_index):
    result = {}
    for word in words:
        if word not in inverted_index:
            continue
        for item in inverted_index[word]:
            num, freq = item
            result[num] = result.get(num, 0) + freq
    sorted_key_list = sorted(result, key=lambda x: result[x], reverse=True)
    for key in sorted_key_list[:10]:
        print('cranfield' + key.zfill(4), result[key])
    print('-' * 30)


datasets_file_name = 'datasets.csv'
ans = build_inverted_index(datasets_file_name)
while 1:
    words = input('Please input keyword(separated by space):').split(' ')
    print('the top 10 documents most revelant to %s is ' % words)

    try:
        search_(words, ans)
    except:
        print('no documnet include word %s' % words)
        pass
