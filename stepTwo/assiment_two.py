import os
import re
import pandas as pd
from collections import OrderedDict, defaultdict, Counter

import csv


def get_doc(filename):
    # 获取文件内容
    res = []
    with open(filename, 'r') as fr:
        lines = fr.readlines()
        # 读取所有行
        for line in lines:
            # 添加到res
            res.append(line.replace("\n", ""))
    return " ".join(res)
    # 返回一个str类型


def replace(inputs):
    # 保留字母和数字，其他字符过滤掉
    res = []
    for x in inputs.split():
        if x.isalpha() or x.isdigit():
            res.append(x)
    return " ".join(res)


def get_freq(doc, word):
    # 计算文档的频率，利用Counter函数
    res = []
    for x, y in Counter(doc.split()).most_common(1000):
        if x == word:
            rows = {x: y}
            res.append(rows)
    return res


def makedatasets():
    save = []
    for file in os.listdir("./cranfieldDocs"):
        # 遍历文件cranfieldDocs下面所有文件
        try:
            file_name = os.path.join("cranfieldDocs", file)
            # 获取文件名
            doc = get_doc(file_name)
            docno = re.search("\d+", doc[:30]).group()
            # 用正则表达式匹配文本内容
            title = re.findall('.*<TITLE>(.*)</TITLE>', doc)[0]
            author = re.findall('.*<AUTHOR>(.*)</AUTHOR>', doc)[0]
            biblto = re.findall('.*<BIBLIO>(.*)</BIBLIO>', doc)[0]
            text = re.findall('.*<TEXT>(.*)</TEXT>', doc)[0]
            rows = OrderedDict()
            # 新建一个字典
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
    # 保存到文件


def build_inverted_index(datasets_file_name):
    # 输入dataset文件名
    file = datasets_file_name
    # 打开csv文件
    with open(file, newline='') as csvfile:
        result = {}
        spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')  # 读取csv
        # 读取每一行
        for row in spamreader:
            # text行编号
            num = row['docno']
            # content按' '分隔
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
    # 遍历关键词
    for word in words:
        if word not in inverted_index:
            continue
        # 对所有含有该关键词的文档进行遍历，统计次数
        for item in inverted_index[word]:
            num, freq = item
            # 对所有word在各文档中的次数求和
            result[num] = result.get(num, 0) + freq
    # 对result里的次数进行排序
    sorted_key_list = sorted(result, key=lambda x: result[x], reverse=True)
    # 取排序前十的
    for key in sorted_key_list[:10]:
        print('cranfield' + key.zfill(4), result[key])
    print('-' * 30)


datasets_file_name = 'datasets.csv'
ans = build_inverted_index(datasets_file_name)

fw = open('out.txt', 'w', encoding='utf-8')
for x,y in ans.items():
    fw.writelines("{} {}".format(x, y)+'\n')

#
# while 1:
#     # 输入关键词
#     words = input('Please input keyword(separated by space):').split(' ')
#     print('the top 10 documents most revelant to %s is ' % words)
#     try:
#         search_(words, ans)
#     except:
#         print('no documnet include word %s' % words)
#         pass
