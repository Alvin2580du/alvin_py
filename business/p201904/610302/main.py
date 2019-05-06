import re
import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
import jieba

sw = []
with open('stopwords.txt', 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        sw.append(line.replace("\n", ""))

# =============================房天下===============================


def get_fangtianxia_urls(file_name, url):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4295.400 QQBrowser/9.7.12661.400'
    }
    text = requests.get(url, headers=headers).text
    soup = BeautifulSoup(text, 'html.parser')
    divlist = soup.find(id='newhouse_loupai_list').find_all(name='li')
    for m in divlist:
        try:
            href = m.find(class_='nlcd_name').find(name='a').attrs['href']
        except:
            continue
        with open("{}.txt".format(file_name), 'a+', encoding='utf-8') as f:
            f.write(href + '\n')


def build_fangtianxia(file_name):
    save = []
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4295.400 QQBrowser/9.7.12661.400'
    }
    with open("{}.txt".format(file_name), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            url = 'https:' + line + 'dianping'
            html = requests.get(url, headers=headers)
            html.encoding = "gbk"
            soup = BeautifulSoup(html.text, 'html.parser')
            divlist = soup.find_all(class_='comm_list')
            for divtag in divlist:
                comm_dict = {}
                try:
                    comm_dict['comment'] = divtag.find(class_='comm_list_con').text.replace(' ', '').replace('\n', '')
                    save.append(comm_dict)
                    print(comm_dict['comment'])
                except:
                    continue

            try:
                # https://tianhaojlc.fang.com/house/ajaxrequest/dianpingList_201501.php
                url2 = 'https://taibeigongguan.fang.com/house/ajaxrequest/dianpingList_201501.php'
                code = re.findall(r"dianpingNewcode = \"(\d+)", html.text)[0]
                for n in range(2, 20):
                    data = {
                        'city': '苏州',
                        'dianpingNewcode': code,
                        'ifjiajing': '0',
                        'page': n,
                        'pagesize': '20'
                    }
                    html2 = requests.post(url2, data=data, headers=headers).text
                    try:
                        html2 = json.loads(html2)
                        comentlist = html2['list']
                        for comment in comentlist:
                            comm_dict = {'comment': comment['content']}
                            save.append(comm_dict)
                            print(comment['content'])
                    except:
                        break
            except:
                pass

    df = pd.DataFrame(save)
    df.to_csv("{}_Results.csv".format(file_name.replace(".txt", "")), index=None)
    print(df.shape)


def classifyWords(wordList):
    # (1) 情感词
    with open("BosonNLP_sentiment_score.txt", 'r', encoding='utf8') as f:
        senList = f.readlines()
    senDict = defaultdict()
    for s in senList:
        try:
            senDict[s.split(' ')[0]] = s.split(' ')[1].strip('\n')
        except Exception:
            pass
    # (2) 否定词
    with open("notDict.txt", 'r', encoding='utf-8') as f:
        notList = f.read().splitlines()
    # (3) 程度副词
    with open("degreeDict.txt", 'r', encoding='utf-8') as f:
        degreeList = f.read().splitlines()
    degreeDict = defaultdict()
    for index, d in enumerate(degreeList):
        if 3 <= index <= 71:
            degreeDict[d] = 2
        elif 74 <= index <= 115:
            degreeDict[d] = 1.25
        elif 118 <= index <= 154:
            degreeDict[d] = 1.2
        elif 157 <= index <= 185:
            degreeDict[d] = 0.8
        elif 188 <= index <= 199:
            degreeDict[d] = 0.5
        elif 202 <= index <= 231:
            degreeDict[d] = 1.5
        else:
            pass

    senWord = defaultdict()
    notWord = defaultdict()
    degreeWord = defaultdict()

    for index, word in enumerate(wordList):
        if word in senDict.keys() and word not in notList and word not in degreeDict.keys():
            senWord[index] = senDict[word]
        elif word in notList and word not in degreeDict.keys():
            notWord[index] = -1
        elif word in degreeDict.keys():
            degreeWord[index] = degreeDict[word]
    return senWord, notWord, degreeWord


def scoreSent(senWord, notWord, degreeWord, segResult):
    W = 1
    score = 0
    # 存所有情感词的位置的列表
    senLoc = list(senWord.keys())
    notLoc = list(notWord.keys())
    degreeLoc = list(degreeWord.keys())
    senloc = -1

    # 遍历句中所有单词segResult，i为单词绝对位置
    for i in range(0, len(segResult)):
        # 如果该词为情感词
        if i in senLoc:

            # loc为情感词位置列表的序号
            senloc += 1
            # 直接添加该情感词分数
            score += W * float(senWord[i])
            if senloc < len(senLoc) - 1:
                # 判断该情感词与下一情感词之间是否有否定词或程度副词
                # j为绝对位置
                for j in range(senLoc[senloc], senLoc[senloc + 1]):
                    # 如果有否定词
                    if j in notLoc:
                        W *= -1
                    # 如果有程度副词
                    elif j in degreeLoc:
                        W *= float(degreeWord[j])
        # i定位至下一个情感词
        if senloc < len(senLoc) - 1:
            i = senLoc[senloc + 1]
    return score


def get_wordDicts(filename):
    words = {}
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        num = 0
        for line in lines:
            newSent = [i for i in jieba.lcut(line.strip().replace("\n", "")) if i not in sw]
            for x in newSent:
                if x not in words.keys():
                    words[x] = num
                    num += 1
    return words


def build_sentments(filename):
    # 情感分析
    wordsDicts = get_wordDicts(filename)
    senWord, notWord, degreeWord = classifyWords(wordsDicts)
    save = []
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        limit = 20000
        num = 0
        for line in lines:
            num += 1
            if num > limit:
                break
            newSent = [i for i in jieba.lcut(line.strip().replace("\n", "")) if i not in sw]
            scores = scoreSent(senWord, notWord, degreeWord, newSent)

            if scores > 0:
                rows = {"sents": line.strip().replace("\n", ""), 'scores': scores, 'labels': 'pos'}
            elif scores == 0:
                rows = {"sents": line.strip().replace("\n", ""), 'scores': scores, 'labels': 'neu'}
            else:
                rows = {"sents": line.strip().replace("\n", ""), 'scores': scores, 'labels': 'neg'}
            save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("情感分析结果.xlsx", index=None)
    print(df.shape)


if __name__ == '__main__':

    method = 'build_sentments'

    if method == 'build_sentments':
        # 情感分析
        build_sentments(filename='data.csv')
