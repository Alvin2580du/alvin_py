import pandas as pd
from collections import OrderedDict
import jieba
import re
from bs4 import BeautifulSoup

import sys
import re
# import HTMLParser
import getopt

jieba.load_userdict("./dictionary/jieba_dict.txt")


# html_parser = HTMLParser.HTMLParser()
space_pat = re.compile(r'\\t|\\n', re.S)
p_pat = re.compile(r'(<p(>| ))|<br>|<br/>', re.S)
sc_tag_pat = re.compile(r'<[^>]+>', re.S)
multi_space_pat = re.compile(r' +', re.S)


def str_q2b(s):

    res = ""
    for u in s:

        c = ord(u)
        if c == 12288:
            c = 32
        elif 65281 <= c <= 65374:
            c -= 65248

        res += chr(c)

    return res


def html_filter(content):
    s1 = space_pat.sub(' ', content).replace(r'\r', '')
    s2 = p_pat.sub(lambda x: ' ' + x.group(0), s1)
    s3 = sc_tag_pat.sub('', s2).strip()
    # s4 = html_parser.unescape(s3.decode('utf8')).encode('utf8')
    # s5 = str_q2b(s3.decode('utf8')).encode('utf8').replace('\xc2\xa0', ' ')
    content_txt = multi_space_pat.sub(' ', s3).strip()
    return content_txt


biaodian = ['，', '。', '?', '？', ',', '.', '、', ':']

bracket = ['>', '<', '﹞', '﹝', '＞', '＜', '》', '《', '】', '【', '）', '（', '(', ')',
           '[', ']', '«', '»', '‹', '›', '〔', '〕', '〈', '〉', '』', '『', '〗', '〖',
           '｝', '｛', '」', '「', '］', '［', '}', '{']

spacialsymbol = ['“', '”', '‘', '’', '〝', '〞', ' ', '"', "'", '＂', '＇', '´', '＇', '>', '<', '^', '¡', '¿',
                 'ˋ', '`', '︶', '︸', '︺', '﹀', '︾', '﹂', '﹄', '﹃', '﹁', '︽', '︿', '︹', '︷', '︵', '/',
                 '|', '\\', '＼', '$', '#', '￥', '&', '”', '“', '×', '@', '~', '’', '^', '*', '%', '～', '⊙', '％',
                 '℃', '＋', '╮', '≧', '≦', '｀', 'ヾ', 'з', 'ω', '∠', '→', 'ㄒ', 'ワ', 'π', '＊', '∩', 'и', 'п']

all_symbol = biaodian + bracket + spacialsymbol


def checknesfromhtml(html):
    soup = BeautifulSoup(html, "lxml")
    try:
        resp = soup.findAll('p')
        save = []
        for i in range(len(resp)):
            res = resp[i].text
            save.append(res)
        return "<pos>".join(save)
    except Exception as e:
        print("eeeeeeeeeeee: {}".format(e))
        pass


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def getnews(msg):
    out = []
    for x in msg:
        if is_chinese(x):
            out.append(x)
    return "".join(out)


label_cache = {}
with open("./datasets/News_pic_label_train.txt", 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        linesp = line.split("\t")
        newsid, label = linesp[0], linesp[1]
        label_cache[newsid] = label

print("labeled news numbers: {}".format(len(label_cache)))


def newinfo():
    save_data = []
    file_object1 = open("./datasets/News_info_train.txt", 'r', encoding='utf-8')
    try:
        counter = 0

        while True:
            line = file_object1.readline()
            if line:
                counter += 1
                linesp = line.split("\t")
                try:
                    newsid, imagesid, msg = linesp[0], linesp[2], linesp[1],
                    news = html_filter(msg)
                    rows = OrderedDict()
                    rows["mewsid"] = newsid
                    rows['imagesid'] = imagesid
                    rows['msg'] = news
                    rows['label'] = label_cache[newsid]
                    rows['length'] = len(news)
                    save_data.append(rows)

                except Exception as e:
                    continue

                if counter % 1000 == 0:
                    print("counter:{}:{}".format(counter, rows))
            else:
                break
    finally:
        file_object1.close()

    df = pd.DataFrame(save_data)
    df.to_csv("./datasets/News_info_train.csv", index=None, line_terminator='\n', encoding="utf-8")


def picinfo():
    save_data = []
    with open("./datasets/News_pic_label_train_example100.txt", 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            linesp = line.split("\t")
            newsid, label, imagesid, msg = linesp[0], linesp[1], linesp[2], linesp[3:]

            rows = OrderedDict(
                {"mewsid": newsid, "label": label, "imagesid": imagesid, 'msg': msg[0].replace("\n", "")})
            save_data.append(rows)
    df = pd.DataFrame(save_data)
    df.to_csv("./datasets/News_pic_label_train_example100.csv", index=None, line_terminator='\n', encoding="utf-8")


if __name__ == "__main__":
    import sys

    # method = sys.argv[1]
    method = 'newinfo'
    if method == 'newinfo':
        newinfo()
