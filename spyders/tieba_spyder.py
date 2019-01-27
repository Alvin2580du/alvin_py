import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import requests
from collections import OrderedDict
from tqdm import tqdm, trange
import urllib.request
from urllib import error
import logging
import re
import random
import time

fw = open("已爬链接.txt", 'a+', encoding='utf-8')
fr = open("已爬链接.txt", 'r', encoding='utf-8')
lines = fr.readlines()
have = []
for line in lines:
    have.append(line.replace("\n", ""))


def isurl(url):
    if requests.get(url).status_code == 200:
        return True
    else:
        return False


ug = ['Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36',
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.80 Safari/537.36",
      'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0',
      ]


def urlhelper(url):
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", random.choice(ug))
        req.add_header("Accept", "*/*")
        req.add_header("Accept-Language", "zh-CN,zh;q=0.9")
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')
        return html

    except error.URLError as e:
        logging.warning("{}".format(e))


def build_links():
    links = []

    for page in range(0, 10000, 50):
        try:
            patt = 'rel="noreferrer" href="/p/\d+'
            # 美腿吧
            # urls = "https://tieba.baidu.com/f?kw=%E7%BE%8E%E8%85%BF&ie=utf-8&pn={}".format(page)
            # 美女吧
            # urls = "https://tieba.baidu.com/f?kw=%E7%BE%8E%E5%A5%B3&ie=utf-8&pn={}".format(page)
            # 女神吧
            urls = "https://tieba.baidu.com/f?kw=%E5%A5%B3%E7%A5%9E&ie=utf-8&pn={}".format(page)
            html = urlhelper(urls)
            href = re.compile(patt).findall(html)
            for x in href:
                tmp = x.replace('rel="noreferrer" href="/p/', "")
                link = "https://tieba.baidu.com/p/{}".format(tmp)
                print(link)
                links.append(link)
        except Exception as e:
            print(page, e)
            continue

    df = pd.DataFrame(links)
    df.to_excel("链接2.xlsx", index=None)
    print(df.shape)


def build_download():
    name = 'nvshen'
    epo = 3
    data = pd.read_excel("链接3.xlsx")
    num = 854
    for one in tqdm(data.values):
        try:
            if one[0] in have:
                continue
            htmls = urlhelper("{}?see_lz=1".format(one[0]))
            fw.writelines(one[0]+"\n")
            soup = BeautifulSoup(htmls, 'lxml')
            total = soup.findAll('img', attrs={"class": 'BDE_Image'})
            for t in total:
                num += 1
                urllib.request.urlretrieve(t['src'], filename='F:\\tiebaPictures\\meinv\\{}_{}_{}.jpg'.format(name, epo, num))
                if num % 50 == 0:
                    print("已下载{} 张美图".format(num))
                    time.sleep(random.random()*3)

        except Exception as e:
            print(one, e)
            continue


build_download()
fw.close()
