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


def isurl(url):
    if requests.get(url).status_code == 200:
        return True
    else:
        return False


def urlhelper(url):
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.80 Safari/537.36")
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
    data = pd.read_excel("链接2.xlsx")
    num = 0
    for one in tqdm(data.values):
        try:
            htmls = urlhelper("{}?see_lz=1".format(one[0]))
            soup = BeautifulSoup(htmls, 'lxml')
            total = soup.findAll('img', attrs={"class": 'BDE_Image'})
            for t in total:
                num += 1
                urllib.request.urlretrieve(t['src'], filename='F:\\tiebaPictures\\meinv\\{}.jpg'.format(num))
                if num % 100 == 0:
                    print("已下载{} 张美图".format(num))
        except Exception as e:
            print(one, e)
            continue

build_links()
exit(1)
build_download()

