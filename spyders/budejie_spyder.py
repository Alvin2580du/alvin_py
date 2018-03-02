
import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib import error
from tqdm import trange, tqdm
import urllib.parse
import pandas as pd

from pyduyp.logger.log import log

"""
百思不得姐： http://www.budejie.com/text/2

"""
def urlhelper(url):
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36")
        req.add_header("Accept", "*/*")
        req.add_header("Accept-Language", "zh-CN,zh;q=0.8")
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')
        return html
    except error.URLError as e:
        log.warning("{}".format(e))


def budejie():
    fw = open("./datasets/xiaohua.txt", 'a+', encoding="utf-8")

    for page in trange(51):
        url = "http://www.budejie.com/text/{}".format(page)
        html = urlhelper(url)
        soup = BeautifulSoup(html, "lxml")
        total = soup.findAll('div', attrs={"class": 'j-r-list-c'})
        for i in range(len(total)):
            content = total[i].text.replace("\n", "")
            if len(content) < 10:
                continue
            fw.writelines(content+"\n")

budejie()




