import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib import error
import urllib.parse
import pandas as pd
from collections import OrderedDict
import logging
import time


def stringpro(inputs):
    inputs = str(inputs)
    return inputs.strip().replace("\n", "").replace("\t", "").lstrip().rstrip()


def isurl(url):
    if requests.get(url).status_code == 200:
        return True
    else:
        return False


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
        logging.warning("{}".format(e))


def get_content(resp):
    save = []
    for i in range(len(resp)):
        content = resp[i].text
        save.append(stringpro(content))
    return " ".join(save)


def get_label(html):
    save = []
    if len(html) == 1:
        soup = BeautifulSoup(html[0], "lxml").find_all("a")
        for x in soup:
            save.append(x.text)
        return ",".join(save)


def build():
    save = []
    for page in range(100):
        time.sleep(3)
        urls = 'https://www.coindesk.com/page/{}/'.format(page)
        if isurl(urls):
            html = urlhelper(urls)
            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('div', attrs={"class": "post-info"})
            for one in range(len(resp)):
                try:
                    rows = OrderedDict()
                    post_info = resp[one]
                    timeauthor = post_info.p.text
                    rows['author'] = str(timeauthor).split("\n")[1]
                    rows['time'] = str(timeauthor).split("\n")[0].replace("|", "")
                    title = post_info.find('h3').get_text()
                    rows['title'] = title
                    position_link = post_info.findAll('a', attrs={'class': 'fade'})
                    link = position_link[0]['href']
                    link_html = urlhelper(link)
                    link_soup = BeautifulSoup(link_html, "lxml")
                    resp2 = link_soup.findAll('p')
                    content = get_content(resp2)
                    label_html = link_soup.find("p", attrs={"class": "single-tags"}).find_all("a")
                    labels = []
                    for th in label_html:
                        label = th.get_text()
                        labels.append(label)
                    rows['labels'] = ",".join(labels)
                    rows['content'] = content
                    save.append(rows)
                    print(rows)
                except Exception as e:
                    logging.warning("------------e:{},{}".format(e, urls))
                    continue
        if page % 3 == 0:
            df = pd.DataFrame(save)
            df.to_csv("./datasets/www.coindesk.com_{}.csv".format(page), index=None)


build()
