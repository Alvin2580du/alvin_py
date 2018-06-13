import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib import error
from tqdm import trange, tqdm
import urllib.parse
import pandas as pd
from collections import OrderedDict
import logging


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


def build():
    for page in range(100):
        save = []
        urls = 'https://www.coindesk.com/page/{}/'.format(page)
        if isurl(urls):
            html = urlhelper(urls)
            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('a', attrs={'class': 'fade'})
            for i in trange(len(resp)):
                rows = OrderedDict()
                resp1 = resp[i]
                try:
                    timeauthor = resp1.find("p").get_text()
                    time = timeauthor.split("by")[0]
                    author = timeauthor.split("by")[1]
                    rows['author'] = author
                    rows['time'] = time
                except Exception as e:
                    continue

                try:
                    title = resp1.find('h3').get_text()
                    rows['title'] = title
                except Exception as e:
                    continue

                try:
                    link_html = urlhelper(resp1['href'])
                    link_soup = BeautifulSoup(link_html, "lxml")
                    resp = link_soup.findAll('p')
                    content = get_content(resp)
                    rows['content'] = content
                except Exception as e:
                    continue
                save.append(rows)
                print(rows)

        df = pd.DataFrame(save)
        df.to_csv("./datasets/www.coindesk.com_{}.csv".format(page))


build()

