import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib import error
from tqdm import trange
import urllib.parse
import pandas as pd

from pyduyp.logger.log import log


def stringpro(inputs):
    inputs = str(inputs)
    return inputs.strip().replace(" ", "").replace("\n", "").replace("\t", "").lstrip().rstrip()


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
        log.warning("{}".format(e))


def get_rwurls():
    url_tm = []
    for page in trange(1, 18):
        urls = "http://baike.baidu.com/fenlei/%E6%94%BF%E6%B2%BB%E4%BA%BA%E7%89%A9?limit=30&index={}&offset=0#gotoList".format(
            page)
        # urls = "http://baike.baidu.com/fenlei/%E7%A7%91%E5%AD%A6%E5%AE%B6?limit=30&index={}&offset=450#gotoList".format(page)
        html = urlhelper(urls)
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all('a', href=True):
            href = a['href']
            if "view" in href:
                nexthref = "http://baike.baidu.com" + href
                if nexthref not in url_tm:
                    url_tm.append(nexthref)

    df = pd.DataFrame(url_tm)
    df.to_excel("urls1.xlsx", index=None)
    print(df.shape)
    exit(1)


def get_data(filename="政治人物"):
    save = []
    urls = pd.read_excel("urls1.xlsx").values.tolist()
    for nexthref in urls:
        print(nexthref)
        html = urlhelper(nexthref[0])
        soup = BeautifulSoup(html, "lxml")
        try:
            resp = soup.findAll('div', attrs={"class": 'para'})
            name = soup.find('h1').text
            intr = []
            for i in range(len(resp)):
                try:
                    content = stringpro(resp[i].text)
                    intr.append(content)
                except Exception as e:
                    log.warning("555: {}".format(e))
            rows = {}
            rows['name'] = name
            rows['introduce'] = " ".join(intr)
            print(rows)
            save.append(rows)
        except Exception as e:
            log.warning("58888:{}".format(e))

    df = pd.DataFrame(save)
    df = df.drop_duplicates()
    df.to_excel("{}.xlsx".format(filename), index=None)
    print(df.shape)
