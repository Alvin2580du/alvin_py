import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib import error
from tqdm import trange, tqdm
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


def geturls():
    all_urls = []
    files = ['shehui', 'tiyu', 'renwu', 'ziran', 'wenhua', 'lishi', 'dili', 'keji']
    for file in files:
        rooturl = 'http://baike.baidu.com/{}'.format(file)
        log.info("{}".format(rooturl))
        html = urlhelper(rooturl)
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all('a', href=True):
            href = a['href']
            if "view" in href:
                log.info("href: {}".format(href))
                all_urls.append(href)
                hrefhtml = urlhelper(href)
                hrefsoup = BeautifulSoup(hrefhtml, "lxml")
                for a in hrefsoup.find_all('a', href=True):
                    href = a['href']
                    if "view" in href and "http" not in href:
                        nexthref = "http://baike.baidu.com" + href
                        log.info("nexthref: {}".format(nexthref))
                        all_urls.append(nexthref)

                    elif "view" in href and "http" in href:
                        nexthref = href
                        log.info("nexthref: {}".format(nexthref))
                        all_urls.append(nexthref)
                    else:
                        continue

    df = pd.DataFrame(all_urls)
    df.to_csv("./localdatasets/baike/urls.txt", index=None)


def geturlofurl(url):
    all_urls = []
    hrefhtml = urlhelper(url)
    hrefsoup = BeautifulSoup(hrefhtml, "lxml")
    for a in hrefsoup.find_all('a', href=True):
        href = a['href']
        if "view" in href and "http" not in href and href.split(".")[-1] != 'htm':
            nexthref = "http://baike.baidu.com" + href
            all_urls.append(nexthref)

        elif "view" in href and "http" in href and href.split(".")[-1] != 'htm':
            nexthref = href
            all_urls.append(nexthref)
        else:
            continue
    return all_urls


def baikespyder():
    urls = pd.read_csv("./localdatasets/baike/urls.txt").values
    number = 0
    res = []
    for url in tqdm(urls):
        if url[0].split(".")[-1] == 'htm':
            continue
        html = urlhelper(url[0])
        soup = BeautifulSoup(html, "lxml")
        try:
            resp = soup.findAll('div', attrs={"class": 'para'})
            for i in range(len(resp)):
                try:
                    content = stringpro(resp[i].text)
                    res.append(content)
                except Exception as e:
                    log.warning("555: {}".format(e))
        except Exception as e:
            log.warning("58888:{}".format(e))

        number += 1
        if number % 500 == 0:
            df = pd.DataFrame(res)
            save_name = "./localdatasets/baike/{}.txt".format(number)
            print(save_name)
            df.to_csv(save_name, index=None)
            res = []


def get_moreurls():
    urls = pd.read_csv("./localdatasets/baike/urls_new.txt").values
    res = []
    for url in tqdm(urls):
        if url[0] not in res:
            res.append(url[0])
        newurls = geturlofurl(url[0])
        for x in newurls:
            if x not in res:
                res.append(x)

    print(len(res))
    df = pd.DataFrame(res)
    df.to_csv("./localdatasets/baike/urls_new1.txt", index=None, header=None)


get_moreurls()

