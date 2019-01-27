import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib import error
import urllib.parse
import pandas as pd
from collections import OrderedDict
import logging
import time
import re
import urllib
from selenium import webdriver


def isurl(url):
    if requests.get(url).status_code == 200:
        return True
    else:
        return False


def urlhelper(url):
    user_agents = [
        'Opera/9.25 (Windows NT 5.1; U; en)',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
        'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
        'Mozilla/5.0 (X11; U; linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
        'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9'
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36']
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       user_agents[6])
        req.add_header("Accept", "*/*")
        req.add_header('Connection', "close")
        req.add_header("Accept-Language", "zh-CN,zh;q=0.9")
        req.add_header("Cookie", "rn8tca8a5oqe72tswy7n3g08puq06vmv")
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')
        return html
    except error.URLError as e:
        logging.warning("{}".format(e))


def get_urls():
    # 这个函数主要是获取待爬取的所有工作的详情页链接
    save = []
    browser = webdriver.Chrome()
    for page in range(7):
        try:
            # urls = "https://sou.zhaopin.com/?p={}&jl=489&kw=%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86&kt=3".format(page)
            urls = 'https://sou.zhaopin.com/?p={}&jl=489&kw=%E6%A1%A3%E6%A1%88%E5%AD%A6&kt=3'.format(page)
            if isurl(urls):
                try:
                    browser.get(urls)
                    time.sleep(2)
                    links = browser.find_elements_by_xpath("//*[@href]")
                    for link in links:
                        link_next = link.get_attribute('href')
                        if "https://jobs.zhaopin.com" in link_next:
                            save.append(link_next)
                            print(link_next)
                except Exception as e:
                    continue
        except:
            print("page:{}".format(page))

    df = pd.DataFrame(save)
    df.to_excel("网址.xlsx", index=None)
    print("恭喜！网址保存成功！")


def build():
    # 用上面获得的所有链接，爬取职位详细信息
    data = pd.read_excel('网址.xlsx').values.tolist()
    urls = list(set([j for i in data for j in i]))
    print(len(urls))
    save = []
    fail_url = []
    for url in urls:
        try:
            html = urlhelper(url)
            soup = BeautifulSoup(html, "lxml")
            title = soup.findAll('h1')[0].text
            info = soup.find("div", attrs={"class": "pos-ul"}).findAll("p")
            info_save = []
            if len(info) < 1:
                info = soup.find("div", attrs={"class": "pos-ul"}).findAll("div")
            for jd in info:
                info_save.append(jd.text)
            company = soup.findAll("div", attrs={"class": "company l"})[0].text
            rows = OrderedDict()
            rows['名称'] = title
            rows['公司'] = re.search("[\u4e00-\u9fa5]+", str(company))[0]
            rows['职责'] = " ".join(info_save)
            rows['网址'] = url
            save.append(rows)
            print(rows)
        except:
            print("爬取失败:{}".format(url))
            fail_url.append(url)
            continue

    print(fail_url)
    df = pd.DataFrame(save)
    df.to_excel("档案学.xlsx", index=None)


get_urls()
build()
