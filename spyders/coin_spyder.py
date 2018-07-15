import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib import error
import urllib.parse
import pandas as pd
from collections import OrderedDict
import logging
import time
from tqdm import trange
import re
from lxml import etree
import json
import urllib
import random

from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


def stringpro(inputs):
    inputs = str(inputs)
    return inputs.strip().replace("\n", "").replace("\t", "").lstrip().rstrip()


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
        req.add_header("Accept-Language", "zh-CN,zh;q=0.9")
        req.add_header("Cookie", "rn8tca8a5oqe72tswy7n3g08puq06vmv")
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


def coindesk():
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
                    timeauthor = post_info.findAll('p', attrs={'class': 'timeauthor'})
                    rows['author'] = timeauthor[0].find('cite').text
                    rows['time'] = timeauthor[0].find('time')['datetime'].replace("T", " ").split("+")[0]
                    # timeauthor = post_info.p.text
                    # rows['author'] = str(timeauthor).split("\n")[1]
                    # rows['time'] = str(timeauthor).split("\n")[0].replace("|", "")
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


def get_author_time(inputs):
    s = str(inputs).replace(",", "").split(" ")
    if len(s) == 6:
        author = s[:1]
        time = s[2:]
    elif len(s) == 7:
        author = s[:2]
        time = s[3:]
    else:
        return '', ''
    return ' '.join(author), " ".join(time)


def get_time(inputs):
    return inputs.replace("T", " ").replace('Z', "").split(".")[0]


def ethnews():
    save = []
    for page in trange(150):
        time.sleep(1)
        urls = 'https://www.ethnews.com/news?page={}'.format(page)
        if isurl(urls):
            html = urlhelper(urls)
            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('div', attrs={"class": "article-thumbnail"})
            for one in range(len(resp)):
                try:
                    rows = OrderedDict()
                    post_info = resp[one]
                    author = post_info.findAll('div', attrs={'class': 'article-thumbnail__info__etc__author'})[0].text
                    times = post_info.findAll('div', attrs={'class': 'article-thumbnail__info__etc__date'})
                    rows['author'] = author
                    rows['time'] = get_time(times[0].find('h6')['data-created-short'])
                    position_link = post_info.find_all('h2', attrs={'class': 'article-thumbnail__info__title'})
                    link_pre = position_link[0].find('a', href=True)['href']
                    link = 'https://www.ethnews.com{}'.format(link_pre)
                    link_html = urlhelper(link)
                    link_soup = BeautifulSoup(link_html, "lxml")
                    title = link_soup.findAll('div', attrs={'class': 'article-cover__inner'})[0].text
                    rows['labels'] = title
                    category = link_soup.findAll('div', attrs={'class': 'article__category'})[0].text
                    rows['labels'] = category
                    summary = link_soup.findAll('p', attrs={'class': 'article__summary'})[0].text
                    rows['summary'] = summary
                    content = link_soup.findAll('div', attrs={'class': 'article__content'})
                    content = get_content(content)
                    rows['content'] = content
                    save.append(rows)
                    print(rows)
                except Exception as e:
                    logging.warning("------------e:{},{}".format(e, urls))
                    continue
        if page % 10 == 0:
            df = pd.DataFrame(save)
            df.to_csv("./datasets/www.ethnews.com_{}.csv".format(page), index=None)


def get_coinspeaker_labels(resp):
    save = []
    for i in range(len(resp)):
        content = resp[i].text
        save.append(content)
    return ",".join(save)


def coinspeaker():
    save = []
    cates = ['https://www.coinspeaker.com/category/story-of-the-day/page',
             'https://www.coinspeaker.com/category/news/cryptocurrencies/page',
             'https://www.coinspeaker.com/category/news/fintech/page',
             'https://www.coinspeaker.com/category/news/payments-and-commerce/page',
             'https://www.coinspeaker.com/category/news/internet-of-things/page']
    for c in cates:
        for page in trange(150):
            time.sleep(1)
            urls = "{}/{}".format(c, page)
            if isurl(urls):
                html = urlhelper(urls)
                soup = BeautifulSoup(html, "lxml")
                resp = soup.findAll('div', attrs={"class": "itemBlock"})
                for one in range(len(resp)):
                    try:
                        rows = OrderedDict()
                        post_info = resp[one]
                        labels_pre = post_info.findAll('a', attrs={'class': 'categoryLabel'})
                        labels = get_coinspeaker_labels(labels_pre)
                        date = post_info.findAll('div', attrs={'class': 'newsDate'})[0].text
                        summary = post_info.findAll('div', attrs={'class': 'newsExcerpt'})[0].text
                        title_pre = post_info.findAll('div', attrs={'class': 'newsTitle'})
                        title = title_pre[0].text
                        rows['title'] = title.replace("\n", "")
                        rows['labels'] = labels.replace("\n", "")
                        rows['date'] = get_month(date.replace("\n", ""))
                        rows['summary'] = summary.replace("\n", "")

                        link = title_pre[0].find('a', href=True)['href']
                        link_html = urlhelper(link)
                        link_soup = BeautifulSoup(link_html, "lxml")
                        content = link_soup.findAll('div', attrs={'class': 'entry-content'})
                        content = get_content(content)
                        rows['content'] = content.replace("\n", "")
                        save.append(rows)
                        print(rows)
                    except Exception as e:
                        logging.warning("------------e:{},{}".format(e, urls))
                        continue

            if page % 10 == 0:
                df = pd.DataFrame(save)
                df.to_csv("./datasets/www.ethnews.com_{}.csv".format(page), index=None)


def get_month(inputs):
    md = {"January": 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
          'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, "December": 12}
    s = inputs.replace(",", "").replace("th", "").split(' ')
    return "{}-{}-{}".format(s[2], md[s[0]], s[1])


def get_time_huobiglobal(inputs):
    return inputs.replace("T", " ").replace('Z', "")


def huobiglobal():
    # https://huobiglobal.zendesk.com/hc/zh-cn/sections/360000007051	//火币
    urls = "https://huobiglobal.zendesk.com/hc/zh-cn/sections/360000007051-%E9%87%8D%E8%A6%81%E5%85%AC%E5%91%8A"
    if isurl(urls):
        html = urlhelper(urls)
        soup = BeautifulSoup(html, "lxml")
        resp = soup.findAll('a', attrs={"class": "article-list-link"})
        save = []
        for one in range(len(resp)):
            try:
                rows = OrderedDict()
                title = resp[one].text
                rows['title'] = title
                link = "{}{}".format("https://huobiglobal.zendesk.com", resp[one]['href'])
                if isurl(link):
                    html_link = urlhelper(link)
                    soup_link = BeautifulSoup(html_link, "lxml")
                    rows['datetime'] = get_time_huobiglobal(soup_link.findAll('time')[0]['datetime'])
                    resp_link = soup_link.findAll('div', attrs={"class": "article-body"})[0].text
                    content = stringpro(resp_link)
                    rows['content'] = content
                    save.append(rows)
                    print(rows)
            except Exception as e:
                continue

        df = pd.DataFrame(save)
        df.to_csv("./datasets/huobiglobal.csv", index=None)


def bithumb_cafe():
    # http://bithumb.cafe/notice
    links_tmp = []
    browser = webdriver.Chrome()
    for page in range(1, 5):
        urls = "https://bithumb.cafe/notice/page/{}".format(page)
        browser.get(urls)
        links = browser.find_elements_by_xpath("//*[@href]")
        for link in links:
            try:
                rows = OrderedDict()
                link_next = link.get_attribute('href')
                if "archives" not in link_next:
                    continue
                links_tmp.append(link_next)
            except Exception as e:
                print("eeeeeeeeeeeeeeee:{}".format(e))
                continue

    print(len(links_tmp), links_tmp[:10])
    save = []

    for url in links_tmp:
        try:
            browser.get(url)
            print("正在抓取：{}".format(url))
            titles = browser.find_element_by_class_name(name='entry-title').text
            rows['title'] = titles
            author = browser.find_element_by_class_name(name='posted-author').text
            rows['author'] = author
            date = browser.find_element_by_class_name(name='posted-date').text
            rows['date'] = date
            tags = browser.find_element_by_class_name(name='posted-tags').text
            rows['tags'] = tags
            content = browser.find_element_by_class_name(name='entry-content').text
            rows['content'] = stringpro(content)
            save.append(rows)
            print(rows)
        except Exception as e:
            print("eeeeeeeeeeeeeeee:{}".format(e))
            continue
    df = pd.DataFrame(save)
    df.to_csv("./datasets/huobiglobal.csv", index=None)


def coinone_co_kr():
    # links_tmp = []
    # for page in range(1, 51):
    #     urls = "https://coinone.co.kr/api/talk/notice/?page={}&searchWord=&searchType=&ordering=-created_at".format(page)
    #     content = urlhelper(urls)
    #     print(content)
    url = 'https://coinone.co.kr/talk/notice'
    html = urlhelper(url)
    soup = BeautifulSoup(html, "lxml")
    resp = soup.findAll('article', attrs={"class": "card summary_card"})
    print(resp)


def coinone_info():
    #  https://twitter.com/coinone_info
    browser = webdriver.Chrome()
    username = "ypducdtu@163.com"
    passwd = "807429twirrer"
    browser.get('https://twitter.com')
    browser.implicitly_wait(10)
    elem = browser.find_element_by_name("session[username_or_email]")
    elem.send_keys(username)
    elem = browser.find_element_by_name("session[password]")
    elem.send_keys(passwd)
    elem = browser.find_element_by_xpath(
        '//*[@id="react-root"]/div/main/div/div/div[1]/div[1]/div[1]/form/div/div[3]/div')
    elem.click()
    url = 'https://twitter.com/coinone_info'
    browser.get(url)
    titles = browser.find_element_by_class_name(name='content')
    print(titles.text)


def info_tmp():
    import tweepy
    import json

    consumer_key = "BT1XlM8VTUnatdZgN4EQcaS8b"
    consumer_secret = "aZNbOHfbcoFAH0as5lhMqWK3QBuHq2R1cZYu6uZaQwPaGU3wH6"
    access_token = "1011973457263751168-0px2slGs7jlg9nxbelBRnuDPiK0CgI"
    access_token_secret = "4b5Co1wFdaDgjbF9MY9GZnkWiYo45hVW3UMprlL3OKSCI"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    search_results = api.search(q='python', count=20)
    # 对对象进行迭代
    for tweet in search_results:
        print(tweet)


def www_upbit_com():
    browser = webdriver.Chrome()
    save = []
    home = 'https://www.upbit.com/home'
    browser.get(home)
    time.sleep(15)
    for page in range(25, 50):
        try:
            rows = OrderedDict()
            url = "https://www.upbit.com/service_center/notice?id={}".format(page)
            browser.get(url)
            content = browser.find_element_by_class_name(name='txtB').text
            title_class = browser.find_element_by_class_name(name='titB')
            title = title_class.find_element_by_tag_name('strong').text
            times_str = title_class.find_element_by_tag_name('span').text
            times = times_str.split('|')[0].split(" ")[1:]
            num = times_str.split("|")[1].split(" ")[1]
            rows['title'] = title
            rows['times'] = " ".join(times)
            rows['num'] = num
            rows['content'] = stringpro(content)
            save.append(rows)
        except Exception as e:
            continue

    df = pd.DataFrame(save)
    df.to_csv("./datasets/www_upbit_com.csv", index=None)


def www_bitstamp_net():
    urls = 'https://www.bitstamp.net/news/'

    headers = {
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Referer': 'https://www.tianyancha.com/',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9'

    }
    links_tmp = []
    browser = webdriver.Chrome()
    k = 0
    step = 10000
    browser.get(urls)
    while True:
        k += 1
        try:
            xx = browser.get_cookies()
            names = []
            values = []
            for x in xx:
                names.append(x['name'])
                values.append(x['value'])
            cookies = dict(zip(names, values))
            response = requests.get(urls, headers=headers, cookies=cookies, timeout=60).json()
            print(response)

            if k % 10 == 0:
                browser.get(urls)
                step += 10000
                browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
                time.sleep(3)

        except Exception as e:
            print(e)
            print(k)
            continue


def blog_kraken_com():

    browser = webdriver.Chrome()
    url = 'https://blog.kraken.com/'
    browser.get(url)
    time.sleep(5)

    while True:
        time.sleep(5)
        button_class = browser.find_element_by_id(id_='infinite-handle')
        times_str = button_class.find_element_by_tag_name('button').click()
        html = urlhelper(url)
        soup = BeautifulSoup(html, "lxml")
        resp = soup.findAll('a', attrs={"class": "more-link"})
        print(len(resp), resp[-1])


if __name__ == '__main__':
    import sys

    # method = sys.argv[1]
    method = 'blog_kraken_com'
    if method == 'coindesk':
        # www.coindesk.com
        coindesk()

    if method == 'ethnews':
        # www.ethnews.com
        ethnews()

    if method == 'coinspeaker':
        # www.coinspeaker.com
        coinspeaker()

    if method == 'huobiglobal':
        huobiglobal()

    if method == 'coinone_co_kr':
        coinone_co_kr()

    if method == 'bithumb_cafe':
        bithumb_cafe()

    if method == 'coinone_info':
        coinone_info()

    if method == 'info_test':
        info_tmp()

    if method == 'www_upbit_com':
        www_upbit_com()

    if method == 'www_bitstamp_net':
        www_bitstamp_net()

    if method == 'blog_kraken_com':
        blog_kraken_com()

